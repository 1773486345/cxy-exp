import os
import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ts_benchmark.baselines.mnrf_lite_v2.backbone import CausalResidualTCN
from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2
from ts_benchmark.baselines.mnrf_lite_v2.training import EndWindowDataset
from ts_benchmark.baselines.mnrf_lite_v2.views import FourViewEncoder, mask_pattern_values
from ts_benchmark.baselines.mnrf_lite_v2.windows import continuous_segments, intersect_segments


@pytest.fixture(scope="session", autouse=True)
def fixed_cpu_threads():
    torch.set_num_threads(1)
    yield


def test_timestamp_gap_exact_window_endpoints_and_cold_starts():
    times = pd.date_range("2024-01-01", periods=400, freq="min").to_numpy()
    times[200:] += np.timedelta64(1, "D")
    segments, audit = continuous_segments(times, 400)
    assert segments == [(0, 200), (200, 400)]
    assert audit["basis"] == "timestamp" and len(audit["gaps"]) == 1
    values = np.zeros((400, 2), dtype=np.float32)
    dataset = EndWindowDataset(values, 0, 400, 128, segments)
    assert np.array_equal(dataset.endpoints, np.r_[127:200, 327:400])


def test_timestamp_order_duplicates_irregular_and_row_order_policy():
    regular = pd.date_range("2024-01-01", periods=8, freq="h").to_numpy()
    assert continuous_segments(regular, 8)[0] == [(0, 8)]
    duplicate = regular.copy()
    duplicate[3] = duplicate[2]
    with pytest.raises(ValueError, match="duplicate or reverse"):
        continuous_segments(duplicate, 8)
    reverse = regular.copy()
    reverse[3] = reverse[2] - np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="duplicate or reverse"):
        continuous_segments(reverse, 8)
    odd = regular.copy()
    odd[4:] += np.timedelta64(17, "m")
    with pytest.raises(ValueError, match="unexplained timestamp interval"):
        continuous_segments(odd, 8)
    segments, audit = continuous_segments(np.arange(8), 8)
    assert segments == [(0, 8)] and audit["basis"] == "row_order"


def test_entity_boundary_also_starts_a_new_segment():
    times = pd.date_range("2024-01-01", periods=8, freq="h").to_numpy()
    segments, audit = continuous_segments(times, 8, boundary_ids=["a"] * 4 + ["b"] * 4)
    assert segments == [(0, 4), (4, 8)]
    assert audit["entity_boundary_rows"] == [4]


def test_short_segment_and_internal_purge_intersection():
    segments = [(0, 20), (25, 70), (80, 130)]
    assert EndWindowDataset(np.zeros((130, 2)), 0, 130, 32, segments).endpoints.tolist() == list(range(56, 70)) + list(range(111, 130))
    assert intersect_segments(segments, 30, 100) == [(30, 70), (80, 100)]


def test_tcn_has_six_block_level_residuals_and_fixed_receptive_field():
    torch.manual_seed(7)
    tcn = CausalResidualTCN()
    x = torch.randn(2, 40, 2, 32)
    block_outputs = []
    handles = [block.register_forward_hook(lambda module, args, output: block_outputs.append(output)) for block in tcn.blocks]
    actual = tcn(x)
    for handle in handles:
        handle.remove()
    hidden = x.permute(0, 2, 3, 1).reshape(4, 32, 40)
    for block in tcn.blocks:
        expected = hidden + block.layers(hidden)
        hidden = expected
    expected = hidden.reshape(2, 2, 32, 40).permute(0, 3, 1, 2)
    assert len(block_outputs) == 6
    assert tcn.receptive_field == 253
    assert torch.allclose(actual, expected)


def test_relation_attention_parallel_equals_per_target_reference_and_masks_self():
    torch.manual_seed(12)
    encoder = FourViewEncoder(variables=2)
    source = torch.randn(3, 2, 32)
    identity = torch.randn(3, 2, 32)
    actual = encoder._relation_from_source(source, identity)
    query = encoder.relation_query(identity).reshape(6, 1, 32)
    key_value = source.unsqueeze(1).expand(-1, 2, -1, -1).reshape(6, 2, 32)
    blocked_self = torch.eye(2, dtype=torch.bool).repeat(3, 1)
    attended, _ = encoder.relation_attention(query, key_value, key_value, key_padding_mask=blocked_self, need_weights=False)
    expected = encoder.relation_adapter(attended[:, 0].reshape(3, 2, 32))
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    changed = source.clone()
    changed[:, 0] += 1000
    before = encoder._relation_from_source(source, identity)[:, 0]
    after = encoder._relation_from_source(changed, identity)[:, 0]
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-6)


def test_relation_attention_parallel_parameter_gradients_match_reference():
    torch.manual_seed(22)
    parallel = FourViewEncoder(variables=3)
    reference = copy.deepcopy(parallel)
    source_parallel = torch.randn(2, 3, 32, requires_grad=True)
    identity_parallel = torch.randn(2, 3, 32, requires_grad=True)
    source_reference = source_parallel.detach().clone().requires_grad_(True)
    identity_reference = identity_parallel.detach().clone().requires_grad_(True)
    actual = parallel._relation_from_source(source_parallel, identity_parallel)
    actual.square().sum().backward()
    query = reference.relation_query(identity_reference).unsqueeze(2)
    repeated_source = source_reference.unsqueeze(1).expand(-1, 3, -1, -1)
    flat_query = query.reshape(-1, 1, 32)
    flat_source = repeated_source.reshape(-1, 3, 32)
    self_mask = torch.eye(3, dtype=torch.bool).repeat(2, 1)
    attended, _ = reference.relation_attention(flat_query, flat_source, flat_source,
                                               key_padding_mask=self_mask, need_weights=False)
    expected = reference.relation_adapter(attended[:, 0].reshape(2, 3, 32))
    expected.square().sum().backward()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(source_parallel.grad, source_reference.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(identity_parallel.grad, identity_reference.grad, atol=1e-6, rtol=1e-6)
    for actual_parameter, expected_parameter in zip(parallel.parameters(), reference.parameters()):
        if actual_parameter.grad is not None:
            assert torch.allclose(actual_parameter.grad, expected_parameter.grad, atol=1e-6, rtol=1e-6)


def test_vectorized_pattern_mask_matches_reference_and_preserves_positions():
    torch.manual_seed(31)
    local = torch.randn(3, 32, 4)
    valid = torch.rand(3, 32, 4) > 0.1
    starts = torch.tensor([[0, 3, 12, 24], [1, 7, 15, 20], [4, 9, 18, 23]])
    actual, actual_valid, target, positions = mask_pattern_values(local, valid, starts)
    expected = local.clone()
    expected_valid = valid.clone()
    expected_target = torch.empty(3, 4, 8)
    for n in range(3):
        for d in range(4):
            start = int(starts[n, d])
            expected_target[n, d] = local[n, start:start + 8, d]
            expected[n, start:start + 8, d] = 0
            expected_valid[n, start:start + 8, d] = False
    expected_positions = (starts.unsqueeze(-1) + torch.arange(8).view(1, 1, 8)) / 31
    assert torch.equal(actual, expected)
    assert torch.equal(actual_valid, expected_valid)
    assert torch.equal(target, expected_target)
    assert torch.equal(positions, expected_positions.float())


def test_sequential_backend_preserves_mnrf_device_mask(monkeypatch):
    from ts_benchmark.utils.parallel.sequential_backend import SequentialBackend
    from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("TS_BENCHMARK_PRESERVE_CUDA_VISIBLE_DEVICES", "1")
    backend = SequentialBackend(gpu_devices=None)
    backend.init()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: original_empty(0))
    model = MNRFLiteV2(device="cuda:0")
    assert str(model.device) == "cuda:0"
    backend.close()
    runner = Path(__file__).resolve().parents[4] / "scripts/run_mnrf_lite_v2_task.sh"
    source = runner.read_text(encoding="utf-8")
    assert "TS_BENCHMARK_PRESERVE_CUDA_VISIBLE_DEVICES=1" in source
    assert "--gpus" not in source
    assert 'DEVICE="cuda:0"' in source and 'DEVICE="cpu"' in source


def test_encoder_batches_never_exceed_train_batch_and_cache_chunk_is_io_only(tmp_path):
    from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig
    from ts_benchmark.baselines.mnrf_lite_v2.training import MNRFTrainer

    config = MNRFConfig(seq_len=32, batch_size=3, encoding_batch_size=3, cache_chunk_size=2)
    trainer = MNRFTrainer(config, 2, torch.device("cpu"))
    values = np.random.default_rng(5).normal(size=(70, 2)).astype(np.float32)
    trainer.fixed_input.fit(values)
    observed = []
    original = trainer.encoder.encode_views

    def counted(batch, *args, **kwargs):
        observed.append(len(batch))
        return original(batch, *args, **kwargs)

    trainer.encoder.encode_views = counted
    trainer.extract_cache(values, (0, 70), "smoke", tmp_path)
    assert max(observed) == 3
    assert sum(observed) == len(EndWindowDataset(values, 0, 70, 32))


def test_run_validator_rejects_missing_artifacts(tmp_path):
    from scripts.validate_mnrf_lite_v2_run import main as validate_run

    with pytest.raises(SystemExit, match="Missing required run artifacts"):
        validate_run(str(tmp_path), "toy")


def test_dedicated_strategy_to_benchmark_report_and_run_validator(monkeypatch, tmp_path):
    from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig
    import ts_benchmark.evaluation.strategy.mnrf_valid_detect as strategy_module
    from ts_benchmark.evaluation.evaluator import Evaluator
    from ts_benchmark.report import report

    n, train_n = 446, 382
    times = pd.date_range("2025-01-01", periods=n, freq="min")
    times = times.to_numpy()
    times[414:] += np.timedelta64(2, "D")
    rng = np.random.default_rng(991)
    frame = pd.DataFrame(rng.normal(size=(n, 2)).astype(np.float32), columns=["a", "b"], index=times)
    labels = np.zeros(n, dtype=np.int64)
    labels[445] = 1
    frame["label"] = labels

    class Pool:
        def get_series(self, name):
            return frame

        def get_series_meta_info(self, name):
            return pd.Series({"train_lens": train_n})

    class FakeDataPool:
        def get_pool(self):
            return Pool()

    monkeypatch.setattr(strategy_module, "DataPool", FakeDataPool)
    run_dir = tmp_path / "result" / "run-integration"
    run_dir.parent.mkdir()
    model_config = dict(seq_len=32, batch_size=512, encoding_batch_size=0,
                        cache_chunk_size=256, stage1_epochs=1, stage2_epochs=1,
                        patience=1, device="cpu", task_name="toy.csv", output_dir=str(run_dir))
    model_factory = lambda: MNRFLiteV2(**model_config)
    evaluator = Evaluator([{"name": "auc_roc"}, {"name": "auc_pr"}])
    strategy = strategy_module.MNRFValidDetect({"strategy_name": "mnrf_valid_detect"}, evaluator)
    result = strategy.execute("toy.csv", model_factory)[0]
    metric_payload = __import__("json").loads(next((run_dir / "results").glob("*_metrics.json")).read_text())
    aligned_path = next((run_dir / "results").glob("*_aligned_scores.npz"))
    with np.load(aligned_path) as aligned:
        assert np.array_equal(aligned["local_time_indices"], np.array([31, 63]))
        assert np.all(aligned["time_indices"] == aligned["local_time_indices"] + train_n)
        assert len(aligned["scores"]) == len(aligned["predictions"]) == 2
    assert metric_payload["status"] == "complete"

    record_path = tmp_path / "integration_record.csv"
    report_config = {
        "log_files_list": [str(record_path)],
        "report_metrics": ["auc_roc", "auc_pr"], "aggregate_type": "mean",
        "fill_type": "mean_value", "null_value_threshold": 0.3,
        "save_path": "run-integration",
        "leaderboard_file_name": "test_report_integration.csv",
    }
    pd.DataFrame([{
            "strategy_args": "mnrf_valid_detect", "model_name": "MNRFLiteV2",
            "model_params": "tiny-cpu", "file_name": "toy.csv",
            "auc_roc": result[0], "auc_pr": result[1],
            "actual_data": "", "inference_data": "", "log_info": "",
        }]).to_csv(record_path, index=False)
    assert isinstance(report_config["log_files_list"], list)
    import ts_benchmark.report.report_csv as report_csv
    monkeypatch.setattr(report_csv, "ROOT_PATH", str(tmp_path))
    report(report_config)
    from scripts.validate_mnrf_lite_v2_run import main as validate_run
    validate_run(str(run_dir), "toy")
    deployed = MNRFLiteV2(device="cpu").load_checkpoint(run_dir / "deployment.pt")
    scored_again = deployed.score_with_metadata(frame.iloc[train_n:].drop(columns="label"), index_offset=train_n)
    with np.load(aligned_path) as aligned:
        assert np.allclose(scored_again["scores"], aligned["scores"], atol=1e-6, rtol=1e-6)
