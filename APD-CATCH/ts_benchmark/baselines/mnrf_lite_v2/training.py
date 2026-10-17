"""Purged splits, two-stage optimization, frozen feature cache and calibration."""

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import MNRFConfig
from .fusion import AlternatingAffineFlow
from .normalization import FixedFeatureStandardizer, FixedInputStandardizer
from .views import FourViewEncoder
from .windows import intersect_segments, window_endpoints


def purged_ranges(length, window):
    gap = window - 1
    available = int(length) - 2 * gap
    if available <= 0:
        raise ValueError(f"training segment length {length} is too short for two {gap}-point purge gaps")
    train_n = int(math.floor(available * 0.8))
    valid_n = int(math.floor(available * 0.1))
    calibration_n = available - train_n - valid_n
    if min(train_n, valid_n, calibration_n) < window:
        raise ValueError(
            f"training segment length {length} cannot produce train/validation/calibration "
            f"windows of length {window} after {gap}-point purge gaps"
        )
    train = (0, train_n)
    validation = (train_n + gap, train_n + gap + valid_n)
    calibration = (validation[1] + gap, validation[1] + gap + calibration_n)
    return {
        "gap": gap,
        "input_length": int(length),
        "train": train,
        "validation": validation,
        "calibration": calibration,
        "window_counts": {
            name: end - start - window + 1
            for name, (start, end) in (
                ("train", train), ("validation", validation), ("calibration", calibration)
            )
        },
    }


class EndWindowDataset(Dataset):
    def __init__(self, values, start, end, window, segments=None):
        self.values = values
        self.start, self.end, self.window = int(start), int(end), int(window)
        selected = segments if segments is not None else [(self.start, self.end)]
        self.endpoints = window_endpoints(intersect_segments(selected, self.start, self.end), self.window)

    def __len__(self):
        return len(self.endpoints)

    def __getitem__(self, index):
        endpoint = int(self.endpoints[index])
        return torch.from_numpy(np.array(self.values[endpoint - self.window + 1:endpoint + 1], dtype=np.float32, copy=True)), endpoint


class GaussianAuxiliaryDecoder(nn.Module):
    def __init__(self, variables, hidden=32, view_dim=8):
        super().__init__()
        self.task_identity = nn.Parameter(torch.randn(3, 8) * 0.02)
        self.decoder = nn.Sequential(nn.Linear(view_dim + 8 + hidden + 1, 64), nn.GELU(), nn.Linear(64, 2))

    def loss(self, z, target, relative_position, variable_identity, task_index):
        batch, variables, width = z.shape
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        if relative_position.ndim == 2:
            positions = relative_position.unsqueeze(-1).unsqueeze(-1)
        elif relative_position.ndim == 3:
            positions = relative_position.unsqueeze(-1)
        else:
            raise ValueError("relative_position must have shape [N,D] or [N,D,K]")
        count = target.size(-1)
        z_rep = z.unsqueeze(2).expand(-1, -1, count, -1)
        task = self.task_identity[task_index].view(1, 1, 1, -1).expand(batch, variables, count, -1)
        identity = variable_identity.view(1, variables, 1, -1).expand(batch, -1, count, -1)
        decoder_input = torch.cat((z_rep, task, identity, positions), dim=-1)
        output = self.decoder(decoder_input)
        mean = output[..., 0]
        sigma = 0.05 + torch.nn.functional.softplus(output[..., 1])
        nll = (target - mean).square() / (2.0 * sigma.square()) + torch.log(sigma) + 0.5 * math.log(2.0 * math.pi)
        return nll.mean()


class MNRFTrainer:
    def __init__(self, config: MNRFConfig, variables: int, device: torch.device):
        self.config = config
        self.variables = variables
        self.device = device
        self.encoder = FourViewEncoder(variables, config.hidden_dim, config.view_dim).to(device)
        self.auxiliary = GaussianAuxiliaryDecoder(variables).to(device)
        self.flow = AlternatingAffineFlow(config.hidden_dim).to(device)
        self.fixed_input = FixedInputStandardizer()
        self.fixed_feature = FixedFeatureStandardizer(config.feature_std_floor)
        self.noise_vectors = None
        self.calibration_center = None
        self.calibration_scale = None
        self.threshold = None
        self.history = {"stage1": [], "stage2": []}
        self.split_info = None
        self.cache_sha256 = {}

    def _windows(self, values, segment, shuffle, seed=None, segments=None):
        start, end = segment
        dataset = EndWindowDataset(values, start, end, self.config.seq_len, segments=segments)
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0,
            generator=generator,
        )

    def _pretrain_batch(self, windows, task, endpoints=None, fixed_seed=None):
        values = windows.to(self.device)
        valid = torch.ones_like(values, dtype=torch.bool)
        mask_start = None
        if task == "pattern":
            if fixed_seed is None:
                mask_start = torch.randint(0, 25, (values.size(0), self.variables), device=self.device)
            else:
                ep = torch.as_tensor(endpoints, device=self.device).view(-1, 1)
                variable = torch.arange(self.variables, device=self.device).view(1, -1)
                mask_start = ((ep * 1103515245 + variable * 12345 + int(fixed_seed)) % 25).long()
        z, target, relative = self.encoder.encode_pretrain(values, valid, task, mask_start=mask_start)
        task_index = {"evolution": 0, "pattern": 1, "relation": 2}[task]
        return self.auxiliary.loss(z, target, relative, self.encoder.point.variable_identity, task_index)

    @torch.no_grad()
    def _validation_stage1(self, values, segment, segments=None):
        self.encoder.eval()
        self.auxiliary.eval()
        loader = self._windows(values, segment, shuffle=False, segments=segments)
        per_task = {name: [0.0, 0] for name in ("evolution", "pattern", "relation")}
        for windows, endpoints in loader:
            endpoints_np = endpoints.numpy()
            for task in per_task:
                loss = self._pretrain_batch(windows, task, endpoints_np, fixed_seed=self.config.seed)
                per_task[task][0] += float(loss.cpu()) * len(endpoints_np)
                per_task[task][1] += len(endpoints_np)
        task_losses = [total / count for total, count in per_task.values()]
        self.encoder.train()
        self.auxiliary.train()
        return float(np.mean(task_losses))

    def stage1(self, values, train_segment, validation_segment, out_dir, segments=None, log_path=None, resume_id=None):
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.auxiliary.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        train_loader = self._windows(values, train_segment, shuffle=True, seed=self.config.seed, segments=segments)
        best = float("inf")
        best_state = None
        patience = 0
        start_epoch = 0
        completed_steps = 0
        latest_path = Path(out_dir) / "stage1_latest.pt"
        if self.config.resume_from and not latest_path.is_file():
            raise FileNotFoundError("explicit resume run has no stage1 epoch checkpoint")
        if self.config.resume_from and latest_path.is_file():
            state = torch.load(latest_path, map_location="cpu", weights_only=False)
            _validate_resume_state(state, "stage1", resume_id)
            self.encoder.load_state_dict(state["encoder"])
            self.auxiliary.load_state_dict(state["auxiliary"])
            optimizer.load_state_dict(state["optimizer"])
            train_loader.generator.set_state(state["loader_rng"])
            _restore_rng(state["rng"])
            start_epoch, best, best_state, patience = state["next_epoch"], state["best"], state["best_state"], state["patience"]
            completed_steps = state.get("completed_steps", 0)
            self.history["stage1"] = state.get("history", [])
        for epoch in range(start_epoch, self.config.stage1_epochs):
            epoch_start = time.perf_counter()
            self.encoder.train()
            self.auxiliary.train()
            task_losses = []
            for windows, _ in train_loader:
                task = random.choice(("evolution", "pattern", "relation"))
                optimizer.zero_grad(set_to_none=True)
                loss = self._pretrain_batch(windows, task)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite stage1 {task} loss")
                loss.backward()
                optimizer.step()
                task_losses.append(float(loss.detach().cpu()))
                completed_steps += 1
            validation_loss = self._validation_stage1(values, validation_segment, segments=segments)
            self.history["stage1"].append({
                "epoch": epoch + 1,
                "steps": len(task_losses),
                "train_nll": float(np.mean(task_losses)),
                "validation_macro_nll": validation_loss,
            })
            if validation_loss < best:
                best = validation_loss
                best_state = {
                    "encoder": {k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()},
                    "auxiliary": {k: v.detach().cpu().clone() for k, v in self.auxiliary.state_dict().items()},
                }
                atomic_torch_save(best_state, Path(out_dir) / "best_encoder.pt")
                atomic_torch_save(best_state, Path(out_dir) / "best_stage1.pt")
                patience = 0
            else:
                patience += 1
            checkpoint = {
                "stage": "stage1", "resume_id": resume_id, "next_epoch": epoch + 1,
                "encoder": self.encoder.state_dict(), "auxiliary": self.auxiliary.state_dict(),
                "optimizer": optimizer.state_dict(), "loader_rng": train_loader.generator.get_state(),
                "rng": _capture_rng(), "best": best, "best_state": best_state,
                "patience": patience, "history": self.history["stage1"], "completed_steps": completed_steps,
            }
            atomic_torch_save(checkpoint, latest_path)
            record = {**self.history["stage1"][-1], "stage": "stage1", "completed_steps": completed_steps, "best_validation": best,
                      "patience": patience, "elapsed_seconds": time.perf_counter() - epoch_start}
            _append_log(log_path, record)
            print(f"MNRF stage1 epoch={epoch + 1} epoch_steps={record['steps']} completed_steps={completed_steps} train_nll={record['train_nll']:.6g} "
                  f"validation={validation_loss:.6g} best={best:.6g} patience={patience} "
                  f"seconds={record['elapsed_seconds']:.2f}", flush=True)
            if patience >= self.config.patience:
                break
        if best_state is None:
            raise RuntimeError("stage1 did not produce a finite validation checkpoint")
        self.encoder.load_state_dict(best_state["encoder"])
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.encoder.eval()
        # The temporary auxiliary decoder is deliberately excluded from deployment.
        del self.auxiliary
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        return best

    @torch.no_grad()
    def extract_cache(self, values, segment, name, cache_dir, segments=None):
        start, end = segment
        dataset = EndWindowDataset(values, start, end, self.config.seq_len, segments=segments)
        count = len(dataset)
        path = Path(cache_dir) / f"{name}_features.npy"
        temporary_path = path.with_name(path.name + f".partial-{os.getpid()}")
        indices_path = Path(cache_dir) / f"{name}_window_end_indices.npy"
        tmp_indices = str(indices_path) + ".tmp.npy"
        np.save(tmp_indices, dataset.endpoints)
        os.replace(tmp_indices, indices_path)
        mmap = np.lib.format.open_memmap(
            temporary_path, mode="w+", dtype=np.float32,
            shape=(count, self.variables, 25),
        )
        self.encoder.eval()
        cursor = 0
        encode_batch = min(self.config.batch_size, self.config.encoding_batch_size or self.config.batch_size)
        io_chunks = max(1, math.ceil(count / self.config.cache_chunk_size))
        next_progress = min(count, self.config.cache_chunk_size)
        io_number = 1
        for offset in range(0, count, encode_batch):
            batch = [dataset[i][0] for i in range(offset, min(count, offset + encode_batch))]
            tensor = torch.stack(batch).to(self.device)
            feature = self.encoder.encode_views(tensor)["features"]
            size = feature.size(0)
            feature_array = np.ascontiguousarray(feature.cpu().numpy())
            mmap[cursor:cursor + size] = feature_array
            cursor += size
            if cursor >= next_progress or cursor == count:
                print(f"MNRF cache={name} windows={cursor}/{count} io_chunk={io_number}/{io_chunks} "
                      f"gpu_batch<={encode_batch}", flush=True)
                io_number += 1
                next_progress = min(count, next_progress + self.config.cache_chunk_size)
        mmap.flush()
        del mmap
        os.replace(temporary_path, path)
        self.cache_sha256[name] = _file_sha256(path)
        return str(path), count, str(indices_path)

    def _fixed_noises(self):
        if self.noise_vectors is None:
            generator = torch.Generator(device="cpu").manual_seed(self.config.noise_seed)
            self.noise_vectors = torch.randn(4, 25, generator=generator) * self.config.train_noise_std
            self.noise_vectors = self.noise_vectors.to(self.device)
        return self.noise_vectors

    def _flow_nll(self, features, noise=None):
        identity = self.encoder.point.variable_identity
        if noise is not None:
            features = features + noise.view(1, 1, -1)
        return self.flow(features, identity)["nll"]

    @torch.no_grad()
    def _normalized_feature_batch(self, features, indices):
        batch = np.asarray(features[indices], dtype=np.float32)
        return (batch - self.fixed_feature.mean_) / self.fixed_feature.scale_

    def _stage2_validation(self, features):
        self.flow.eval()
        noises = self._fixed_noises()
        total_loss = 0.0
        total_count = 0
        for offset in range(0, len(features), self.config.batch_size):
            batch = torch.as_tensor(self._normalized_feature_batch(
                features, slice(offset, offset + self.config.batch_size)
            ), device=self.device)
            values = [self._flow_nll(batch, noise).mean() for noise in noises]
            size = len(batch)
            total_loss += float(torch.stack(values).mean().cpu()) * size
            total_count += size
        self.flow.train()
        return total_loss / max(total_count, 1)

    def stage2(self, train_features, validation_features, out_dir, log_path=None, resume_id=None):
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.encoder.eval()
        optimizer = torch.optim.AdamW(
            self.flow.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        best = float("inf")
        best_state = None
        patience = 0
        completed_steps = 0
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed + 17)
        self._fixed_noises()
        start_epoch = 0
        latest_path = Path(out_dir) / "stage2_latest.pt"
        if self.config.resume_from and latest_path.is_file():
            state = torch.load(latest_path, map_location="cpu", weights_only=False)
            _validate_resume_state(state, "stage2", resume_id)
            self.flow.load_state_dict(state["flow"])
            optimizer.load_state_dict(state["optimizer"])
            generator.set_state(state["sampler_rng"])
            _restore_rng(state["rng"])
            start_epoch, best, best_state, patience = state["next_epoch"], state["best"], state["best_state"], state["patience"]
            completed_steps = state.get("completed_steps", 0)
            self.history["stage2"] = state.get("history", [])
        for epoch in range(start_epoch, self.config.stage2_epochs):
            epoch_start = time.perf_counter()
            self.flow.train()
            indices = torch.randperm(len(train_features), generator=generator).numpy()
            losses = []
            for start in range(0, len(indices), self.config.batch_size):
                selection = indices[start:start + self.config.batch_size]
                feature = torch.as_tensor(self._normalized_feature_batch(train_features, selection), device=self.device)
                noise = torch.randn(
                    feature.size(0), 1, 25, device=self.device
                ) * self.config.train_noise_std
                optimizer.zero_grad(set_to_none=True)
                result = self.flow(feature + noise, self.encoder.point.variable_identity)
                loss = result["nll"].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite stage2 flow NLL")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                completed_steps += 1
            validation_loss = self._stage2_validation(validation_features)
            self.history["stage2"].append({
                "epoch": epoch + 1,
                "steps": len(losses),
                "train_nll": float(np.mean(losses)),
                "validation_fixed_noise_nll": validation_loss,
            })
            if validation_loss < best:
                best = validation_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.flow.state_dict().items()}
                atomic_torch_save(best_state, Path(out_dir) / "best_fusion.pt")
                patience = 0
            else:
                patience += 1
            checkpoint = {
                "stage": "stage2", "resume_id": resume_id, "next_epoch": epoch + 1,
                "flow": self.flow.state_dict(), "optimizer": optimizer.state_dict(),
                "sampler_rng": generator.get_state(), "rng": _capture_rng(),
                "best": best, "best_state": best_state, "patience": patience,
                "history": self.history["stage2"], "completed_steps": completed_steps,
            }
            atomic_torch_save(checkpoint, latest_path)
            record = {**self.history["stage2"][-1], "stage": "stage2", "completed_steps": completed_steps, "best_validation": best,
                      "patience": patience, "elapsed_seconds": time.perf_counter() - epoch_start}
            _append_log(log_path, record)
            print(f"MNRF stage2 epoch={epoch + 1} epoch_steps={record['steps']} completed_steps={completed_steps} train_nll={record['train_nll']:.6g} "
                  f"validation={validation_loss:.6g} best={best:.6g} patience={patience} "
                  f"seconds={record['elapsed_seconds']:.2f}", flush=True)
            if patience >= self.config.patience:
                break
        if best_state is None:
            raise RuntimeError("stage2 did not produce a finite validation checkpoint")
        self.flow.load_state_dict(best_state)
        return best

    @torch.no_grad()
    def per_variable_scores(self, features, output_path=None):
        noises = self._fixed_noises()
        output = None
        if output_path is not None:
            output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32,
                                                shape=(len(features), self.variables))
        results = [] if output is None else None
        self.encoder.eval()
        self.flow.eval()
        for offset in range(0, len(features), self.config.batch_size):
            batch = torch.as_tensor(self._normalized_feature_batch(
                features, slice(offset, offset + self.config.batch_size)
            ), device=self.device)
            scores = torch.stack([self._flow_nll(batch, noise) for noise in noises]).mean(dim=0)
            values = scores.cpu().numpy()
            if output is None:
                results.append(values)
            else:
                output[offset:offset + len(values)] = values
        if output is not None:
            output.flush()
            return output
        return np.concatenate(results, axis=0)

    def calibrate(self, validation_features, calibration_features, work_dir=None):
        work = Path(work_dir or ".")
        work.mkdir(parents=True, exist_ok=True)
        validation_scores = self.per_variable_scores(validation_features, work / "validation_nll.npy")
        self.calibration_center = np.asarray([np.median(validation_scores[:, d]) for d in range(self.variables)], dtype=np.float32)
        mad = np.asarray([np.median(np.abs(validation_scores[:, d] - self.calibration_center[d]))
                          for d in range(self.variables)], dtype=np.float32)
        self.calibration_scale = np.maximum(1.4826 * mad, self.config.score_mad_floor).astype(np.float32)
        calibration_scores = self.per_variable_scores(calibration_features, work / "calibration_nll.npy")
        aggregate = np.empty(len(calibration_scores), dtype=np.float32)
        for start in range(0, len(aggregate), self.config.batch_size):
            stop = min(len(aggregate), start + self.config.batch_size)
            aggregate[start:stop] = np.max(
                (calibration_scores[start:stop] - self.calibration_center) / self.calibration_scale, axis=1
            )
        self.threshold = float(np.quantile(aggregate, self.config.calibration_quantile))

    def fingerprint(self):
        digest = hashlib.sha256()
        digest.update(json.dumps({"variables": self.variables, "hidden_dim": self.config.hidden_dim,
                                 "view_dim": self.config.view_dim, "seq_len": self.config.seq_len,
                                 "pattern_len": self.config.pattern_len}, sort_keys=True).encode())
        for key, value in sorted(self.encoder.state_dict().items()):
            digest.update(key.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        for value in (self.fixed_input.mean_, self.fixed_input.scale_):
            digest.update(np.asarray(value, dtype=np.float32).tobytes())
        return digest.hexdigest()


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _capture_rng():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _validate_resume_state(state, stage, resume_id):
    if state.get("stage") != stage or state.get("resume_id") != resume_id:
        raise ValueError(f"{stage} checkpoint is incompatible with current run identity")


def _append_log(path, record):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
