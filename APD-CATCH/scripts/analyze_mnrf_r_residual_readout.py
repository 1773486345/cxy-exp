#!/usr/bin/env python
"""Fixed PSM/Creditcard R-readout analysis; saved arrays only, no models/fit.

Writes just README.md and results.csv to a new directory. Budget thresholds
are post-hoc test-label diagnostics; saved calibration thresholds stay fixed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "PSM": ("full_gmm-full-20261024T115119Z-3976714", 87650),
    "Creditcard": ("full_gmm-full-20261024T115131Z-3977864", 142213),
}
COMMON = ROOT / "result/score/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun"


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(value, message):
    if not value:
        raise ValueError(message)


def strict_axis(values):
    a = np.asarray(values)
    check(a.ndim == 1 and a.dtype.kind in "iu" and np.all(np.diff(a) > 0),
          "coordinates must be sorted unique integers")
    return a


def deciles(official, coordinates):
    """Split the official axis, never rebase onto the compact scored axis."""
    official, coordinates = strict_axis(official), strict_axis(coordinates)
    pos = np.searchsorted(official, coordinates)
    check(np.all(pos < len(official)) and np.array_equal(official[pos], coordinates),
          "coordinates outside the official test axis")
    edges = np.arange(11, dtype=np.int64) * len(official) // 10
    return np.searchsorted(edges, pos, side="right"), edges


def budget_point(scores, labels, fp_cap):
    """All tied scores move together; max TP, then min FP, strict >."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    check(scores.ndim == labels.ndim == 1 and scores.shape == labels.shape
          and len(scores) > 0 and np.isfinite(scores).all()
          and np.isin(labels, [0, 1]).all(), "invalid score/label vector")
    check(isinstance(fp_cap, (int, np.integer)) and fp_cap >= 0, "invalid FP cap")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    tp = np.r_[0, np.cumsum(labels[order], dtype=np.int64)[ends]]
    fp = np.r_[0, np.cumsum(1 - labels[order], dtype=np.int64)[ends]]
    eligible = np.flatnonzero(fp <= fp_cap)
    chosen = eligible[np.lexsort((fp[eligible], -tp[eligible]))[0]]
    if chosen == 0:
        threshold = float(sorted_scores[0])
    else:
        end = ends[chosen - 1]
        threshold = (float(sorted_scores[end + 1]) if end + 1 < len(scores)
                     else float(np.nextafter(sorted_scores[-1], -np.inf)))
    prediction = scores > threshold
    check(int(np.sum(prediction & (labels == 1))) == tp[chosen]
          and int(np.sum(prediction & (labels == 0))) == fp[chosen],
          "strict threshold does not reproduce selected tie group")
    return threshold, prediction


def stats(labels, prediction, scores):
    tp = int(np.sum((labels == 1) & prediction))
    fp = int(np.sum((labels == 0) & prediction))
    positive = int(np.sum(labels == 1))
    negative = len(labels) - positive
    return dict(points=len(labels), positive_points=positive, normal_points=negative,
                tp=tp, fp=fp, fn=positive - tp, tn=negative - fp,
                recall=tp / positive if positive else None,
                fpr=fp / negative if negative else None,
                precision=tp / (tp + fp) if tp + fp else None,
                ap=float(average_precision_score(labels, scores)) if positive else None,
                roc_auc=float(roc_auc_score(labels, scores)) if positive and negative else None)


class Inputs:
    """Hash only files consumed here, not feature caches or model weights."""
    def __init__(self):
        self.hashes = {}

    def record(self, path, expected=None):
        path = Path(path)
        actual = sha(path)
        check(expected is None or actual == expected, f"digest mismatch: {path}")
        self.hashes[path] = actual

    def json(self, path):
        self.record(path)
        return read_json(path)

    def unchanged(self):
        for path, before in self.hashes.items():
            check(sha(path) == before, f"input changed during analysis: {path}")


def load_task(task, inputs):
    rid, expected_count = SOURCES[task]
    source = ROOT / "result/score/MNRF-ViewProbe" / task / rid
    diagnostic = ROOT / "result/analysis/MNRF-FrozenDecoder" / task / rid
    manifest = inputs.json(source / "probe_manifest.json")
    receipt = inputs.json(diagnostic / "diagnostic_manifest.json")
    check(manifest["stage_status"] == manifest["head_status"] == "complete", "incomplete Full")
    check(Path(receipt["source_run"]).resolve() == source.resolve(), "wrong decoder source")
    registration = receipt["source_registration"]
    for key in ("feature_fingerprint", "source_identity_sha256", "cache_sha256",
                "endpoint_sha256", "gmm_fit_endpoints_sha256"):
        check(registration[key] == manifest[key], f"decoder registration differs: {key}")
    check(registration["run_id"] == rid and receipt["calibration_quantile"] == 0.995,
          "wrong run or calibration protocol")
    inputs.record(source / "analysis_inputs.npz", manifest["analysis_artifacts"]["analysis_inputs_sha256"])
    with np.load(source / "analysis_inputs.npz", allow_pickle=False) as a:
        axes = {key: a[key] for key in a.files}
    official = strict_axis(axes["test_global_indices"])
    complete = strict_axis(axes["all_global_indices"])
    lookup = np.searchsorted(complete, official)
    check(np.all(lookup < len(complete)) and np.array_equal(complete[lookup], official),
          "test axis outside full axis")
    for field in ("labels", "time_values_ns", "segment_ids"):
        check(np.array_equal(axes[f"test_{field}"], axes[f"all_{field}"][lookup]),
              f"test/full axis mismatch: {field}")
    path = source / "results/aligned_scores.npz"
    inputs.record(path, manifest["aligned_scores_sha256"])
    with np.load(path, allow_pickle=False) as a:
        full = {k: a[k] for k in a.files}
    coordinates = strict_axis(full["time_indices"])
    mask = full["valid_mask"]
    labels = full["actual"]
    check(mask.dtype == bool and mask.shape == official.shape
          and np.array_equal(official[mask], coordinates)
          and len(coordinates) == expected_count, "Full score mask/coordinates differ")
    for compact, aligned in (("scores", "aligned_scores"), ("predictions", "aligned_predictions"),
                             ("actual", "aligned_actual")):
        check(np.array_equal(full[compact], full[aligned][mask]), f"Full compact mapping: {compact}")
    check(np.array_equal(labels, axes["test_labels"][mask])
          and np.all(full["aligned_actual"][~mask] == -1)
          and np.array_equal(full["time_values_ns"], axes["test_time_values_ns"][mask])
          and np.array_equal(full["local_indices"], np.flatnonzero(mask)), "Full labels/time mismatch")
    check(np.isin(labels, [0, 1]).all(), "nonbinary labels")
    inputs.record(source / "heads/calibration.npz", manifest["calibration_sha256"])
    with np.load(source / "heads/calibration.npz", allow_pickle=False) as c:
        center, scale = c["center"], c["scale"]
        threshold = float(c["threshold"])
        check(float(c["quantile"]) == 0.995, "wrong Full quantile")
    check(threshold == float(full["threshold"]) and np.all(scale > 0)
          and np.isfinite(center).all() and np.isfinite(scale).all(), "Full calibration differs")
    inputs.record(source / "results/test_nll.npy", manifest["test_nll_sha256"])
    nll = np.load(source / "results/test_nll.npy", mmap_mode="r", allow_pickle=False)
    check(nll.shape == (len(labels), len(center)), "Full NLL dimensions")
    for start in range(0, len(labels), 4096):
        block = nll[start:start + 4096]
        check(np.isfinite(block).all() and np.allclose(np.max((block - center) / scale, axis=1),
              full["scores"][start:start + len(block)], rtol=1e-5, atol=1e-6), "Full NLL reaggregation")
    check(np.array_equal(full["predictions"], full["scores"] > threshold), "Full predictions differ")
    methods = {"Full-GMM": dict(scores=full["scores"], threshold=threshold, path=path)}
    evidence_path = diagnostic / "R_test_prediction_evidence.npz"
    inputs.record(evidence_path)
    with np.load(evidence_path, allow_pickle=False) as a:
        check(np.array_equal(a["endpoints"], coordinates)
              and a["variable_names"].tolist() == manifest["variable_names"], "R evidence axes/layout")
        residual, r_nll = a["residual"], a["per_variable_nll"]
    inputs.record(diagnostic / "metrics.csv")
    metrics = read_csv(diagnostic / "metrics.csv")
    for score_name, method in (("squared_residual", "R-residual"), ("nll", "R-auxiliary-NLL")):
        path = diagnostic / f"R_{score_name}_test_scores.npz"
        inputs.record(path)
        with np.load(path, allow_pickle=False) as a:
            check(np.array_equal(a["global_indices"], coordinates) and np.array_equal(a["labels"], labels),
                  f"R score coordinates/labels: {method}")
            scores, c, s, threshold = a["scores"], a["center"], a["scale"], float(a["threshold"])
        check(c.shape == s.shape == (len(manifest["variable_names"]),)
              and np.isfinite(c).all() and np.isfinite(s).all() and np.all(s > 0), "R scale/layout")
        for start in range(0, len(labels), 4096):
            block = residual[start:start + 4096] ** 2 if score_name == "squared_residual" else r_nll[start:start + 4096]
            check(np.isfinite(block).all() and np.allclose(np.max((block - c) / s, axis=1),
                  scores[start:start + len(block)], rtol=1e-5, atol=1e-6), f"R score reaggregation: {method}")
        row, = [r for r in metrics if r["view"] == "R" and r["variant"] == "decoder" and r["score"] == score_name]
        check(threshold == float(row["threshold"]), "R saved threshold differs from metrics")
        calculated = stats(labels, scores > threshold, scores)
        for field, metric in (("ap", "average_precision"), ("roc_auc", "auc_roc"), ("tp", "tp"), ("fp", "fp")):
            check(np.isclose(calculated[field], float(row[metric]), rtol=0, atol=1e-12), f"R saved metric: {field}")
        methods[method] = dict(scores=scores, threshold=threshold, path=path)
    gaps = []
    for method in ("CATCH-common", "IsolationForest", "PCA"):
        path = COMMON / task / method / "results/aligned_scores.npz"
        if not path.is_file():
            gaps.append((method, "missing saved aligned_scores.npz", path))
            continue
        record_path = path.parent.parent / "evidence" / ("calibration.json" if method == "CATCH-common" else "baseline.json")
        record = inputs.json(record_path)
        check(record["status"] == "complete" and record["calibration_quantile"] == 0.995, "incomplete baseline")
        contract_path = Path(record["common_contract_path"])
        inputs.record(contract_path, record["common_contract_sha256"])
        contract = read_json(contract_path)
        check(Path(contract["probe_run"]).resolve() == source.resolve()
              and contract["probe_source_identity"] == manifest["source_identity"]
              and contract["variable_names"] == manifest["variable_names"], "baseline source identity")
        inputs.record(path, record["test_scores_sha256" if method == "CATCH-common" else "score_sha256"])
        with np.load(path, allow_pickle=False) as a:
            check(np.array_equal(strict_axis(a["time_indices"]), coordinates)
                  and np.array_equal(a["actual"], labels)
                  and np.array_equal(a["time_values_ns"], full["time_values_ns"]), "baseline pointwise axes")
            scores, threshold = a["scores"], float(a["threshold"])
            check(np.array_equal(a["predictions"], scores > threshold)
                  and threshold == record["threshold"], "baseline saved prediction/threshold")
        methods["PCA-residual" if method == "PCA" else method] = dict(scores=scores, threshold=threshold, path=path)
    for method in methods.values():
        check(np.isfinite(method["scores"]).all() and np.isfinite(method["threshold"]), "nonfinite scores/threshold")
    # These diagnostic references have only summary rows, no saved point scores.
    for variant in ("persistence", "train_target_mean", "decoder_latent_train_mean"):
        rows = [r for r in metrics if r["view"] == "R" and r["variant"] == variant and r["score"] == "squared_residual"]
        check(len(rows) == 1, f"missing simple reference summary: {variant}")
        gaps.append((variant, "aggregate metrics only; no saved pointwise score/calibration bundle", diagnostic / "metrics.csv"))
    return dict(task=task, run_id=rid, official=official, coordinates=coordinates, labels=labels,
                methods=methods, gaps=gaps, metrics=metrics, manifest=manifest)


def analyze(task):
    labels, coordinates = task["labels"], task["coordinates"]
    segment, edges = deciles(task["official"], coordinates)
    rows = []
    scopes = [("all_common", np.ones(len(labels), dtype=bool))]
    if task["task"] == "PSM":
        scopes.append(("exclude_deciles_7_8", ~np.isin(segment, (7, 8))))
    for scope, keep in scopes:
        scoped_labels = labels[keep]
        cap = int(np.sum(scoped_labels == 0)) // 100
        for method, record in task["methods"].items():
            scores = record["scores"][keep]
            selected = [("saved_q995", record["threshold"], scores > record["threshold"], None)]
            for budget in ([cap] if task["task"] == "PSM" else [222, 1480]):
                threshold, prediction = budget_point(scores, scoped_labels, budget)
                selected.append(("posthoc_fpr_1pct" if task["task"] == "PSM" else f"posthoc_fp_{budget}",
                                 threshold, prediction, budget))
            for rule, threshold, prediction, budget in selected:
                begin = len(rows)
                for part in [0] + [i for i in range(1, 11) if np.any(segment[keep] == i)]:
                    pick = np.ones(np.sum(keep), dtype=bool) if part == 0 else segment[keep] == part
                    row = dict(dataset=task["task"], source_run_id=task["run_id"], method=method,
                               scope=scope, decile=part or "global", operating_point=rule,
                               global_fp_budget=budget, threshold=threshold,
                               threshold_source="saved validation median/MAD + calibration q=.995" if rule == "saved_q995" else "post-hoc test-label ranking diagnostic; no calibration",
                               official_test_points=len(task["official"]), scope_points=int(np.sum(keep)),
                               official_global_start=int(task["official"][edges[part - 1]]) if part else int(task["official"][0]),
                               official_global_stop_exclusive=int(task["official"][edges[part] - 1]) + 1 if part else int(task["official"][-1]) + 1,
                               score_artifact_path=str(record["path"].relative_to(ROOT)),
                               status="complete_exploratory", reason="")
                    row.update(stats(scoped_labels[pick], prediction[pick], scores[pick]))
                    rows.append(row)
                group = rows[begin:]
                for field in ("tp", "fp", "points", "positive_points", "normal_points"):
                    check(sum(r[field] for r in group[1:]) == group[0][field],
                          f"segment accounting: {field}")
    for method, reason, path in task["gaps"]:
        row = dict(dataset=task["task"], source_run_id=task["run_id"], method=method,
                   scope="all_common", decile="global", operating_point="not_available",
                   status="aggregate_only" if method in ("persistence", "train_target_mean", "decoder_latent_train_mean") else "missing",
                   reason=reason, score_artifact_path=str(path.relative_to(ROOT)))
        reference = [r for r in task["metrics"] if r["view"] == "R" and r["variant"] == method and r["score"] == "squared_residual"]
        if reference:
            row.update(ap=float(reference[0]["average_precision"]), roc_auc=float(reference[0]["auc_roc"]),
                       points=int(reference[0]["valid_test_points"]))
        rows.append(row)
    return rows


def table(rows, fields):
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in fields) + " |")
    return lines


def report(rows, tasks, inputs):
    lines = ["# PSM / Creditcard 冻结 R 残差读出验证", "",
             "## 设计决定", "",
             "优先把 **R 平方残差读出**纳入下一轮预先固定的独立验证，而不是将 R 辅助 NLL 视为等价替代。Creditcard 的收益较大且覆盖多个交易区段；PSM 的整体收益集中，第 7、8 段之外仍超过 Full GMM，却未超过 CATCH-common/IF，不能据此宣布普遍优于基线。读出候选和报警候选必须分开：PSM 保存校准阈值仍有大量误报，不能把预算内排序改善称作报警问题解决。", "",
             "本轮没有找到与训练、校准及此前模型/评分选择隔离、且具有可对齐保存分数的评估集。检查范围为两组指定来源的 analysis_inputs、对应 FrozenDecoder 诊断目录，以及现有 common 比较契约/逐点包；目录中仅有已反复分析的官方测试集与训练侧统计。以下全部为**探索性、事后排序诊断**；重新切成十分位或去掉区段不构成新留出集。", "",
             "## 范围、规则与核对", "",
             "- 固定 seed=2021 的两组原 Full，所有字段、原 R validation median/MAD、跨变量 max 均保持不变。只读 saved score、residual/NLL 和元数据，不加载权重/特征缓存，不做前向/拟合/重新校准。R 残差、R 辅助 NLL 与四视图 Full GMM 是三种不同评分。",
             "- 分别对齐 PSM 87,650、Creditcard 142,213 个全局坐标、标签及来源轴；baseline 同时核对时间、完成记录、分数文件摘要、源数据/变量身份和保存阈值。Full 的紧凑分数/掩码/全轴映射、NLL 重聚合与保存预测均核对；R 的原始逐变量残差/NLL 按保存 center/scale 重聚合复核。rtol=1e-5、atol=1e-6；原 AP/AUC、TP/FP 与诊断 metrics.csv 一致。",
             "- 每种方法扫描自己的完整连续分数。相同分数整组进入/退出；最大化 TP，再最小化 FP。严格 `score > threshold`，不插值。FP 上限为整数：PSM floor(63272×.01)=632；排除 7/8 后 floor(52568×.01)=525；Creditcard 固定 222 / 1480。实际 FP 允许小于上限。",
             "- 十等分先对完整官方测试轴按 `floor(j*N/10)` 固定，再映射共同有效点；不是对紧凑数组重分块。PSM 官方轴 87,841，Creditcard 142,404，各有 191 点未评分，本轮未补分。各分段 TP/FP 使用**全局**工作点，绝不逐段另选阈值。CSV 的 decile=global 为全局，1..10 为区段。",
             "- 保存 q=.995 阈值逐方法保留，严格 > 复算。各方法不是共同数值阈值或相同实际 FPR。PSM 数据时间值不擅自解释为墙钟；Creditcard 十分位仅交易行序，不证明实体/session 连续性。", "",
             "## 全局结果（AP 与阈值无关）", ""]
    global_rows = [r for r in rows if r["decile"] == "global" and r["status"] == "complete_exploratory"]
    display = []
    for r in global_rows:
        display.append(dict(dataset=r["dataset"], scope=r["scope"], method=r["method"], rule=r["operating_point"],
                            AP=f'{r["ap"]:.6f}', TP=r["tp"], FP=r["fp"],
                            Recall=f'{100*r["recall"]:.3f}%', FPR=f'{100*r["fpr"]:.4f}%'))
    lines += table(display, ["dataset", "scope", "method", "rule", "AP", "TP", "FP", "Recall", "FPR"])
    def get(task, method, rule, scope="all_common", part="global"):
        match, = [r for r in rows if r["dataset"] == task and r["method"] == method
                  and r["operating_point"] == rule and r["scope"] == scope and r["decile"] == part]
        return match

    r = get("PSM", "R-residual", "posthoc_fpr_1pct")
    core_tp = sum(get("PSM", "R-residual", "posthoc_fpr_1pct", part=i)["tp"] for i in (7, 8))
    outside = get("PSM", "R-residual", "posthoc_fpr_1pct", "exclude_deciles_7_8")
    lines += ["", "## PSM：局部收益、外推边界与报警", "",
              f'全样本 1% FPR 预算内 R 残差检出 {r["tp"]} 点，其中第 7、8 段合计 {core_tp}（{core_tp/r["tp"]:.2%}）；该两段包含 6,864/24,378 个有效异常点。去掉两段后并非沿用旧工作点：重新对余下 70,082 点（17,514 异常、52,568 正常）扫描 floor(52568×.01)=525 的预算。R 残差 TP/FP={outside["tp"]}/{outside["fp"]}、Recall={outside["recall"]:.3%}、AP={outside["ap"]:.6f}；相对 Full GMM 仍有收益，但不能把原全局优势外推为各阶段都强于基线。', ""]
    comparisons = []
    for method in ("Full-GMM", "R-auxiliary-NLL", "CATCH-common", "IsolationForest", "PCA-residual"):
        other = get("PSM", method, "posthoc_fpr_1pct", "exclude_deciles_7_8")
        comparisons.append(dict(method=method, TP=other["tp"], FP=other["fp"],
                                R_minus_TP=outside["tp"]-other["tp"], AP=f'{other["ap"]:.6f}'))
    lines += table(comparisons, ["method", "TP", "FP", "R_minus_TP", "AP"])
    lines += ["", "保存阈值的原正常高分时段对照（十分位 4、6、7、8，单元为 FP / 正常点 FPR）：", ""]
    comparison = []
    for i in (4, 6, 7, 8):
        row = {"decile": i}
        for method in ("Full-GMM", "R-residual", "R-auxiliary-NLL"):
            item = get("PSM", method, "saved_q995", part=i)
            row[method] = f'{item["fp"]} / {item["fpr"]:.2%}'
        comparison.append(row)
    lines += table(comparison, ["decile", "Full-GMM", "R-residual", "R-auxiliary-NLL"])
    rsave = get("PSM", "R-residual", "saved_q995")
    fsave = get("PSM", "Full-GMM", "saved_q995")
    old_periods = sum(get("PSM", "R-residual", "saved_q995", part=i)["fp"] for i in (4, 6, 7, 8))
    lines += ["", f'R 残差保存阈值全局 FP={rsave["fp"]}、FPR={rsave["fpr"]:.3%}，Full GMM 为 {fsave["fp"]}/{fsave["fpr"]:.3%}；R 残差在原 4/6/7/8 段仍有 {old_periods} 个误报（占自身 FP {old_periods/rsave["fp"]:.2%}）。逐段变化不一致，降低总体误报不等于修复原正常高分工况。其保存阈值 Recall 为 {rsave["recall"]:.3%}，Full 为 {fsave["recall"]:.3%}，不能只报更低 FPR。',
              "预算内 R 残差优于 R 辅助 NLL 支持对读出公式进行独立确认；保存阈值下的误报则仍要求另行验证训练侧校准代表性/漂移。这里没有重新校准，也不据测试预算选部署阈值。原时段依据：[PSM 正常高分诊断](../../MNRF-PSM-ScoreDiagnosis/psm-score-diagnosis-20261103T180743Z-306520/README.md)。", "",
              "## Creditcard：完整扫描 Full GMM 后的收益", ""]
    comparison = []
    for rule in ("posthoc_fp_222", "posthoc_fp_1480"):
        residual = get("Creditcard", "R-residual", rule)
        full = get("Creditcard", "Full-GMM", rule)
        comparison.append(dict(budget=rule, R_TP_FP=f'{residual["tp"]}/{residual["fp"]}',
                               Full_TP_FP=f'{full["tp"]}/{full["fp"]}',
                               extra_TP=residual["tp"]-full["tp"],
                               extra_Recall_pp=f'{100*(residual["recall"]-full["recall"]):.3f}'))
    lines += table(comparison, ["budget", "R_TP_FP", "Full_TP_FP", "extra_TP", "extra_Recall_pp"])
    lines += ["", "Full GMM 1,480 预算是对所有工作点完整扫描，结果不是引用保存报警点：同样 54 TP 的扫描工作点仅需 1,406 FP，保存阈值则为 54 TP / 1,480 FP。R 残差 1,480 预算最佳为 185 TP / 1,242 FP（保存阈值为 184 TP / 1,167 FP）。", "",
              "新增/丢失 TP 是逐点集合变化，非仅净差；以下按固定官方交易十等分统计：", ""]
    task = next(t for t in tasks if t["task"] == "Creditcard")
    segment, _ = deciles(task["official"], task["coordinates"])
    comparison = []
    for budget in (222, 1480):
        _, rp = budget_point(task["methods"]["R-residual"]["scores"], task["labels"], budget)
        _, fp = budget_point(task["methods"]["Full-GMM"]["scores"], task["labels"], budget)
        for i in range(1, 11):
            positive = (task["labels"] == 1) & (segment == i)
            comparison.append(dict(budget=budget, decile=i,
                                   added_TP=int(np.sum(positive & rp & ~fp)),
                                   lost_TP=int(np.sum(positive & ~rp & fp))))
    lines += table(comparison, ["budget", "decile", "added_TP", "lost_TP"])
    lines += ["", "222 预算的净增 104 TP 分布于第 1–9 段，第 1/8 段分别增 34/19，合计占净增 50.96%；1,480 预算的净增 131 TP 覆盖全部十段，第 1/8 段分别增 22/23，合计占净增 34.35%。两个预算下均未丢失 Full GMM 已检出的异常点。第 1 段有 79/221 个异常点，不能把原始 TP 集中程度与每段检出能力混淆；完整分母在 CSV 中。", "",
              "R 残差相对 R 辅助 NLL 的排序差异在同一冻结解码器上出现，未改变表示/变量选择；与 Full GMM 的差异同时包含读出公式和任务差异，不能宣称已证明 GMM 根因。交易区段跨度支持它不只是一个局部事件，但仍是已用于分析/选择候选的测试集，尚未证明独立数据上的稳定性。", "",
              "## 下一版设计：读出与报警分开验收", "",
              "1. **读出优先级：优先验证 R 平方残差，保留全字段、保存标准化和 max 规则作为固定候选。** Creditcard 的同预算收益跨多个区段；PSM 排除主要获益时段后仍优于当前 Full GMM，但低于 CATCH-common/IF，故只支持有数据/阶段依赖的候选，不支持统一替换正式模型或宣布全局胜出。不得按当前测试结果删除变量、调尺度或拼接逐数据集最佳头。",
              "2. **报警决定：不批准把保存 q=.995 的 R 残差报警直接当作已解决方案。** PSM 仍有高 FPR 和原高分时段误报，排序改善与校准迁移必须独立判定；Creditcard 保存报警明显更有用，但也需独立留出评估。",
              "3. **下一次最小验证：** 在未用于当前选择的留出数据/未来时间区间上，预先固定 R 残差与 Full、R NLL、可取得的简单参照，固定数据与坐标和训练侧阈值规则；分别验收低误报排序和保存校准报警。若没有独立数据，则不能把历史分块复验升级为确认性证据。本轮不启动该实验。", ""]
    lines += ["## 完整区段统计与简单参照缺口", "",
              "各方法全部十等分的 TP、FP、异常/正常点数、Recall、FPR、AP，以及 PSM 排除 7/8 后的区段结果均在 [results.csv](results.csv)。每行 operating_point 指定全局选定的工作点，decile 指定统计区段；global_fp_budget 不意味着对各段分别使用该预算。所有分段计数之和均已核对等于对应全局计数。", "",
              "persistence、训练目标均值、latent 训练均值替换只有既有汇总行，未保存可对齐的一维逐点总分和校准包；本轮不通过重算预测或当前输入补造。CSV 将其标为 aggregate_only，保留原 AP/AUC，仅不参与同预算和逐段比较。此缺口不等于参照失效或训练失败。", ""]
    lines += table([r for r in rows if r["status"] in ("aggregate_only", "missing")],
                   ["dataset", "method", "status", "ap", "reason"])
    lines += ["", "## 来源与文件保护", "",
              f"实际 HEAD：`{subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}`。",
              "全部已消费输入的小文件/逐点证据在分析前后 SHA256 相同；未重新计算大型特征缓存摘要。以下记录实际读取路径与摘要。", ""]
    score_inputs = [(p, h) for p, h in inputs.hashes.items()
                    if p.name in ("aligned_scores.npz", "R_squared_residual_test_scores.npz", "R_nll_test_scores.npz", "analysis_inputs.npz")]
    lines += table([dict(path=str(p.relative_to(ROOT)), sha256=h) for p, h in score_inputs], ["path", "sha256"])
    lines += ["", f"另核对来源 manifest、诊断收据、metrics.csv、Full 校准/NLL、R 逐变量证据及 common 契约/完成记录；总计 {len(inputs.hashes)} 个实际消费文件在本次读取前后摘要不变。来源与方法评分路径同时保存在结果 CSV。报告未改写原历史表、模型或评分产物。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    check(not args.output_dir.exists(), "output exists; preserve historical analysis")
    with threadpool_limits(limits=1):
        inputs = Inputs()
        tasks = [load_task(task, inputs) for task in SOURCES]
        rows = [row for task in tasks for row in analyze(task)]
        inputs.unchanged()
        text = report(rows, tasks, inputs)
        args.output_dir.mkdir(parents=True)
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (args.output_dir / "results.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (args.output_dir / "README.md").write_text(text)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
