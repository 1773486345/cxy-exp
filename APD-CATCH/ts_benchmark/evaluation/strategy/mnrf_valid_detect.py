"""MNRF evaluation preserving the exact valid window-end index."""

import base64
import json
import os
import pickle
import re
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.evaluation.metrics.classification_metrics_score import __all__ as SCORE_METRICS
from ts_benchmark.evaluation.strategy.anomaly_detect import UnFixedDetectScore
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.models import ModelFactory
from ts_benchmark.utils.data_processing import split_before


class MNRFValidDetect(UnFixedDetectScore):
    """Train on metadata's official train_lens and score only complete windows."""

    REQUIRED_CONFIGS = ["strategy_name"]

    @staticmethod
    def accepted_metrics():
        return SCORE_METRICS

    def split_data(self, series_name: str):
        data = DataPool().get_pool().get_series(series_name)
        train_length = int(DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item())
        train, test = split_before(data, train_length)
        train_data = train.loc[:, train.columns != "label"]
        train_label = train.loc[:, ["label"]]
        test_data = test.loc[:, train.columns != "label"]
        test_label = test.loc[:, ["label"]]
        return train_data, train_label, test_data, test_label

    def execute(self, series_name: str, model_factory: ModelFactory):
        model = model_factory()
        self.model = model
        train_data, train_label, test_data, test_label = self.split_data(series_name)
        start = time.time()
        model.detect_fit(train_data, train_label)
        fit_end = time.time()
        scored = model.score_with_metadata(test_data)
        inference_end = time.time()
        valid = scored["valid_mask"]
        indices = scored["time_indices"]
        actual = test_label.to_numpy().reshape(-1)
        if len(actual) != len(valid) or len(indices) != len(scored["scores"]):
            raise RuntimeError("MNRF score/index/label alignment contract violated")
        if len(indices) == 0:
            raise RuntimeError("MNRF test segment is shorter than one complete scoring window")
        selected_actual = actual[indices]
        values, log_info = self.evaluator.evaluate_with_log(
            actual=selected_actual.astype(float),
            predicted=scored["scores"].astype(float),
        )
        predictions = scored["predictions"].astype(np.int64)
        test_time_index = np.asarray(test_data.index)[indices]
        tp = int(((selected_actual == 1) & (predictions == 1)).sum())
        fp = int(((selected_actual == 0) & (predictions == 1)).sum())
        fn = int(((selected_actual == 1) & (predictions == 0)).sum())
        tn = int(((selected_actual == 0) & (predictions == 0)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        fpr = fp / max(1, fp + tn)
        metric_payload = {
            "status": "failed" if log_info or not np.isfinite(np.asarray(values, dtype=float)).all() else "complete",
            "task": series_name,
            "fit_seconds": float(fit_end - start),
            "inference_seconds": float(inference_end - fit_end),
            "auc_roc": float(values[0]) if len(values) > 0 else float("nan"),
            "auc_pr": float(values[1]) if len(values) > 1 else float("nan"),
            "evaluation_log": log_info,
            "valid_points": int(valid.sum()),
            "test_points": int(len(valid)),
            "coverage": float(valid.mean()),
            "valid_start_index": int(indices[0]) if len(indices) else None,
            "valid_end_index_inclusive": int(indices[-1]) if len(indices) else None,
            "valid_start_timestamp": str(test_time_index[0]) if len(test_time_index) else None,
            "valid_end_timestamp": str(test_time_index[-1]) if len(test_time_index) else None,
            "threshold": float(scored["threshold"]),
            "precision_at_calibrated_threshold": precision,
            "recall_at_calibrated_threshold": recall,
            "f1_at_calibrated_threshold": f1,
            "normal_false_alarm_rate": fpr,
            "auc_roc_defined": bool(np.unique(selected_actual).size == 2),
            "auc_pr_protocol": "average_precision",
        }
        if not metric_payload["auc_roc_defined"]:
            metric_payload["auc_undefined_reason"] = "valid evaluation labels contain a single class"
        run_dir = model.output_dir
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", series_name)
        np.savez_compressed(
            run_dir / "results" / f"{safe_name}_aligned_scores.npz",
            scores=scored["scores"],
            predictions=predictions,
            time_indices=indices,
            time_index_values=test_time_index,
            valid_mask=valid,
            actual=actual,
            per_variable_nll=scored["per_variable_nll"],
            per_variable_calibrated=scored["per_variable_calibrated"],
            threshold=np.asarray(scored["threshold"]),
        )
        with open(run_dir / "results" / f"{safe_name}_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metric_payload, handle, indent=2)
        inference = {
            "scores": scored["scores"],
            "predictions": predictions,
            "time_indices": indices,
            "valid_mask": valid,
            "threshold": scored["threshold"],
            "metrics": metric_payload,
        }
        actual_pickle = base64.b64encode(pickle.dumps(test_label.iloc[indices])).decode("utf-8")
        inference_pickle = base64.b64encode(pickle.dumps(inference)).decode("utf-8")
        return [values + [
            series_name,
            fit_end - start,
            inference_end - fit_end,
            float(np.mean(predictions)) if len(predictions) else np.nan,
            actual_pickle,
            inference_pickle,
            log_info,
        ]]
