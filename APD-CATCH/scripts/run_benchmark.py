# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, NoReturn

import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ts_benchmark.utils.get_file_name import get_unique_file_suffix
from ts_benchmark.report import report
from ts_benchmark.common.constant import CONFIG_PATH, ROOT_PATH, THIRD_PARTY_PATH
from ts_benchmark.pipeline import pipeline
from ts_benchmark.utils.parallel import ParallelBackend


def _register_mnrf_report_references(args, report_config):
    """Bind a successfully written report to each MNRF run, without affecting other models."""
    import hashlib
    import pandas as pd
    import tempfile

    report_path = Path(ROOT_PATH) / "result" / (report_config.get("save_path") or "") / report_config["leaderboard_file_name"]
    if not report_path.is_file():
        raise FileNotFoundError(f"benchmark report was not written: {report_path}")
    report = pd.read_csv(report_path)
    for model_name, encoded_params in zip(args.model_name, args.model_hyper_params):
        if model_name != "mnrf_lite_v2.MNRFLiteV2" or not encoded_params:
            continue
        params = json.loads(encoded_params)
        run_dir = Path(params["output_dir"])
        if report_path.resolve().parent != run_dir.resolve():
            raise ValueError(
                f"MNRF report must be saved inside its run directory: report={report_path}, run={run_dir}"
            )
        context_path = run_dir / "benchmark_context.json"
        if not context_path.is_file():
            raise ValueError(f"MNRF benchmark context is missing: {context_path}")
        context = json.loads(context_path.read_text(encoding="utf-8"))
        task = context.get("task")
        if (context.get("run_id") != run_dir.name or not task
                or params.get("task_name") != task):
            raise ValueError(f"MNRF benchmark context identity is invalid: {context_path}")
        source_logs = report_config.get("log_files_list")
        if isinstance(source_logs, pd.DataFrame):
            source_records = source_logs
        else:
            source_records = pd.concat([pd.read_csv(path) for path in source_logs], ignore_index=True)
        strategy_names = set()
        for value in source_records.get("strategy_args", []):
            try:
                decoded = json.loads(value) if isinstance(value, str) else value
            except (TypeError, ValueError):
                decoded = value
            strategy_names.add(decoded.get("strategy_name") if isinstance(decoded, dict) else str(decoded))
        if ("file_name" not in source_records.columns
                or set(source_records["file_name"].astype(str)) != {task}
                or "model_name" not in source_records.columns
                or set(source_records["model_name"].astype(str)) != {"MNRFLiteV2"}
                or strategy_names != {"mnrf_valid_detect"}):
            raise ValueError(f"MNRF benchmark records do not uniquely belong to task {task}")
        model_columns = [column for column in report.columns if str(column).startswith("MNRFLiteV2;")]
        if len(model_columns) != 1 or "metric_name" not in report:
            raise ValueError(f"MNRF report has no unique model metric column: {report_path}")
        metric_values = {}
        task_column = "file_name" if "file_name" in report else None
        selected = report if task_column is None else report.loc[report[task_column] == task]
        for metric in ("auc_roc", "auc_pr"):
            rows = selected.loc[selected.metric_name == metric]
            if len(rows) != 1:
                raise ValueError(f"MNRF report metric {metric} is missing or ambiguous for {task}")
            metric_values[metric] = float(rows.iloc[0][model_columns[0]])
        digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
        reference = {
            "schema_version": 1,
            "task": task,
            "model_name": "MNRFLiteV2",
            "run_id": run_dir.name,
            "benchmark_invocation_id": context["benchmark_invocation_id"],
            "report_path": str(report_path.resolve()),
            "report_sha256": digest,
            "model_column": model_columns[0],
            "metrics": metric_values,
        }
        destination = run_dir / "report_reference.json"
        fd, temporary = tempfile.mkstemp(prefix=".report-reference-", dir=run_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(reference, handle, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


sys.path.insert(0, THIRD_PARTY_PATH)

warnings.filterwarnings("ignore")


def build_data_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the data loader config from commandline arguments and configuration dict
    """
    data_config = config_data["data_config"]
    data_config["data_name_list"] = args.data_name_list
    if args.data_set_name is not None:
        data_config["data_set_name"] = args.data_set_name
    return data_config


def build_model_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the model config from commandline arguments and configuration dict
    """
    model_config = config_data.get("model_config", None)

    if args.adapter is not None:
        args.adapter = [None if item == "None" else item for item in args.adapter]
        if len(args.model_name) > len(args.adapter):
            args.adapter.extend([None] * (len(args.model_name) - len(args.adapter)))
    else:
        args.adapter = [None] * len(args.model_name)

    if args.model_hyper_params is not None:
        args.model_hyper_params = [
            None if item == "None" else item for item in args.model_hyper_params
        ]
        if len(args.model_name) > len(args.model_hyper_params):
            args.model_hyper_params.extend(
                [None] * (len(args.model_name) - len(args.model_hyper_params))
            )
    else:
        args.model_hyper_params = [None] * len(args.model_name)

    for adapter, model_name, model_hyper_params in zip(
        args.adapter, args.model_name, args.model_hyper_params
    ):
        model_config["models"].append(
            {
                "adapter": adapter,
                "model_name": model_name,
                "model_hyper_params": json.loads(model_hyper_params)
                if model_hyper_params is not None
                else {},
            }
        )

    return model_config


def build_evaluation_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the evaluation config from commandline arguments and configuration dict
    """
    evaluation_config = config_data["evaluation_config"]

    metric_list = []
    if args.metrics != "all" and args.metrics is not None:
        for metric in args.metrics:
            metric = json.loads(metric)
            metric_list.append(metric)
        evaluation_config["metrics"] = metric_list

    default_strategy_args = evaluation_config["strategy_args"]
    strategy_args_updates = (
        json.loads(args.strategy_args) if args.strategy_args else None
    )

    if strategy_args_updates is not None:
        default_strategy_args.update(strategy_args_updates)

    if args.seed is not None:
        default_strategy_args["seed"] = args.seed

    return evaluation_config


def build_report_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the report config from commandline arguments and configuration dict
    """
    report_config = config_data["report_config"]
    report_config["aggregate_type"] = args.aggregate_type
    report_config["save_path"] = args.save_path

    return report_config


def init_worker(env: Dict) -> NoReturn:
    """
    An initializer function for each worker that does some global setup
    """
    sys.path.insert(0, THIRD_PARTY_PATH)
    torch.set_num_threads(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run_benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # script name
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Evaluation config file path",
    )

    parser.add_argument(
        "--data-name-list",
        type=str,
        nargs="+",
        default=None,
        help="List of series names entered by the user",
    )

    parser.add_argument(
        "--data-set-name",
        type=str,
        nargs="+",
        default=None,
        help="List of dataset name names entered by the user,"
             "only takes effect when data_name_list is not specified",
    )

    # model_config
    parser.add_argument(
        "--adapter",
        type=str,
        nargs="+",
        default=None,
        help="Adapter used to adapt the method to our pipeline",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        nargs="+",
        required=True,
        help="The relative path of the model that needs to be evaluated",
    )
    parser.add_argument(
        "--model-hyper-params",
        type=str,
        nargs="+",
        default=None,
        help=(
            "The input parameters corresponding to the models to be evaluated "
            "should correspond one-to-one with the --model-name options."
        ),
    )

    # evaluation_config
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Evaluation metrics that need to be calculated",
    )
    parser.add_argument(
        "--strategy-args",
        type=str,
        default=None,
        help="Parameters required for evaluating strategies",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed that is set before evaluating any model-series pair, "
             "by default, use the seed value in the config file"
    )

    # evaluation engine
    parser.add_argument(
        "--eval-backend",
        type=str,
        default="sequential",
        choices=["sequential", "ray"],
        help="Evaluation backend, use ray for parallel evaluation",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=os.cpu_count(),
        help="Number of cpus to use, only available in both backends",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="List of gpu devices to use, only available in ray backends",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count(),
        help="Number of evaluation workers",
    )
    # TODO: should timeout be part of the configuration file?
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Time limit for each evaluation task, in seconds",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=100,
        help="Max tasks to run on a single worker when using parallel backends",
    )

    # report_config
    parser.add_argument(
        "--aggregate_type",
        default="mean",
        help="Select the baseline algorithm to compare",
    )

    parser.add_argument(
        "--report-method",
        type=str,
        default="csv",
        choices=[
            "dash",
            "csv",
        ],
        help="Presentation form of algorithm performance comparison results",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="The relative path for saving evaluation results, relative to the result folder",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    torch.set_num_threads(3)
    with open(os.path.join(CONFIG_PATH, args.config_path), "r") as file:
        config_data = json.load(file)

    required_configs = [
        "data_config",
        "model_config",
        "evaluation_config",
        "report_config",
    ]
    for config_name in required_configs:
        if config_data.get(config_name) is None:
            raise ValueError(f"{config_name} is none")

    data_config = build_data_config(args, config_data)
    model_config = build_model_config(args, config_data)
    evaluation_config = build_evaluation_config(args, config_data)
    report_config = build_report_config(args, config_data)

    ParallelBackend().init(
        backend=args.eval_backend,
        n_workers=args.num_workers,
        n_cpus=args.num_cpus,
        gpu_devices=args.gpus,
        default_timeout=args.timeout,
        max_tasks_per_child=args.max_tasks_per_child,
        worker_initializers=[init_worker],
    )

    try:
        log_filenames = pipeline(
            data_config,
            model_config,
            evaluation_config,
            save_path=args.save_path,
        )

    finally:
        ParallelBackend().close(force=True)

    report_config["log_files_list"] = log_filenames
    if args.report_method == "csv":
        filename = get_unique_file_suffix()
        leaderboard_file_name = "test_report" + filename
        report_config["leaderboard_file_name"] = leaderboard_file_name
    report(report_config, report_method=args.report_method)
    if args.report_method == "csv":
        _register_mnrf_report_references(args, report_config)
