#!/usr/bin/env python

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MTAD_GAT_REPO = PROJECT_ROOT.parent / "baseline_repos" / "mtad-gat"
TRAINING_SCRIPT = MTAD_GAT_REPO / "training.py"


def make_example(window, label, anomaly=0):
    record = {
        "input": tf.train.Feature(float_list=tf.train.FloatList(value=window.reshape([-1]))),
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=label.reshape([-1]))),
        "anomaly": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(anomaly)])),
    }
    return tf.train.Example(features=tf.train.Features(feature=record))


def write_tfrecord(values, labels, window_size, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tf.io.TFRecordWriter(str(path)) as writer:
        position = 0
        while position + window_size + 1 < values.shape[0]:
            target_pos = position + window_size + 1
            anomaly = 0 if labels is None else labels[target_pos]
            example = make_example(
                values[position : position + window_size],
                values[target_pos],
                anomaly,
            )
            writer.write(example.SerializeToString())
            count += 1
            position += 1
    return count


def run_command(cmd, cwd, log_path, timeout):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    log_path.write_text(completed.stdout)
    print(completed.stdout[-12000:])
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(cmd)}")


def run_predict(model_dir, tfrecord_path, pred_dir, args, log_name):
    pred_dir.mkdir(parents=True, exist_ok=True)
    cmd = base_training_cmd(args) + [
        "--action=PREDICT",
        "--prediction_task=inference_score",
        f"--test_file={tfrecord_path}",
        f"--output_dir={model_dir}",
    ]
    run_command(cmd, pred_dir, pred_dir / log_name, args.timeout)
    score_path = pred_dir / "inference_score.csv"
    if not score_path.exists():
        raise RuntimeError(f"MTAD-GAT prediction did not write {score_path}")
    scores = np.atleast_1d(np.loadtxt(str(score_path), dtype=np.float64))
    if scores.size == 0:
        raise RuntimeError("MTAD-GAT produced empty scores.")
    if not np.isfinite(scores).all():
        raise RuntimeError("MTAD-GAT produced non-finite scores.")
    if np.nanmax(scores) == np.nanmin(scores):
        raise RuntimeError("MTAD-GAT produced constant scores.")
    return scores


def base_training_cmd(args):
    return [
        sys.executable,
        str(TRAINING_SCRIPT),
        f"--run_mode={args.run_mode}",
        f"--batch_size={args.batch_size}",
        f"--window_size={args.window_size}",
        f"--num_features={args.num_features}",
        f"--GRU_hidden_size={args.gru_hidden}",
        f"--fc_hidden_size={args.fc_hidden}",
        f"--VAE_latent_space_dimension={args.vae_latent}",
        f"--conv1d_filter_width={args.conv1d_filter_width}",
        f"--learning_rate={args.learning_rate}",
        f"--clip_gradients={args.clip_gradients}",
        f"--dropout_prob={args.dropout_prob}",
        f"--gamma={args.gamma}",
        f"--save_checkpoints_steps={args.save_checkpoints_steps}",
        f"--log_step_count_steps={args.log_step_count_steps}",
        f"--keep_checkpoint_max={args.keep_checkpoint_max}",
        f"--shuffle_buffer_size={args.shuffle_buffer_size}",
        f"--dataset_reader_buffer_size={args.dataset_reader_buffer_size}",
        f"--random_seed={args.seed}",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--run-mode", default="FORECASTING", choices=["FORECASTING", "RECONSTRUCTING", "BOTH"])
    parser.add_argument("--num-train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--fc-hidden", type=int, default=64)
    parser.add_argument("--vae-latent", type=int, default=18)
    parser.add_argument("--conv1d-filter-width", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--clip-gradients", type=float, default=0.1)
    parser.add_argument("--dropout-prob", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--save-checkpoints-steps", type=int, default=1000)
    parser.add_argument("--log-step-count-steps", type=int, default=100)
    parser.add_argument("--keep-checkpoint-max", type=int, default=2)
    parser.add_argument("--shuffle-buffer-size", type=int, default=29000)
    parser.add_argument("--dataset-reader-buffer-size", type=int, default=1048576)
    parser.add_argument("--timeout", type=int, default=60000)
    args = parser.parse_args()

    if not TRAINING_SCRIPT.exists():
        raise RuntimeError(f"MTAD-GAT training script not found: {TRAINING_SCRIPT}")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(args.input)
    train = np.asarray(arrays["train"], dtype=np.float32)
    test = np.asarray(arrays["test"], dtype=np.float32)
    test_labels = np.asarray(arrays["label"], dtype=np.int64) if "label" in arrays else None
    args.num_features = int(train.shape[1])

    train_record = work_dir / "records" / "train.tfrecords"
    test_record = work_dir / "records" / "test.tfrecords"
    train_count = write_tfrecord(train, None, args.window_size, train_record)
    test_count = write_tfrecord(test, test_labels, args.window_size, test_record)
    if train_count <= 0 or test_count <= 0:
        raise RuntimeError(
            f"Not enough records for MTAD-GAT: train={train_count}, test={test_count}, window={args.window_size}"
        )

    model_dir = work_dir / "model"
    train_cmd = base_training_cmd(args) + [
        "--action=TRAIN",
        f"--train_file={train_record}",
        f"--output_dir={model_dir}",
        f"--num_train_steps={args.num_train_steps}",
    ]
    run_command(train_cmd, work_dir, work_dir / "train.log", args.timeout)

    train_score = run_predict(model_dir, train_record, work_dir / "predict_train", args, "predict_train.log")
    test_score = run_predict(model_dir, test_record, work_dir / "predict_test", args, "predict_test.log")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        train_score=train_score,
        test_score=test_score,
        train_records=np.array([train_count], dtype=np.int64),
        test_records=np.array([test_count], dtype=np.int64),
    )


if __name__ == "__main__":
    main()
