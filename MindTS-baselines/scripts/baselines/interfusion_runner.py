#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import pickle
import shutil
import sys
import uuid

import numpy as np


DEFAULT_INTERFUSION_REPO = "/media/h3c/users/wangyueyang1/cxy/baseline_repos/InterFusion"


def _window_shapes(window_length):
    half = (window_length + 1) // 2
    quarter = (half + 1) // 2
    z2_dim = (quarter + 1) // 2
    return z2_dim, [quarter, quarter, half, half, window_length]


def _dump_pickle(path, value):
    with open(path, "wb") as f:
        pickle.dump(value, f, protocol=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--window-length", type=int, default=100)
    parser.add_argument("--z-dim", type=int, default=3)
    parser.add_argument("--z2-dim", type=int, default=None)
    parser.add_argument("--rnn-hidden", type=int, default=100)
    parser.add_argument("--dense-hidden", type=int, default=100)
    parser.add_argument("--arnn-hidden", type=int, default=100)
    parser.add_argument("--posterior-flow-layers", type=int, default=4)
    parser.add_argument("--posterior-flow-type", default="rnvp")
    parser.add_argument("--pretrain-max-epoch", type=int, default=10)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--test-n-z", type=int, default=1)
    parser.add_argument("--valid-portion", type=float, default=0.3)
    parser.add_argument("--initial-lr", type=float, default=0.001)
    parser.add_argument("--max-train-size", type=int, default=None)
    parser.add_argument("--max-test-size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    repo = os.path.abspath(DEFAULT_INTERFUSION_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)

    import mltk  # noqa: E402
    from algorithm.stack_predict import PredictConfig, main as predict_main  # noqa: E402
    from algorithm.stack_train import ExpConfig, main as train_main  # noqa: E402

    arrays = np.load(args.input)
    train = np.asarray(arrays["train"], dtype=np.float32)
    test = np.asarray(arrays["test"], dtype=np.float32)
    label = np.asarray(arrays.get("label", np.zeros([len(test)], dtype=np.float32)), dtype=np.float32).reshape(-1)
    x_dim = int(train.shape[1])

    run_id = "MindTS_{}".format(uuid.uuid4().hex)
    processed_dir = os.path.join(repo, "data", "processed")
    train_pkl = os.path.join(processed_dir, run_id + "_train.pkl")
    test_pkl = os.path.join(processed_dir, run_id + "_test.pkl")
    label_pkl = os.path.join(processed_dir, run_id + "_test_label.pkl")

    work_dir = os.path.abspath(args.work_dir)
    train_dir = os.path.join(work_dir, "train")
    predict_dir = os.path.join(work_dir, "predict")
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    try:
        _dump_pickle(train_pkl, train)
        _dump_pickle(test_pkl, test)
        _dump_pickle(label_pkl, label)

        z2_dim, output_shape = _window_shapes(int(args.window_length))
        if args.z2_dim is not None:
            z2_dim = int(args.z2_dim)

        train_config = ExpConfig()
        train_config.seed = int(args.seed)
        train_config.dataset = run_id
        train_config.model.x_dim = x_dim
        train_config.model.window_length = int(args.window_length)
        train_config.model.output_shape = output_shape
        train_config.model.z2_dim = z2_dim
        train_config.model.z_dim = int(args.z_dim)
        train_config.model.rnn_hidden_units = int(args.rnn_hidden)
        train_config.model.dense_hidden_units = int(args.dense_hidden)
        train_config.model.arnn_hidden_units = int(args.arnn_hidden)
        train_config.model.posterior_flow_layers = int(args.posterior_flow_layers)
        train_config.model.posterior_flow_type = (
            None if args.posterior_flow_type == "None" else args.posterior_flow_type
        )
        train_config.train.batch_size = int(args.batch_size)
        train_config.train.pretrain_max_epoch = int(args.pretrain_max_epoch)
        train_config.train.max_epoch = int(args.max_epoch)
        train_config.train.max_train_size = args.max_train_size
        train_config.train.valid_portion = float(args.valid_portion)
        train_config.train.initial_lr = float(args.initial_lr)
        train_config.train.early_stopping = False
        train_config.test.max_test_size = args.max_test_size
        train_config.test.test_n_z = int(args.test_n_z)
        train_config.test.test_batch_size = int(args.test_batch_size)
        train_config.save_ckpt = False
        train_config.write_summary = False
        train_config.write_histogram_summary = False

        train_exp = mltk.Experiment(train_config, output_dir=train_dir)
        train_exp.__enter__()
        try:
            train_exp.save_config()
            train_main(train_exp, train_config)
        finally:
            train_exp.__exit__(None, None, None)

        import tensorflow as tf  # noqa: E402
        tf.reset_default_graph()

        predict_config = PredictConfig()
        predict_config.load_model_dir = train_dir
        predict_config.test_n_z = int(args.test_n_z)
        predict_config.test_batch_size = int(args.test_batch_size)
        predict_config.max_test_size = args.max_test_size
        predict_config.use_mcmc = False
        predict_config.mcmc_track = False
        predict_config.plot_recons_results = False
        predict_config.save_results = True

        predict_exp = mltk.Experiment(predict_config, output_dir=predict_dir)
        predict_exp.__enter__()
        try:
            predict_main(predict_exp, predict_config)
        finally:
            predict_exp.__exit__(None, None, None)

        analysis_dir = os.path.join(predict_dir, predict_config.output_dirs)
        with open(os.path.join(analysis_dir, predict_config.train_score_filename), "rb") as f:
            train_score = np.asarray(pickle.load(f), dtype=np.float64)
        with open(os.path.join(analysis_dir, predict_config.test_score_filename), "rb") as f:
            test_score = np.asarray(pickle.load(f), dtype=np.float64)

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez(args.output, train_score=-train_score, test_score=-test_score)
    finally:
        for path in (train_pkl, test_pkl, label_pkl):
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
