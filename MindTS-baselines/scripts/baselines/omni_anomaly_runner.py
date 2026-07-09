#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow as tf


DEFAULT_OMNI_REPO = "/media/h3c/users/wangyueyang1/cxy/baseline_repos/OmniAnomaly"
if DEFAULT_OMNI_REPO not in sys.path:
    sys.path.insert(0, DEFAULT_OMNI_REPO)

from omni_anomaly.model import OmniAnomaly  # noqa: E402
from omni_anomaly.prediction import Predictor  # noqa: E402
from omni_anomaly.training import Trainer  # noqa: E402


class Config(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--window-length", type=int, default=100)
    parser.add_argument("--z-dim", type=int, default=3)
    parser.add_argument("--rnn-hidden", type=int, default=500)
    parser.add_argument("--dense-dim", type=int, default=500)
    parser.add_argument("--nf-layers", type=int, default=20)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--test-batch-size", type=int, default=50)
    parser.add_argument("--test-n-z", type=int, default=1)
    parser.add_argument("--valid-portion", type=float, default=0.3)
    parser.add_argument("--initial-lr", type=float, default=0.001)
    parser.add_argument("--lr-anneal-factor", type=float, default=0.5)
    parser.add_argument("--lr-anneal-epoch-freq", type=int, default=40)
    parser.add_argument("--std-epsilon", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--valid-step-freq", type=int, default=100)
    parser.add_argument("--posterior-flow-type", default="nf")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    arrays = np.load(args.input)
    train = np.asarray(arrays["train"], dtype=np.float32)
    test = np.asarray(arrays["test"], dtype=np.float32)
    x_dim = int(train.shape[1])
    config = Config(
        x_dim=x_dim,
        use_connected_z_q=True,
        use_connected_z_p=True,
        z_dim=args.z_dim,
        rnn_cell="GRU",
        rnn_num_hidden=args.rnn_hidden,
        window_length=args.window_length,
        dense_dim=args.dense_dim,
        posterior_flow_type=args.posterior_flow_type if args.posterior_flow_type != "None" else None,
        nf_layers=args.nf_layers,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        test_n_z=args.test_n_z,
        initial_lr=args.initial_lr,
        lr_anneal_epoch_freq=args.lr_anneal_epoch_freq,
        lr_anneal_factor=args.lr_anneal_factor,
        std_epsilon=args.std_epsilon,
        gradient_clip_norm=args.gradient_clip_norm,
        valid_step_freq=args.valid_step_freq,
        get_score_on_dim=False,
    )

    sess_config = tf.ConfigProto(device_count={"GPU": 0})
    sess_config.gpu_options.allow_growth = True
    with tf.variable_scope("model") as model_vs:
        model = OmniAnomaly(config=config, name="model")
        trainer = Trainer(
            model=model,
            model_vs=model_vs,
            max_epoch=args.max_epoch,
            batch_size=args.batch_size,
            valid_batch_size=args.test_batch_size,
            initial_lr=args.initial_lr,
            lr_anneal_epochs=args.lr_anneal_epoch_freq,
            lr_anneal_factor=args.lr_anneal_factor,
            grad_clip_norm=args.gradient_clip_norm,
            valid_step_freq=args.valid_step_freq,
        )
        predictor = Predictor(
            model,
            batch_size=args.test_batch_size,
            n_z=args.test_n_z,
            last_point_only=True,
        )

        with tf.Session(config=sess_config).as_default():
            if args.max_epoch > 0:
                trainer.fit(train, valid_portion=args.valid_portion)
            train_score, _train_z, _ = predictor.get_score(train)
            test_score, _test_z, _ = predictor.get_score(test)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        train_score=-np.asarray(train_score, dtype=np.float64),
        test_score=-np.asarray(test_score, dtype=np.float64),
    )


if __name__ == "__main__":
    main()
