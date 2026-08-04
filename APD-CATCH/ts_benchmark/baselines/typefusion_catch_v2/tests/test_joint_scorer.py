import unittest

import torch

from .common import tiny_config
from ts_benchmark.baselines.typefusion_catch_v2.layers.relation_aware_joint_scorer import RelationAwareJointScorer


class JointScorerTests(unittest.TestCase):
    def test_pair_tokens_temporal_mixer_and_bound(self):
        config = tiny_config()
        scorer = RelationAwareJointScorer(config)
        tokens = torch.randn(2, config.seq_len, 4, config.joint_dim)
        logits = torch.randn(2, config.seq_len, 4)
        output = scorer(tokens, logits)
        self.assertEqual(output["relation_tokens"].shape[2], 6)
        self.assertEqual(output["joint_score"].shape, (2, config.seq_len))
        self.assertLessEqual(output["relation_delta"].abs().max().item(), config.relation_correction_cap + 1e-6)
