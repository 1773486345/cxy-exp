import unittest

import torch

from .common import tiny_config
from ts_benchmark.baselines.typefusion_catch_v2.layers.relation_aware_joint_scorer import RelationAwareJointScorer


class JointScorerTests(unittest.TestCase):
    def test_ten_tokens_and_bounded_relation_delta(self):
        config = tiny_config()
        scorer = RelationAwareJointScorer(config)
        tokens = torch.randn(2, config.seq_len, 4, config.joint_dim)
        evidence = torch.randn(2, config.seq_len, 4)
        context = torch.randn(2, config.seq_len, config.joint_dim)
        output = scorer(tokens, evidence, context)
        self.assertEqual(output["joint_score"].shape, (2, config.seq_len))
        self.assertLessEqual(output["relation_delta"].abs().max().item(), config.relation_correction_cap + 1e-6)
        self.assertTrue(torch.isfinite(output["joint_score"]).all())
