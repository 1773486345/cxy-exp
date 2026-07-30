import unittest

import torch

from .common import make_model


class FinalScoreTests(unittest.TestCase):
    def test_score_is_softplus_joint_logit(self):
        model = make_model().eval()
        x = torch.randn(2, model.config.seq_len, model.config.c_in)
        output = model(x, compute_loss=False)
        self.assertTrue(torch.equal(output["total_score"], output["joint_score"]))
        self.assertTrue(torch.allclose(output["joint_score"], torch.nn.functional.softplus(output["joint_logit"])))
        self.assertEqual(output["joint_score"].shape, (2, model.config.seq_len))
