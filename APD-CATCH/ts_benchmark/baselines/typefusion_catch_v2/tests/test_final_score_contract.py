import unittest

import torch

from .common import make_model


class FinalScoreTests(unittest.TestCase):
    def test_only_joint_softplus_is_formal_score(self):
        model = make_model().eval()
        output = model(torch.randn(2, model.config.seq_len, model.config.c_in), compute_loss=False)
        self.assertTrue(torch.equal(output["total_score"], output["joint_score"]))
        self.assertTrue(torch.allclose(output["joint_score"], torch.nn.functional.softplus(output["joint_logit"])))
        self.assertEqual(output["joint_score"].ndim, 2)
