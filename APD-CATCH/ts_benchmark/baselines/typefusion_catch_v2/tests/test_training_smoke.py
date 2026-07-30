import unittest

import torch

from .common import make_model


class TrainingSmokeTests(unittest.TestCase):
    def test_phase_b_forward_backward(self):
        model = make_model().train()
        x = torch.randn(2, model.config.seq_len, model.config.c_in)
        intervention = model.intervention_generator.generate(x, validation=True, sample_indices=range(2))
        output = model(x, intervention=intervention)
        self.assertTrue(torch.isfinite(output["losses"]["total"]))
        output["losses"]["total"].backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable))
        self.assertTrue(all(p.grad is None for p in model.anchor.parameters()))
