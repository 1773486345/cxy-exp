import unittest

import torch

from .common import make_model


class TrainingSmokeTests(unittest.TestCase):
    def test_single_stage_backward_all_required_modules(self):
        model = make_model().train()
        x = torch.randn(2, model.config.seq_len, model.config.c_in)
        intervention = model.intervention_generator.generate(x, validation=True, sample_indices=[0, 1])
        output = model(x, intervention=intervention)
        output["losses"]["total"].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
