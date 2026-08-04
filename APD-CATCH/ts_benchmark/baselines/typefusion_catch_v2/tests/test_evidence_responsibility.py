import unittest

import torch

from .common import make_model


class EvidenceResponsibilityTests(unittest.TestCase):
    def test_compound_targets_are_multi_hot_and_noncompetitive(self):
        model = make_model()
        x = torch.randn(2, model.config.seq_len, model.config.c_in)
        intervention = model.intervention_generator.generate(x, validation=True, sample_indices=[100, 101])
        output = model(x, intervention=intervention)
        self.assertTrue(torch.isfinite(output["losses"]["responsibility"]))
        targets = output["intervention_output"]["type_targets"]
        self.assertTrue(torch.all((targets.sum(dim=1) == 0) | (targets.sum(dim=1) == 1) | (targets.sum(dim=1) == 2)))
