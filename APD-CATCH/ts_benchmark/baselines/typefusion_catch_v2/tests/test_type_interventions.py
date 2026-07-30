import unittest

import torch

from ts_benchmark.baselines.typefusion_catch_v2.layers.type_interventions import TypeInterventionGenerator


class InterventionTests(unittest.TestCase):
    def test_shapes_and_validation_reproducibility(self):
        x = torch.randn(8, 32, 4)
        generator = TypeInterventionGenerator(2021)
        first = generator.generate(x, validation=True, sample_indices=range(8))
        second = generator.generate(x, validation=True, sample_indices=range(8))
        for key in ("corrupted_x", "type_targets", "type_masks", "union_mask"):
            self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertEqual(tuple(first["type_masks"].shape), (8, 4, 32, 4))
