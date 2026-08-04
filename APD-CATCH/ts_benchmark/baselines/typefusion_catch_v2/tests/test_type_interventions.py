import unittest

import torch

from ts_benchmark.baselines.typefusion_catch_v2.layers.type_interventions import TypeInterventionGenerator


class InterventionTests(unittest.TestCase):
    def test_validation_reproducibility_and_training_rng_progress(self):
        x = torch.randn(8, 32, 4)
        generator = TypeInterventionGenerator(2021)
        first = generator.generate(x, validation=True, sample_indices=range(8))
        second = generator.generate(x, validation=True, sample_indices=range(8))
        self.assertTrue(torch.equal(first["corrupted_x"], second["corrupted_x"]))
        training_first = generator.generate(x)
        training_second = generator.generate(x)
        self.assertFalse(torch.equal(training_first["corrupted_x"], training_second["corrupted_x"]))

    def test_shapes_and_batch_one_donor(self):
        x = torch.arange(32.0).view(1, 32, 1).repeat(1, 1, 4)
        result = TypeInterventionGenerator(3).generate(x, validation=True, sample_indices=[0])
        self.assertEqual(tuple(result["type_masks"].shape), (1, 4, 32, 4))
        self.assertEqual(tuple(result["union_mask"].shape), (1, 32))
