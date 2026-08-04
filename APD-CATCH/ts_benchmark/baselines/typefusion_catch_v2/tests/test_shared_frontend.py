import unittest

import torch

from .common import tiny_config
from ts_benchmark.baselines.typefusion_catch_v2.layers.shared_representation_frontend import SharedRepresentationFrontend


class SharedFrontendTests(unittest.TestCase):
    def test_shapes_and_frequency_gradients(self):
        config = tiny_config()
        frontend = SharedRepresentationFrontend(config)
        x = torch.randn(2, config.seq_len, config.c_in)
        output = frontend(x)
        self.assertEqual(output["h_time"].shape, (2, config.seq_len, config.d_model))
        self.assertEqual(output["h_freq"].shape, (2, config.num_patches, config.c_in, config.d_model))
        loss = output["h_freq"].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(frontend.frequency_patch_embedding.weight.grad).all())
