import unittest

import numpy as np
import torch
import torch.nn as nn

from ts_benchmark.baselines.PatternAD.PatternAD import (
    JointMultivariateReconstructor,
    PatternAD,
    PatternADConfig,
)
from ts_benchmark.baselines.PatternAD.utils.pattern_scoring import PatternAwareScorer


def _small_config(**overrides):
    values = {
        "enc_in": 2,
        "seq_len": 6,
        "d_model": 8,
        "d_ff": 16,
        "n_heads": 2,
        "e_layers": 1,
        "dropout": 0.0,
        "context_window": 3,
    }
    values.update(overrides)
    return PatternADConfig(**values)


class _MaskEchoModel(nn.Module):
    def forward(self, x, mask=None):
        mask_value = torch.zeros_like(x) if mask is None else mask.to(x.dtype)
        return {
            "mean": mask_value,
            "scale": 1.0 + mask_value,
            "df": 4.0 + mask_value,
        }


class PatternADCoreTest(unittest.TestCase):
    def test_score_diagnostics_append_and_summarize_scale(self):
        detector = PatternAD(
            enc_in=2,
            seq_len=6,
            score_mask_ratio=0.5,
            reconstruction_distribution="gaussian",
            pattern_score_mode="nll",
        )
        detector.device = torch.device("cpu")
        detector.model = _MaskEchoModel()
        detector.pattern_scorer = PatternAwareScorer(
            score_mode="nll", distribution="gaussian"
        )
        detector.pattern_scorer.fitted = True
        batch = torch.zeros(2, 6, 2)
        loader = [
            (batch, None, None, None),
            (batch, None, None, None),
        ]

        score = detector._collect_multi_scores(loader, total_len=9)
        self.assertEqual(score.shape, (9,))
        first = detector.get_diagnostics()["score_calls"][0]
        self.assertEqual(first["call_index"], 0)
        self.assertEqual(first["batch_count"], 2)
        self.assertEqual(first["window_count"], 4)
        self.assertEqual(first["scale"]["count"], 48)
        self.assertEqual(first["scale"]["finite_count"], 48)
        self.assertAlmostEqual(first["scale"]["min"], 2.0)
        self.assertAlmostEqual(first["scale"]["max"], 2.0)
        self.assertAlmostEqual(first["scale"]["mean"], 2.0)
        self.assertEqual(first["scale"]["lower_bound_count"], 0)
        self.assertEqual(first["scale"]["upper_bound_count"], 0)
        components = detector.get_last_score_components()
        self.assertEqual(
            set(components),
            {
                "raw_squared_residual",
                "standardized_squared_residual",
                "predicted_scale",
                "log_scale",
            },
        )
        for values in components.values():
            self.assertEqual(values.shape, (9,))
            self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(components["raw_squared_residual"], 1.0)
        np.testing.assert_allclose(
            components["standardized_squared_residual"], 0.25
        )
        np.testing.assert_allclose(components["predicted_scale"], 2.0)
        np.testing.assert_allclose(components["log_scale"], np.log(2.0))

        detector._collect_multi_scores([(batch[:1], None, None, None)], 6)
        calls = detector.get_diagnostics()["score_calls"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["call_index"], 1)

    def test_context_features_ignore_values_at_masked_positions(self):
        model = JointMultivariateReconstructor(_small_config())
        x = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, 2, 0] = True
        mask[:, 4, 1] = True

        changed = x.clone()
        changed[mask] = torch.tensor([1e5, -1e5])
        original_features = model._context_features(x, mask)
        changed_features = model._context_features(changed, mask)

        torch.testing.assert_close(original_features, changed_features)

        model.eval()
        masked_x = x.masked_fill(mask, 0.0)
        masked_changed = changed.masked_fill(mask, 0.0)
        original_output = model(masked_x, mask)
        changed_output = model(masked_changed, mask)
        torch.testing.assert_close(original_output, changed_output)

    def test_context_off_uses_a_learned_constant_through_the_same_path(self):
        model = JointMultivariateReconstructor(
            _small_config(use_context_conditioning=False)
        )
        first = model._conditioning_features(
            torch.randn(2, 6, 2), torch.rand(2, 6, 2) > 0.5
        )
        second = model._conditioning_features(
            torch.randn(2, 6, 2) * 100.0, torch.rand(2, 6, 2) > 0.5
        )
        torch.testing.assert_close(first, second)

        model(torch.randn(2, 6, 2)).sum().backward()
        self.assertIsNotNone(model.context_control.grad)

    def test_context_scale_prior_is_target_blind_and_regime_sensitive(self):
        model = JointMultivariateReconstructor(
            _small_config(
                reconstruction_distribution="gaussian",
                use_context_scale_prior=True,
                context_scale_floor=0.01,
                context_scale_prior_mix=0.5,
            )
        )
        model.eval()
        quiet = torch.tensor(
            [[[0.00, 0.00], [0.10, -0.10], [0.00, 0.00], [-0.10, 0.10], [0.00, 0.00], [0.10, -0.10]]]
        )
        volatile = quiet * 8.0
        mask = torch.zeros_like(quiet, dtype=torch.bool)
        mask[:, 2, :] = True

        quiet_output = model(quiet.masked_fill(mask, 0.0), mask)
        volatile_output = model(volatile.masked_fill(mask, 0.0), mask)
        self.assertTrue(
            torch.all(volatile_output["scale"][mask] > quiet_output["scale"][mask] * 2.0)
        )

        changed = quiet.clone()
        changed[mask] = 1e5
        changed_output = model(changed.masked_fill(mask, 0.0), mask)
        torch.testing.assert_close(quiet_output["scale"], changed_output["scale"])

    def test_context_off_scale_prior_is_regime_invariant(self):
        model = JointMultivariateReconstructor(
            _small_config(
                reconstruction_distribution="gaussian",
                use_context_conditioning=False,
                use_context_scale_prior=True,
            )
        )
        model.eval()
        mask = torch.zeros(1, 6, 2, dtype=torch.bool)
        mask[:, 2, :] = True
        quiet = torch.randn(1, 6, 2) * 0.1
        volatile = quiet * 10.0

        quiet_output = model(quiet.masked_fill(mask, 0.0), mask)
        volatile_output = model(volatile.masked_fill(mask, 0.0), mask)
        torch.testing.assert_close(quiet_output["scale"], volatile_output["scale"])

    def test_visible_trend_bridges_masked_transition_and_suppresses_scale(self):
        model = JointMultivariateReconstructor(
            _small_config(
                reconstruction_distribution="gaussian",
                use_context_scale_prior=True,
                context_window=5,
                context_scale_prior_mix=1.0,
                context_transition_scale_suppression=2.0,
            )
        )
        alternating = torch.tensor(
            [[[-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]]]
        )
        transition = torch.tensor(
            [[[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]
        )
        mask = torch.zeros_like(transition, dtype=torch.bool)
        mask[:, 2, :] = True

        alternating_stats = model._visible_context_statistics(
            alternating.masked_fill(mask, 0.0), mask
        )
        transition_stats = model._visible_context_statistics(
            transition.masked_fill(mask, 0.0), mask
        )
        self.assertTrue(
            torch.all(
                transition_stats["trend"][mask].abs()
                > alternating_stats["trend"][mask].abs()
            )
        )
        alternating_prior = model._contextual_scale_prior(
            alternating, alternating_stats
        )
        transition_prior = model._contextual_scale_prior(transition, transition_stats)
        self.assertTrue(torch.all(transition_prior[mask] < alternating_prior[mask]))

    def test_d2_complementary_masks_cover_each_position_once(self):
        detector = PatternAD(
            enc_in=2,
            seq_len=6,
            score_mask_ratio=0.35,
            d_model=8,
            d_ff=16,
            n_heads=2,
        )
        x = torch.zeros(3, 6, 2)
        masks = detector._complementary_score_masks(x)

        self.assertEqual(len(masks), 2)
        coverage = torch.stack(masks).sum(dim=0)
        self.assertTrue(torch.equal(coverage, torch.ones_like(coverage)))
        for mask in masks:
            self.assertTrue(torch.equal(mask.sum(dim=-1), torch.ones(3, 6, dtype=torch.long)))

        detector.model = _MaskEchoModel()
        outputs, score_mask = detector._predict_for_scoring(x)
        torch.testing.assert_close(outputs["mean"], torch.ones_like(x))
        torch.testing.assert_close(outputs["scale"], torch.full_like(x, 2.0))
        self.assertTrue(score_mask.all())

        unmasked_detector = PatternAD(
            enc_in=2,
            seq_len=6,
            score_mask_ratio=0.35,
            use_conditional_scoring=False,
        )
        unmasked_detector.model = _MaskEchoModel()
        forced_outputs, forced_mask = unmasked_detector._predict_for_scoring(
            x, force_conditional=True
        )
        torch.testing.assert_close(forced_outputs["mean"], torch.ones_like(x))
        self.assertTrue(forced_mask.all())

    def test_distribution_variants_have_equal_parameter_count(self):
        counts = []
        for distribution in ("mse", "gaussian", "student_t"):
            model = JointMultivariateReconstructor(
                _small_config(distribution_mode=distribution)
            )
            counts.append(sum(parameter.numel() for parameter in model.parameters()))
            output = model(torch.randn(2, 6, 2))
            if distribution == "mse":
                self.assertEqual(output.shape, (2, 6, 2))
            else:
                self.assertEqual(output["mean"].shape, (2, 6, 2))
                self.assertTrue((output["scale"] > 0).all())
                self.assertTrue((output["df"] > 2).all())
        self.assertEqual(len(set(counts)), 1)

        context_on = JointMultivariateReconstructor(
            _small_config(use_context_conditioning=True)
        )
        context_off = JointMultivariateReconstructor(
            _small_config(use_context_conditioning=False)
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in context_on.parameters()),
            sum(parameter.numel() for parameter in context_off.parameters()),
        )

        self.assertEqual(PatternADConfig().pattern_score_mode, "raw")
        self.assertEqual(
            PatternADConfig(distribution_mode="gaussian").pattern_score_mode,
            "tail_probability",
        )

    def test_train_masks_use_a_dedicated_seeded_rng(self):
        x = torch.zeros(8, 12, 4)
        left = PatternAD(train_mask_seed=17)
        right = PatternAD(train_mask_seed=17)

        left_first = left._mask_input(x)[1]
        torch.rand(1000)
        right_first = right._mask_input(x)[1]
        torch.testing.assert_close(left_first, right_first)

        torch.rand(777)
        left_second = left._mask_input(x)[1]
        torch.rand(333)
        right_second = right._mask_input(x)[1]
        torch.testing.assert_close(left_second, right_second)

        other = PatternAD(train_mask_seed=18)
        self.assertFalse(torch.equal(left_first, other._mask_input(x)[1]))

    def test_model_initialization_repeats_for_same_seed_and_changes_for_new_seed(self):
        torch.manual_seed(31)
        first = JointMultivariateReconstructor(_small_config())
        torch.manual_seed(31)
        repeated = JointMultivariateReconstructor(_small_config())
        torch.manual_seed(32)
        changed = JointMultivariateReconstructor(_small_config())

        first_parameters = torch.cat([value.flatten() for value in first.parameters()])
        repeated_parameters = torch.cat(
            [value.flatten() for value in repeated.parameters()]
        )
        changed_parameters = torch.cat(
            [value.flatten() for value in changed.parameters()]
        )
        torch.testing.assert_close(first_parameters, repeated_parameters)
        self.assertFalse(torch.equal(first_parameters, changed_parameters))

    def test_gaussian_nll_distinguishes_same_residual_by_scale(self):
        scorer = PatternAwareScorer(score_mode="nll", distribution="gaussian")
        true = np.array([[[0.5]]], dtype=np.float64)
        pred = np.zeros_like(true)
        scorer.fit(true, pred)

        narrow = scorer.score_windows(
            true,
            pred,
            distribution_params={"scale": np.full_like(true, 0.25)},
        )
        wide = scorer.score_windows(
            true,
            pred,
            distribution_params={"scale": np.full_like(true, 1.0)},
        )

        self.assertLess(float(wide.item()), float(narrow.item()))

    def test_gaussian_tail_scores_rarity_instead_of_cross_scale_density(self):
        true = np.array([[[0.1]]], dtype=np.float64)
        pred = np.zeros_like(true)
        narrow_scale = {"scale": np.full_like(true, 0.25)}
        wide_scale = {"scale": np.full_like(true, 1.0)}

        nll = PatternAwareScorer(score_mode="nll", distribution="gaussian")
        nll.fit(true, pred)
        tail = PatternAwareScorer(
            score_mode="tail_probability", distribution="gaussian"
        )
        tail.fit(true, pred)

        self.assertGreater(
            float(nll.score_windows(true, pred, distribution_params=wide_scale).item()),
            float(nll.score_windows(true, pred, distribution_params=narrow_scale).item()),
        )
        self.assertLess(
            float(tail.score_windows(true, pred, distribution_params=wide_scale).item()),
            float(tail.score_windows(true, pred, distribution_params=narrow_scale).item()),
        )

    def test_raw_score_matches_masked_mean_squared_error(self):
        scorer = PatternAwareScorer(score_mode="raw")
        true = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        pred = np.zeros_like(true)
        mask = np.array([[[True, False], [True, True]]])

        score = scorer.score_windows(true, pred, score_mask=mask)

        np.testing.assert_allclose(score, np.array([[1.0, 12.5]]))

    def test_student_t_loss_is_finite_and_backward_compatible_mse_is_unchanged(self):
        target = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
        mask = torch.tensor([[[True, False], [False, True]]])

        raw_detector = PatternAD(reconstruction_full_loss_weight=0.1)
        raw_mean = torch.tensor([[[1.0, 1.0], [2.0, 1.0]]])
        raw_outputs = {
            "mean": raw_mean,
            "scale": torch.ones_like(raw_mean),
            "df": torch.full_like(raw_mean, 4.0),
        }
        expected = ((raw_mean - target) ** 2)[mask].mean()
        expected = expected + 0.1 * ((raw_mean - target) ** 2).mean()
        torch.testing.assert_close(
            raw_detector._reconstruction_loss(raw_outputs, target, mask), expected
        )

        student_detector = PatternAD(distribution_mode="student_t")
        mean = torch.zeros_like(target, requires_grad=True)
        raw_scale = torch.zeros_like(target, requires_grad=True)
        raw_df = torch.zeros_like(target, requires_grad=True)
        student_outputs = {
            "mean": mean,
            "scale": torch.nn.functional.softplus(raw_scale) + 1e-3,
            "df": 2.0 + torch.nn.functional.softplus(raw_df),
        }
        loss = student_detector._reconstruction_loss(student_outputs, target, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for value in (mean.grad, raw_scale.grad, raw_df.grad):
            self.assertIsNotNone(value)
            self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.equal(raw_scale.grad[~mask], torch.zeros_like(raw_scale.grad[~mask])))
        self.assertTrue(torch.equal(raw_df.grad[~mask], torch.zeros_like(raw_df.grad[~mask])))


if __name__ == "__main__":
    unittest.main()
