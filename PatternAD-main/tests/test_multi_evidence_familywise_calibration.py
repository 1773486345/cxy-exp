import unittest

import numpy as np

from scripts.multi_evidence.familywise_calibration import (
    DEFAULT_COMPONENT_ALPHAS,
    FamilyWiseEvidenceReliabilityCalibration,
)
from scripts.multi_evidence.multi_target_familywise_calibration import (
    MultiTargetFamilyWiseEvidenceReliabilityCalibration,
)
from scripts.multi_evidence.reliability_calibration import conformal_upper_threshold


COMPONENTS = ("temporal_residual", "cross_residual", "disagreement")


class FamilyWiseCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(813)
        cls.windows = rng.normal(size=(720, 9, 4)).astype(np.float32)
        cls.raw = {
            component: rng.uniform(0.0, 3.0, size=720)
            for component in COMPONENTS
        }

    def _fit(self):
        return FamilyWiseEvidenceReliabilityCalibration(
            dimensions=4,
            target_index=1,
            component_alphas=DEFAULT_COMPONENT_ALPHAS,
            reliability_strata=3,
            min_reference_per_stratum=50,
            min_outer_per_stratum=50,
        ).fit(
            self.windows[:240],
            self.windows[240:480],
            {name: values[240:480] for name, values in self.raw.items()},
            self.windows[480:],
            {name: values[480:] for name, values in self.raw.items()},
        )

    def test_thresholds_are_per_stratum_and_component_alpha(self):
        calibration = self._fit()
        outer = calibration.transform(
            self.windows[480:], {name: values[480:] for name, values in self.raw.items()}
        )
        metadata = calibration.metadata()
        self.assertEqual(metadata["threshold_scope"], "per_reliability_stratum")
        self.assertEqual(
            metadata["threshold_fit_split"],
            "outer_calibration_normal_per_reliability_stratum",
        )
        self.assertEqual(metadata["component_alphas"], DEFAULT_COMPONENT_ALPHAS)
        self.assertEqual(metadata["familywise_control"]["method"], "bonferroni")
        for component in COMPONENTS:
            tail_name = f"{component}_tail"
            stratum_name = f"{component}_reliability_stratum"
            strata = outer[stratum_name].astype(np.int64)
            expected = np.empty(len(strata), dtype=np.float64)
            for stratum in range(3):
                mask = strata == stratum
                threshold = conformal_upper_threshold(
                    outer[tail_name][mask], DEFAULT_COMPONENT_ALPHAS[component]
                )
                self.assertAlmostEqual(
                    calibration.thresholds_[component][stratum], threshold
                )
                expected[mask] = threshold
            np.testing.assert_allclose(
                calibration.threshold_array(outer, tail_name), expected
            )
            self.assertTrue(
                np.array_equal(
                    calibration.exceeds(outer, tail_name), outer[tail_name] > expected
                )
            )

    def test_outer_support_fails_closed_without_global_fallback(self):
        calibration = FamilyWiseEvidenceReliabilityCalibration(
            dimensions=4,
            target_index=1,
            reliability_strata=3,
            min_reference_per_stratum=50,
            min_outer_per_stratum=50,
        )
        with self.assertRaisesRegex(ValueError, "Outer calibration has only"):
            calibration.fit(
                self.windows[:240],
                self.windows[240:480],
                {name: values[240:480] for name, values in self.raw.items()},
                self.windows[480:520],
                {name: values[480:520] for name, values in self.raw.items()},
            )

    def test_features_remain_terminal_blind_and_input_restricted(self):
        calibration = self._fit()
        base = self.windows[:1].copy()
        drivers_changed = base.copy()
        drivers_changed[:, :, [0, 2, 3]] += np.linspace(0.0, 3.0, 9)[None, :, None]
        terminal_changed = base.copy()
        terminal_changed[:, -1, 1] += 5.0
        target_changed = base.copy()
        target_changed[:, :, 1] += np.linspace(0.0, 3.0, 9)[None, :]
        base_features = calibration.features(base)
        driver_features = calibration.features(drivers_changed)
        terminal_features = calibration.features(terminal_changed)
        target_features = calibration.features(target_changed)
        np.testing.assert_allclose(
            base_features["temporal_residual"], driver_features["temporal_residual"]
        )
        np.testing.assert_allclose(
            base_features["disagreement"], terminal_features["disagreement"]
        )
        np.testing.assert_allclose(
            base_features["cross_residual"], target_features["cross_residual"]
        )

    def test_target_calibrations_are_matrix_isolated(self):
        rng = np.random.default_rng(999)
        raw = {component: rng.uniform(0.0, 3.0, size=(720, 2)) for component in COMPONENTS}
        first = MultiTargetFamilyWiseEvidenceReliabilityCalibration(
            dimensions=4,
            target_indices=(0, 3),
            min_reference_per_stratum=50,
            min_outer_per_stratum=50,
        ).fit(
            self.windows[:240],
            self.windows[240:480],
            {name: values[240:480] for name, values in raw.items()},
            self.windows[480:],
            {name: values[480:] for name, values in raw.items()},
        )
        changed = {name: values.copy() for name, values in raw.items()}
        changed["cross_residual"][240:, 0] += 100.0
        second = MultiTargetFamilyWiseEvidenceReliabilityCalibration(
            dimensions=4,
            target_indices=(0, 3),
            min_reference_per_stratum=50,
            min_outer_per_stratum=50,
        ).fit(
            self.windows[:240],
            self.windows[240:480],
            {name: values[240:480] for name, values in changed.items()},
            self.windows[480:],
            {name: values[480:] for name, values in changed.items()},
        )
        first_target_three = first.metadata()["calibration_by_target"]["3"]
        second_target_three = second.metadata()["calibration_by_target"]["3"]
        self.assertEqual(first_target_three, second_target_three)
        transformed = first.transform(
            self.windows[480:], {name: values[480:] for name, values in raw.items()}
        )
        self.assertEqual(transformed["cross_residual_tail"].shape, (240, 2))
        self.assertTrue(first.metadata()["no_target_flattening"])


if __name__ == "__main__":
    unittest.main()
