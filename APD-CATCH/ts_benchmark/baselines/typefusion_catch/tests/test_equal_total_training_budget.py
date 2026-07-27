"""Formal equal-total-steps budget tests on a tiny normal DataFrame."""

import unittest

from ts_benchmark.baselines.typefusion_catch.TypeFusionCATCH import TypeFusionCATCH
from ts_benchmark.baselines.typefusion_catch.tests.common import small_normal_frame, tiny_fit_kwargs


class EqualTotalTrainingBudgetTests(unittest.TestCase):
    def test_formal_budget_matches_reference_steps_without_validation_steps(self) -> None:
        adapter = TypeFusionCATCH(
            **tiny_fit_kwargs(catch_train_epochs=4, patience=10, training_budget_mode="equal_total_steps")
        )
        adapter.detect_fit(small_normal_frame())
        summary = adapter.training_budget_summary
        self.assertEqual(summary["mode"], "equal_total_steps")
        self.assertEqual(summary["reference_total_steps"], 4)
        self.assertEqual(summary["branch_pretrain_steps"], 1)
        self.assertEqual(summary["fusion_train_steps"], 1)
        self.assertEqual(summary["joint_finetune_steps"], 2)
        self.assertEqual(summary["actual_total_steps"], 4)
        self.assertEqual(sum(adapter.stage_optimizer_steps.values()), 4)
        self.assertLessEqual(summary["actual_total_steps"], summary["reference_total_steps"])

    def test_too_small_reference_budget_fails_explicitly(self) -> None:
        adapter = TypeFusionCATCH(**tiny_fit_kwargs(catch_train_epochs=2))
        with self.assertRaisesRegex(ValueError, "at least 3 reference optimizer steps"):
            adapter.detect_fit(small_normal_frame())

    def test_debug_epoch_budget_is_explicitly_distinct_from_formal_budget(self) -> None:
        adapter = TypeFusionCATCH(
            **tiny_fit_kwargs(
                fit_mode="single_stage",
                training_stage="branch_pretrain",
                training_budget_mode="debug_stage_epochs",
                branch_pretrain_epochs=2,
            )
        )
        adapter.detect_fit(small_normal_frame())
        self.assertEqual(adapter.training_budget_summary["mode"], "debug_stage_epochs")
        self.assertEqual(adapter.stage_optimizer_steps["branch_pretrain"], 2)

    def test_early_stopping_can_reduce_but_never_exceed_the_formal_budget(self) -> None:
        adapter = TypeFusionCATCH(
            **tiny_fit_kwargs(catch_train_epochs=9, patience=1, training_budget_mode="equal_total_steps")
        )
        # A constant validation loss gives one improvement then one stale
        # validation pass, exercising early stop before each 3-step allocation.
        adapter._validation_loss = lambda loader, compute_joint: 1.0
        adapter.detect_fit(small_normal_frame())
        summary = adapter.training_budget_summary
        self.assertLess(summary["actual_total_steps"], summary["reference_total_steps"])
        self.assertLessEqual(summary["actual_total_steps"], summary["reference_total_steps"])


if __name__ == "__main__":
    unittest.main()
