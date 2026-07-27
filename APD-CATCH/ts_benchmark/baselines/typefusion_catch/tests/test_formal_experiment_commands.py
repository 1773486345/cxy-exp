"""Static checks for the two prepared formal PSM commands."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
import unittest


class FormalExperimentCommandTests(unittest.TestCase):
    @staticmethod
    def _commands() -> list[list[str]]:
        repository = Path(__file__).resolve().parents[4]
        command_file = repository / "TYPEFUSION_CATCH_EXPERIMENT_COMMANDS.md"
        command_lines = [
            line.strip()
            for line in command_file.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("python ")
        ]
        if len(command_lines) != 2:
            raise AssertionError("the formal command document must contain exactly two commands")
        for line in command_lines:
            if any(token in line for token in ("&", ";", "&&", "||")):
                raise AssertionError(f"command is not independent: {line}")
            if re.search(r"\b(?:for|while|until|do|done)\b", line):
                raise AssertionError(f"command must not contain a shell loop: {line}")
        return [shlex.split(line) for line in command_lines]

    @staticmethod
    def _argument(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    def test_commands_are_equal_protocol_and_typefusion_is_explicit(self) -> None:
        catch_command, typefusion_command = self._commands()
        catch_params = json.loads(self._argument(catch_command, "--model-hyper-params"))
        typefusion_params = json.loads(
            self._argument(typefusion_command, "--model-hyper-params")
        )
        repository = Path(__file__).resolve().parents[4]
        evaluator_config = json.loads(
            (repository / "config/unfixed_detect_score_multi_config.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(self._argument(catch_command, "--model-name"), "catch.CATCH")
        self.assertEqual(
            self._argument(typefusion_command, "--model-name"), "typefusion_catch.TypeFusionCATCH"
        )
        for command in (catch_command, typefusion_command):
            self.assertEqual(self._argument(command, "--seed"), "2021")
            self.assertEqual(self._argument(command, "--data-name-list"), "PSM.csv")
            self.assertEqual(
                self._argument(command, "--config-path"), "unfixed_detect_score_multi_config.json"
            )
            self.assertEqual(self._argument(command, "--gpus"), "0")
            self.assertEqual(self._argument(command, "--num-workers"), "1")
            self.assertEqual(self._argument(command, "--timeout"), "60000")
            self.assertNotIn("--data-set-name", command)

        self.assertEqual(
            evaluator_config["evaluation_config"]["strategy_args"]["strategy_name"],
            "unfixed_detect_score",
        )
        self.assertEqual(evaluator_config["evaluation_config"]["strategy_args"]["seed"], 2021)

        for field in ("seq_len", "patch_size", "patch_stride", "batch_size", "lr"):
            self.assertEqual(catch_params[field], typefusion_params[field], field)
        self.assertEqual(catch_params["num_epochs"], typefusion_params["catch_train_epochs"])
        self.assertEqual(typefusion_params["seed"], 2021)
        self.assertEqual(typefusion_params["fit_mode"], "three_stage")
        self.assertEqual(typefusion_params["training_budget_mode"], "equal_total_steps")
        self.assertEqual(typefusion_params["joint_finetune_lr_scale"], 0.1)
        self.assertEqual(typefusion_params["lambda_freq"], 0.1)
        self.assertEqual(typefusion_params["lambda_mask"], 0.1)

    def test_commands_exclude_forbidden_evaluation_shortcuts(self) -> None:
        repository = Path(__file__).resolve().parents[4]
        document = (repository / "TYPEFUSION_CATCH_EXPERIMENT_COMMANDS.md").read_text(
            encoding="utf-8"
        ).lower()
        command_text = "\n".join(" ".join(command) for command in self._commands()).lower()
        for forbidden in (
            "detect_label",
            "score_fusion",
            "threshold",
            "calibration",
            "test_label",
            "test-label",
            "&",
        ):
            self.assertNotIn(forbidden, command_text)
        self.assertIn("equal_total_steps", document)


if __name__ == "__main__":
    unittest.main()
