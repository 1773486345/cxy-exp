from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_apd_catch_paper import (
    ASD_FILES,
    apd_params,
    expand_datasets,
    load_dataset,
    missing_data_files,
    paper_group,
    write_summaries,
)
from ts_benchmark.baselines.apd_catch import APDCATCH


class PaperRunnerTest(unittest.TestCase):
    def test_paper_dataset_expansion_covers_all_source_files(self):
        files = expand_datasets(["all"])
        self.assertEqual(len(files), 23)
        self.assertEqual([name for name in files if name.startswith("ASD_")], ASD_FILES)
        self.assertEqual(paper_group("ASD_dataset_7.csv"), "ASD")
        self.assertEqual(paper_group("Genesis.csv"), "Genesis")

    def test_original_catch_parameters_are_mapped_without_score_leakage(self):
        params, original = apd_params("PSM.csv", "adaptive", 17)
        self.assertEqual(params["variant"], "adaptive")
        self.assertEqual(params["seed"], 17)
        self.assertEqual(params["seq_len"], original["seq_len"])
        self.assertEqual(params["d_model"], original["d_model"])
        self.assertIn("score_lambda", original)
        self.assertNotIn("score_lambda", params)
        self.assertNotIn("anomaly_ratio", params)

    def test_all_paper_configs_build_with_equal_variant_budgets(self):
        for file_name in expand_datasets(["all"]):
            parameter_counts = []
            for variant in ("causal_catch", "fixed", "adaptive"):
                params, _ = apd_params(file_name, variant, 17)
                detector = APDCATCH(**params)
                detector._build_model(n_vars=3)
                parameter_counts.append(
                    sum(
                        parameter.numel()
                        for parameter in detector.model.parameters()
                        if parameter.requires_grad
                    )
                )
            self.assertEqual(
                len(set(parameter_counts)),
                1,
                msg=f"variant parameter mismatch for {file_name}: {parameter_counts}",
            )

    def test_official_long_form_data_is_loaded_and_split_from_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root = Path(temporary) / "dataset" / "anomaly_detect"
            data_root = dataset_root / "data"
            data_root.mkdir(parents=True)
            length = 14
            labels = np.zeros(length, dtype=np.int64)
            labels[11:13] = 1
            rows = []
            for column, values in (
                ("channel_1", np.linspace(0.0, 1.0, length)),
                ("channel_2", np.linspace(1.0, 2.0, length)),
                ("label", labels),
            ):
                rows.extend(
                    {
                        "date": index,
                        "data": value,
                        "cols": column,
                    }
                    for index, value in enumerate(values)
                )
            pd.DataFrame(rows).to_csv(data_root / "Genesis.csv", index=False)
            pd.DataFrame(
                [{"file_name": "Genesis.csv", "train_lens": 9}]
            ).to_csv(dataset_root / "DETECT_META.csv", index=False)

            self.assertEqual(
                missing_data_files(Path(temporary) / "dataset", ["Genesis.csv"]), []
            )
            train, train_labels, test, test_labels = load_dataset(
                Path(temporary) / "dataset", "Genesis.csv"
            )
            self.assertEqual(train.shape, (9, 2))
            self.assertEqual(test.shape, (5, 2))
            np.testing.assert_array_equal(train_labels, labels[:9])
            np.testing.assert_array_equal(test_labels, labels[9:])

    def test_asd_subseries_are_aggregated_before_paper_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            for index, auc_roc in ((1, 0.7), (2, 0.9)):
                path = output_root / f"ASD_dataset_{index}" / "adaptive" / "seed_17.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "dataset_file": f"ASD_dataset_{index}.csv",
                            "paper_dataset": "ASD",
                            "variant": "adaptive",
                            "seed": 17,
                            "timing_seconds": {"fit": 1.0, "inference": 0.5},
                            "fit_summary": {
                                "trainable_parameters": 10,
                                "epochs": 1,
                            },
                            "metrics": {
                                "auc_roc": auc_roc,
                                "affiliation_f": 0.6 + index * 0.1,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            write_summaries(output_root)
            summary = pd.read_csv(output_root / "summary_paper_comparison.csv")
            self.assertEqual(len(summary), 1)
            self.assertAlmostEqual(summary.loc[0, "auc_roc"], 0.8)
            self.assertAlmostEqual(summary.loc[0, "paper_catch_auc_roc"], 0.824)
            self.assertAlmostEqual(
                summary.loc[0, "delta_auc_roc_vs_paper_catch"], -0.024
            )


if __name__ == "__main__":
    unittest.main()
