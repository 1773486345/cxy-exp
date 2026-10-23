import json

import pytest

from scripts.run_mnrf_branch_ablation import COMBINATIONS, EXPECTED_LAYOUT, build_plan, selected_indices
from scripts.summarize_mnrf_branch_ablation import contribution_deltas, segment_counts_from_axis


def test_matrix_has_all_nine_combinations_and_dimensions():
    assert [name for name, _ in COMBINATIONS] == [
        "S", "SE", "SP", "SR", "FULL", "WITHOUT_S", "WITHOUT_E", "WITHOUT_P", "WITHOUT_R"
    ]
    assert [len(selected_indices(views)) for _, views in COMBINATIONS] == [1, 9, 9, 9, 25, 24, 17, 17, 17]
    assert EXPECTED_LAYOUT["slices"] == {"S": [0, 1], "E": [1, 9], "P": [9, 17], "R": [17, 25]}


def test_plan_keeps_blocked_dataset_matrix_without_starting_work():
    plan = build_plan(["PSM"], validate_sources=False)
    assert plan["positions"] == 9
    assert len(plan["rows"]) == 9
    assert all(row["status"] == "blocked" for row in plan["rows"])
    assert all(row["selected_dim"] in {1, 9, 17, 24, 25} for row in plan["rows"])


def test_contribution_signs_match_removal_definition():
    values = {
        "S": {"average_precision": 0.40, "auc_roc": 0.60},
        "SE": {"average_precision": 0.50, "auc_roc": 0.70},
        "SP": {"average_precision": 0.39, "auc_roc": 0.59},
        "SR": {"average_precision": 0.40, "auc_roc": 0.60},
        "FULL": {"average_precision": 0.55, "auc_roc": 0.75},
        "WITHOUT_S": {"average_precision": 0.30, "auc_roc": 0.50},
        "WITHOUT_E": {"average_precision": 0.56, "auc_roc": 0.76},
        "WITHOUT_P": {"average_precision": 0.50, "auc_roc": 0.70},
        "WITHOUT_R": {"average_precision": 0.55, "auc_roc": 0.75},
    }
    deltas = contribution_deltas(values)
    assert deltas["E"]["added"]["average_precision"] == pytest.approx(0.10)
    assert deltas["E"]["deleted"]["average_precision"] == pytest.approx(-0.01)
    assert deltas["S"]["added"]["average_precision"] is None
    assert deltas["S"]["deleted"]["average_precision"] == pytest.approx(0.25)


def test_segment_counts_preserve_unscored_anomaly_and_segment_boundaries():
    all_global = list(range(20))
    all_labels = [0] * 20
    all_labels[10:15] = [1] * 5
    all_labels[17:19] = [1] * 2
    all_segments = [0] * 10 + [1] * 5 + [2] * 5
    test_global = list(range(10, 20))
    details = segment_counts_from_axis(all_labels, all_global, all_segments, test_global,
                                       [17], [1])
    assert details["total"] == 2
    assert details["scored"] == 1
    assert details["details"][0]["valid_points"] == 0
    assert details["details"][0]["no_valid_scores"] is True
    assert details["details"][1]["hit"] is True
