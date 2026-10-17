import copy

import pytest

from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig
from ts_benchmark.baselines.mnrf_lite_v2.layout import FeatureLayout
from scripts.validate_mnrf_lite_v2_run import validate_artifact_shapes


@pytest.mark.parametrize("dims", [(8, 8, 8), (4, 4, 4), (3, 5, 7), (8, 16, 32)])
def test_layout_round_trip_and_config_resolution(dims):
    layout = FeatureLayout(*dims)
    assert FeatureLayout.from_dict(layout.to_dict()).to_dict() == layout.to_dict()
    config = MNRFConfig.from_kwargs(evolution_dim=dims[0], pattern_dim=dims[1], relation_dim=dims[2])
    assert config.layout.to_dict() == layout.to_dict()


@pytest.mark.parametrize("bad_dimensions", [
    {"S": 2, "E": 3, "P": 5, "R": 7},
    {"S": True, "E": 3, "P": 5, "R": 7},
    {"E": 3, "P": 5, "R": 7},
    *[
        {"S": 1, **{name: (value if key == name else default)
                     for name, default in (("E", 3), ("P", 5), ("R", 7))}}
        for key in ("E", "P", "R")
        for value in (True, "3", 3.5, 0, -1)
    ],
])
def test_invalid_layout_dimensions_rejected_by_layout_and_config(bad_dimensions):
    valid = FeatureLayout(3, 5, 7).to_dict()
    valid["dimensions"] = bad_dimensions
    with pytest.raises(ValueError, match="layout|dimension|S must"):
        FeatureLayout.from_dict(valid)
    with pytest.raises(ValueError, match="layout|dimension|S must"):
        MNRFConfig.from_kwargs(order=valid["order"], dimensions=valid["dimensions"],
                               slices=valid["slices"], F=valid["F"],
                               evolution_dim=3, pattern_dim=5, relation_dim=7)


@pytest.mark.parametrize("field,value", [
    ("order", None), ("dimensions", None), ("slices", None), ("F", None),
])
def test_required_layout_fields_cannot_be_omitted(field, value):
    layout = FeatureLayout(3, 5, 7).to_dict()
    if value is None:
        layout.pop(field)
    else:
        layout[field] = value
    with pytest.raises(ValueError, match="required|missing|schema"):
        FeatureLayout.from_dict(layout)


@pytest.mark.parametrize("field,value", [
    ("order", ["S", "P", "E", "R"]),
    ("F", True),
    ("F", 17),
    ("slices", {"S": [0, True], "E": [1, 4], "P": [4, 9], "R": [9, 16]}),
    ("slices", {"S": [0, 1], "E": [1, 4], "P": [4, 9], "R": [9, 15]}),
])
def test_inconsistent_order_total_or_slice_types_rejected(field, value):
    layout = FeatureLayout(3, 5, 7).to_dict()
    layout[field] = value
    with pytest.raises(ValueError, match="layout|order|F|slice"):
        FeatureLayout.from_dict(layout)


def test_partial_layout_metadata_in_config_is_rejected():
    with pytest.raises(ValueError, match="complete|required|missing"):
        MNRFConfig.from_kwargs(dimensions={"S": 2, "E": 3, "P": 5, "R": 7})


def test_shared_artifact_shape_contract_accepts_and_rejects_actual_shapes():
    layout = FeatureLayout(3, 5, 7)
    valid = dict(
        layout=layout, variable_count=2,
        cache_shapes={"train": (10, 2, 16), "validation": (2, 2, 16), "calibration": (2, 2, 16)},
        index_shapes={"train": (10,), "validation": (2,), "calibration": (2,)},
        expected_counts={"train": 10, "validation": 2, "calibration": 2},
        input_mean_shape=(2,), input_scale_shape=(2,), feature_mean_shape=(2, 16),
        feature_scale_shape=(2, 16), calibration_center_shape=(2,),
        calibration_scale_shape=(2,), noise_shape=(4, 16), context="fixture",
    )
    validate_artifact_shapes(**valid)
    for key, bad, expected in (
        ("cache_shapes", {"train": (10, 2, 25), "validation": (2, 2, 16), "calibration": (2, 2, 16)}, "cache train shape mismatch"),
        ("feature_mean_shape", (2, 25), "feature_mean shape mismatch"),
        ("feature_scale_shape", (2, 15), "feature_scale shape mismatch"),
        ("noise_shape", (4, 25), "fixed scoring noise shape mismatch"),
        ("calibration_scale_shape", (1,), "calibration_scale shape mismatch"),
        ("index_shapes", {"train": (9,), "validation": (2,), "calibration": (2,)}, "window indices train shape mismatch"),
    ):
        invalid = dict(valid)
        invalid[key] = bad
        with pytest.raises(ValueError, match=expected):
            validate_artifact_shapes(**invalid)
