"""Canonical S/E/P/R feature layout shared by the whole MNRF pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureLayout:
    evolution_dim: int = 8
    pattern_dim: int = 8
    relation_dim: int = 8

    def __post_init__(self):
        for name, value in (("evolution_dim", self.evolution_dim),
                            ("pattern_dim", self.pattern_dim),
                            ("relation_dim", self.relation_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def dimensions(self):
        return {"S": 1, "E": self.evolution_dim,
                "P": self.pattern_dim, "R": self.relation_dim}

    @property
    def slices(self):
        result = {}
        offset = 0
        for name, size in self.dimensions.items():
            result[name] = (offset, offset + size)
            offset += size
        return result

    @property
    def total_dim(self):
        return 1 + self.evolution_dim + self.pattern_dim + self.relation_dim

    @property
    def A_indices(self):
        slices = self.slices
        return tuple(range(*slices["S"])) + tuple(range(*slices["R"]))

    @property
    def B_indices(self):
        slices = self.slices
        return tuple(range(*slices["E"])) + tuple(range(*slices["P"]))

    def to_dict(self):
        return {
            "order": ["S", "E", "P", "R"],
            "dimensions": self.dimensions,
            "slices": {name: list(bounds) for name, bounds in self.slices.items()},
            "F": self.total_dim,
        }

    @classmethod
    def from_dict(cls, value):
        required = {"order", "dimensions", "slices", "F"}
        if not isinstance(value, dict):
            raise ValueError("feature layout schema must be an object")
        missing = required.difference(value)
        if missing:
            raise ValueError(f"feature layout is missing required fields: {sorted(missing)}")

        dimensions = value["dimensions"]
        if not isinstance(dimensions, dict) or set(dimensions) != {"S", "E", "P", "R"}:
            raise ValueError("feature layout dimensions must declare exactly S, E, P, and R")
        for name, dimension in dimensions.items():
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise ValueError(f"feature layout dimension {name} must be an integer")
        if dimensions["S"] != 1:
            raise ValueError("feature layout dimension S must be exactly 1")
        for name in ("E", "P", "R"):
            if dimensions[name] <= 0:
                raise ValueError(f"feature layout dimension {name} must be a positive integer")

        expected_order = ["S", "E", "P", "R"]
        order = value["order"]
        if (not isinstance(order, list) or len(order) != len(expected_order)
                or any(not isinstance(name, str) for name in order) or order != expected_order):
            raise ValueError(f"feature layout order must be exactly {expected_order}")
        total = value["F"]
        if isinstance(total, bool) or not isinstance(total, int):
            raise ValueError("feature layout F must be an integer")

        # Construct the canonical layout only after every dimension declaration
        # has passed strict type and range validation.
        layout = cls(dimensions["E"], dimensions["P"], dimensions["R"])
        if total != layout.total_dim:
            raise ValueError(f"feature layout F mismatch: expected {layout.total_dim}, got {total}")
        slices = value["slices"]
        if not isinstance(slices, dict) or set(slices) != {"S", "E", "P", "R"}:
            raise ValueError("feature layout slices must declare exactly S, E, P, and R")
        for name, bounds in slices.items():
            if (not isinstance(bounds, list) or len(bounds) != 2
                    or any(isinstance(endpoint, bool) or not isinstance(endpoint, int)
                           for endpoint in bounds)):
                raise ValueError(f"feature layout slice {name} must contain two integer endpoints")
        expected_slices = {name: list(bounds) for name, bounds in layout.slices.items()}
        if slices != expected_slices:
            raise ValueError(f"feature layout slices mismatch: expected {expected_slices}, got {slices}")
        return layout


def require_layout(value, expected, context):
    """Strictly decode a persisted layout and compare its full canonical form."""
    try:
        # Internal callers may already hold a validated canonical object;
        # persisted metadata must still pass the strict dictionary decoder.
        actual = value if isinstance(value, FeatureLayout) else FeatureLayout.from_dict(value)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError(f"{context}: invalid feature_layout: {error}") from error
    expected_layout = expected if isinstance(expected, FeatureLayout) else FeatureLayout.from_dict(expected)
    if actual.to_dict() != expected_layout.to_dict():
        raise ValueError(f"{context}: feature_layout mismatch; expected {expected_layout.to_dict()}, got {actual.to_dict()}")
    return actual
