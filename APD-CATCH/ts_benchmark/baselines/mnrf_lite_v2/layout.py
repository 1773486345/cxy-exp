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
        layout = cls(
            evolution_dim=int(value["dimensions"]["E"]),
            pattern_dim=int(value["dimensions"]["P"]),
            relation_dim=int(value["dimensions"]["R"]),
        )
        if value.get("order") != ["S", "E", "P", "R"] or value.get("slices") != {
                key: list(bounds) for key, bounds in layout.slices.items()} or value.get("F") != layout.total_dim:
            raise ValueError("saved S/E/P/R feature layout is inconsistent")
        return layout
