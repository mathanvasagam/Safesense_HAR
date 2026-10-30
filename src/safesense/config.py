"""Small, serializable experiment configuration."""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentConfig:
    """Defaults shared by preprocessing, training, and evaluation commands."""

    seed: int = 11
    num_classes: int = 27
    sequence_length: int = 32
    depth_features: int = 92
    skeleton_features: int = 60
    imu_features: int = 6
    train_subjects: tuple[int, ...] = (1, 3, 5, 7)
    test_subjects: tuple[int, ...] = (2, 4, 6, 8)
    development_train_subjects: tuple[int, ...] = (1, 3, 5)
    development_valid_subjects: tuple[int, ...] = (7,)
    batch_size: int = 4
    max_epochs: int = 300
    learning_rate: float = 9.8e-4
    weight_decay: float = 1.6e-4
    model_width: int = 96
    dropout: float = 0.1
    lag_candidates: tuple[int, ...] = field(default_factory=lambda: (-4, -3, -2, -1, 0, 1, 2, 3, 4))

    def __post_init__(self) -> None:
        if self.num_classes <= 0 or self.sequence_length <= 0:
            raise ValueError("num_classes and sequence_length must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not self.lag_candidates or 0 not in self.lag_candidates:
            raise ValueError("lag_candidates must include zero")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible configuration values."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        """Build a config while retaining defaults for omitted values."""
        data = dict(values)
        for name in (
            "train_subjects",
            "test_subjects",
            "development_train_subjects",
            "development_valid_subjects",
            "lag_candidates",
        ):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)
