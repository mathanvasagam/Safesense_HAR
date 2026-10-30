from safesense.config import ExperimentConfig


def test_config_matches_plan_defaults() -> None:
    config = ExperimentConfig()
    assert config.sequence_length == 32
    assert config.depth_features == 92
    assert config.skeleton_features == 60
    assert config.imu_features == 6
    assert config.train_subjects == (1, 3, 5, 7)
    assert config.test_subjects == (2, 4, 6, 8)
    assert config.lag_candidates == (-4, -3, -2, -1, 0, 1, 2, 3, 4)


def test_config_round_trips_through_mapping() -> None:
    original = ExperimentConfig(seed=22)
    assert ExperimentConfig.from_mapping(original.to_dict()) == original
