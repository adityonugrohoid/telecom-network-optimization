"""Tests for data quality and validation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from network_optimization.data_generator import NetworkOptDataGenerator


@pytest.fixture
def sample_data():
    generator = NetworkOptDataGenerator(
        seed=42,
        n_samples=20000,
        n_cells=5,
        n_episodes=10,
        steps_per_episode=20,
    )
    return generator.generate()


class TestDataQuality:

    def test_no_missing_values(self, sample_data):
        critical_cols = ["episode_id", "step", "action", "reward"]
        for col in critical_cols:
            if col in sample_data.columns:
                assert sample_data[col].isna().sum() == 0, f"Missing values in {col}"

    def test_data_types(self, sample_data):
        assert pd.api.types.is_numeric_dtype(sample_data["step"])
        assert pd.api.types.is_numeric_dtype(sample_data["reward"])

    def test_value_ranges(self, sample_data):
        assert sample_data["load"].min() >= 0
        assert sample_data["load"].max() <= 1
        assert sample_data["sinr"].min() >= -5
        assert sample_data["sinr"].max() <= 25
        assert sample_data["interference"].min() >= 0
        assert sample_data["interference"].max() <= 1
        assert sample_data["step"].min() >= 0

    def test_categorical_values(self, sample_data):
        expected_actions = {"increase_power", "decrease_power", "adjust_tilt", "load_balance", "no_action"}
        assert set(sample_data["action"].unique()).issubset(expected_actions)

    def test_sample_size(self, sample_data):
        # 10 episodes * 20 steps = 200
        assert len(sample_data) == 200

    def test_episode_structure(self, sample_data):
        # Each episode should have sequential steps from 0 to max
        for episode_id in sample_data["episode_id"].unique():
            episode_data = sample_data[sample_data["episode_id"] == episode_id].sort_values("step")
            steps = episode_data["step"].values
            expected_steps = np.arange(len(steps))
            np.testing.assert_array_equal(steps, expected_steps)

    def test_done_flag(self, sample_data):
        # done==True only at last step of each episode
        for episode_id in sample_data["episode_id"].unique():
            episode_data = sample_data[sample_data["episode_id"] == episode_id].sort_values("step")
            # All steps except last should have done==False
            assert (~episode_data["done"].iloc[:-1]).all(), (
                f"Episode {episode_id}: non-terminal steps should have done=False"
            )
            # Last step should have done==True
            assert episode_data["done"].iloc[-1], (
                f"Episode {episode_id}: terminal step should have done=True"
            )


class TestDataGenerator:

    def test_generator_reproducibility(self):
        gen1 = NetworkOptDataGenerator(
            seed=42,
            n_samples=20000,
            n_cells=5,
            n_episodes=10,
            steps_per_episode=20,
        )
        gen2 = NetworkOptDataGenerator(
            seed=42,
            n_samples=20000,
            n_cells=5,
            n_episodes=10,
            steps_per_episode=20,
        )
        df1 = gen1.generate()
        df2 = gen2.generate()
        pd.testing.assert_frame_equal(df1, df2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
