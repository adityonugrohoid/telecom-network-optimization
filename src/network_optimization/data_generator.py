"""
Domain-informed synthetic data generator for Telecom Network Optimization.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_GEN_CONFIG, RAW_DATA_DIR, ensure_directories


class TelecomDataGenerator:
    """Base class for generating synthetic telecom data."""

    def __init__(self, seed: int = 42, n_samples: int = 10_000):
        self.seed = seed
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement generate()")

    def generate_sinr(
        self, n: int, base_sinr_db: float = 10.0, noise_std: float = 5.0
    ) -> np.ndarray:
        sinr = self.rng.normal(base_sinr_db, noise_std, n)
        return np.clip(sinr, -5, 25)

    def sinr_to_throughput(
        self, sinr_db: np.ndarray, network_type: np.ndarray, noise_factor: float = 0.2
    ) -> np.ndarray:
        sinr_linear = 10 ** (sinr_db / 10)
        capacity_factor = np.log2(1 + sinr_linear)
        max_throughput = np.where(network_type == "5G", 300, 50)
        throughput = capacity_factor * max_throughput / 5
        noise = self.rng.normal(1, noise_factor, len(throughput))
        throughput = throughput * noise
        return np.clip(throughput, 0.1, max_throughput)

    def generate_congestion_pattern(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        hour = timestamps.hour
        day_of_week = timestamps.dayofweek
        congestion = 0.5 + 0.3 * np.sin((hour - 6) * np.pi / 12)
        peak_morning = (hour >= 9) & (hour <= 11)
        peak_evening = (hour >= 18) & (hour <= 21)
        congestion = np.where(peak_morning | peak_evening, congestion * 1.3, congestion)
        is_weekend = day_of_week >= 5
        congestion = np.where(is_weekend, congestion * 0.8, congestion)
        noise = self.rng.normal(0, 0.1, len(congestion))
        congestion = congestion + noise
        return np.clip(congestion, 0, 1)

    def congestion_to_latency(
        self, congestion: np.ndarray, base_latency_ms: float = 20
    ) -> np.ndarray:
        latency = base_latency_ms * (1 + 5 * congestion**2)
        jitter = self.rng.normal(0, 5, len(latency))
        latency = latency + jitter
        return np.clip(latency, 10, 300)

    def compute_qoe_mos(
        self,
        throughput_mbps: np.ndarray,
        latency_ms: np.ndarray,
        packet_loss_pct: np.ndarray,
        app_type: np.ndarray,
    ) -> np.ndarray:
        mos_throughput = 1 + 4 * (1 - np.exp(-throughput_mbps / 10))
        latency_penalty = np.clip(latency_ms / 100, 0, 2)
        loss_penalty = packet_loss_pct / 2
        mos = mos_throughput - latency_penalty - loss_penalty
        video_mask = app_type == "video_streaming"
        mos = np.where(video_mask, mos - packet_loss_pct * 0.5, mos)
        gaming_mask = app_type == "gaming"
        mos = np.where(gaming_mask, mos - latency_penalty * 0.5, mos)
        return np.clip(mos, 1, 5)

    def save(self, df: pd.DataFrame, filename: str) -> Path:
        ensure_directories()
        output_path = RAW_DATA_DIR / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df):,} rows to {output_path}")
        return output_path


class NetworkOptDataGenerator(TelecomDataGenerator):
    """Generate synthetic state-action-reward tuples for RL-based network optimization."""

    ACTIONS = [
        "increase_power",
        "decrease_power",
        "adjust_tilt",
        "load_balance",
        "no_action",
    ]

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = 20_000,
        n_episodes: int = 400,
        steps_per_episode: int = 50,
        n_cells: int = 50,
    ):
        super().__init__(seed=seed, n_samples=n_samples)
        self.n_episodes = n_episodes
        self.steps_per_episode = steps_per_episode
        self.n_cells = n_cells

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    def _random_state(self, n: int = 1) -> dict:
        """Sample a random initial state vector for *n* rows."""
        return {
            "load": self.rng.uniform(0.1, 1.0, n),
            "sinr": self.rng.uniform(-5, 25, n),
            "interference": self.rng.uniform(0, 1, n),
            "throughput": self.rng.uniform(1, 200, n),
            "latency": self.rng.uniform(10, 200, n),
            "connected_users": self.rng.integers(10, 501, n),
            "prb_utilization": self.rng.uniform(0.1, 0.95, n),
        }

    # ------------------------------------------------------------------
    # Action effects (probabilistic transitions)
    # ------------------------------------------------------------------

    def _apply_action(self, state: dict, action: str) -> dict:
        """Return a next-state dict after applying *action* to *state*.

        Each action has domain-motivated probabilistic effects on the KPIs.
        """
        n = len(state["load"])
        next_state = {k: v.copy() for k, v in state.items()}

        if action == "increase_power":
            sinr_gain = self.rng.uniform(1, 3, n)
            next_state["sinr"] = state["sinr"] + sinr_gain
            intf_gain = self.rng.uniform(0.05, 0.1, n)
            next_state["interference"] = state["interference"] + intf_gain
            # Higher SINR -> marginal throughput improvement
            next_state["throughput"] = state["throughput"] * (1 + sinr_gain / 25)
            next_state["latency"] = state["latency"] * (1 - sinr_gain / 100)

        elif action == "decrease_power":
            sinr_loss = self.rng.uniform(1, 2, n)
            next_state["sinr"] = state["sinr"] - sinr_loss
            intf_loss = self.rng.uniform(0.05, 0.1, n)
            next_state["interference"] = state["interference"] - intf_loss
            next_state["throughput"] = state["throughput"] * (1 - sinr_loss / 50)
            next_state["latency"] = state["latency"] * (1 + sinr_loss / 100)

        elif action == "adjust_tilt":
            sinr_delta = self.rng.uniform(-1, 1, n)
            next_state["sinr"] = state["sinr"] + sinr_delta
            tp_factor = 1 + self.rng.uniform(0.05, 0.10, n)
            next_state["throughput"] = state["throughput"] * tp_factor
            next_state["latency"] = state["latency"] * (1 - 0.02 * tp_factor)

        elif action == "load_balance":
            # Redistribute load: some cells decrease, others increase
            load_shift = self.rng.uniform(-0.15, 0.10, n)
            next_state["load"] = state["load"] + load_shift
            latency_reduction = self.rng.uniform(0.05, 0.15, n)
            next_state["latency"] = state["latency"] * (1 - latency_reduction)
            next_state["prb_utilization"] = state["prb_utilization"] + load_shift * 0.5
            next_state["connected_users"] = state["connected_users"] + self.rng.integers(-20, 21, n)

        else:  # no_action
            drift_sinr = self.rng.normal(0, 0.3, n)
            drift_tp = self.rng.normal(0, 1.0, n)
            drift_lat = self.rng.normal(0, 1.0, n)
            drift_load = self.rng.normal(0, 0.02, n)
            next_state["sinr"] = state["sinr"] + drift_sinr
            next_state["throughput"] = state["throughput"] + drift_tp
            next_state["latency"] = state["latency"] + drift_lat
            next_state["load"] = state["load"] + drift_load

        # Clip all features to valid ranges
        next_state["load"] = np.clip(next_state["load"], 0.1, 1.0)
        next_state["sinr"] = np.clip(next_state["sinr"], -5, 25)
        next_state["interference"] = np.clip(next_state["interference"], 0, 1)
        next_state["throughput"] = np.clip(next_state["throughput"], 1, 200)
        next_state["latency"] = np.clip(next_state["latency"], 10, 200)
        next_state["connected_users"] = np.clip(next_state["connected_users"], 10, 500).astype(int)
        next_state["prb_utilization"] = np.clip(next_state["prb_utilization"], 0.1, 0.95)

        return next_state

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_reward(state: dict, next_state: dict) -> np.ndarray:
        """Weighted KPI improvement reward.

        reward = 0.3 * (sinr_delta / 25)
               + 0.3 * (throughput_delta / 200)
               - 0.2 * (latency_delta / 200)
               - 0.2 * interference_delta
        """
        sinr_delta = next_state["sinr"] - state["sinr"]
        throughput_delta = next_state["throughput"] - state["throughput"]
        latency_delta = next_state["latency"] - state["latency"]
        interference_delta = next_state["interference"] - state["interference"]

        reward = (
            0.3 * (sinr_delta / 25.0)
            + 0.3 * (throughput_delta / 200.0)
            - 0.2 * (latency_delta / 200.0)
            - 0.2 * interference_delta
        )
        return reward

    # ------------------------------------------------------------------
    # Main generation
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate the full state-action-reward dataset for RL training."""
        cell_ids = [f"CELL_{i:04d}" for i in range(self.n_cells)]
        records = []

        for ep in range(self.n_episodes):
            # Assign a random cell for this episode
            cell_id = self.rng.choice(cell_ids)

            # Initial state for the episode
            current_state = self._random_state(n=1)
            # Squeeze to scalar-like arrays of length 1 for step-by-step processing
            current_state = {k: v for k, v in current_state.items()}

            for step in range(self.steps_per_episode):
                # Choose a random action (uniform exploration policy)
                action = self.rng.choice(self.ACTIONS)

                # Compute next state
                next_state = self._apply_action(current_state, action)

                # Compute reward
                reward = self._compute_reward(current_state, next_state)

                done = step == self.steps_per_episode - 1

                row = {
                    "episode_id": ep,
                    "step": step,
                    "cell_id": cell_id,
                    # Current state features
                    "load": float(current_state["load"][0]),
                    "sinr": round(float(current_state["sinr"][0]), 2),
                    "interference": round(float(current_state["interference"][0]), 4),
                    "throughput": round(float(current_state["throughput"][0]), 2),
                    "latency": round(float(current_state["latency"][0]), 2),
                    "connected_users": int(current_state["connected_users"][0]),
                    "prb_utilization": round(float(current_state["prb_utilization"][0]), 4),
                    # Action
                    "action": action,
                    # Next state features
                    "next_load": float(next_state["load"][0]),
                    "next_sinr": round(float(next_state["sinr"][0]), 2),
                    "next_interference": round(float(next_state["interference"][0]), 4),
                    "next_throughput": round(float(next_state["throughput"][0]), 2),
                    "next_latency": round(float(next_state["latency"][0]), 2),
                    "next_connected_users": int(next_state["connected_users"][0]),
                    "next_prb_utilization": round(float(next_state["prb_utilization"][0]), 4),
                    # Reward & terminal flag
                    "reward": round(float(reward[0]), 6),
                    "done": done,
                }
                records.append(row)

                # Advance state
                current_state = next_state

        df = pd.DataFrame(records)
        return df


def main() -> None:
    """Generate and save the network-optimization RL dataset using project config."""
    config = DATA_GEN_CONFIG
    params = config["use_case_params"]

    generator = NetworkOptDataGenerator(
        seed=config["random_seed"],
        n_samples=config["n_samples"],
        n_episodes=params["n_episodes"],
        steps_per_episode=params["steps_per_episode"],
        n_cells=params["n_cells"],
    )

    df = generator.generate()

    print(f"Generated dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Episodes: {df['episode_id'].nunique()}")
    print(f"Unique cells: {df['cell_id'].nunique()}")
    print(f"Actions distribution:\n{df['action'].value_counts()}")
    print("\nReward statistics:")
    print(df["reward"].describe())

    generator.save(df, "network_optimization")


if __name__ == "__main__":
    main()
