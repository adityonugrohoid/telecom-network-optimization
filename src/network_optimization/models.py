"""
ML model training and evaluation for Telecom Network Optimization.
"""

import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from .config import MODEL_CONFIG


class BaseModel:
    """Base class for ML models."""

    def __init__(self, config: dict = None):
        self.config = config or MODEL_CONFIG
        self.model = None
        self.feature_names = None
        self.is_trained = False

    def prepare_data(self, df, target_col, test_size=0.2, random_state=42):
        y = df[target_col]
        X = df.drop(columns=[target_col])
        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Train set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        raise NotImplementedError("Subclasses must implement train()")

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, task_type="classification"):
        y_pred = self.predict(X_test)
        metrics = {}
        if task_type == "classification":
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["precision"] = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            metrics["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            metrics["f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            if len(np.unique(y_test)) == 2:
                y_proba = self.model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        elif task_type == "regression":
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_test, y_pred)
            metrics["r2"] = 1 - (
                np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
            )
        return metrics

    def get_feature_importance(self):
        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError("Model does not support feature importance")
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    def save(self, filepath):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def cross_validate_model(model, X, y, cv_folds=5, scoring="accuracy"):
    """Perform cross-validation on a trained model."""
    scores = cross_val_score(model.model, X, y, cv=cv_folds, scoring=scoring)
    return {
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "all_scores": scores,
    }


def print_metrics(metrics, title="Model Performance"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:8.4f}")
    print(f"{'=' * 50}\n")


# ---------------------------------------------------------------------------
# Reinforcement-Learning classes for Network Optimization
# ---------------------------------------------------------------------------


class NetworkEnvironment:
    """Simulated network environment for RL-based optimization.

    Wraps a DataFrame of pre-generated state transitions so that an RL agent
    can interact with the environment through the standard reset/step API.
    """

    def __init__(self, states_df: pd.DataFrame):
        """
        Parameters
        ----------
        states_df : pd.DataFrame
            Must contain columns for state features, an ``action`` column, a
            ``reward`` column, and a ``done`` column.  Rows are assumed to
            represent sequential transitions.
        """
        self.states_df = states_df.reset_index(drop=True)

        # Identify state feature columns (everything except action/reward/done)
        reserved = {"action", "reward", "done"}
        self.state_columns = [c for c in states_df.columns if c not in reserved]

        self._current_idx = 0

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial state."""
        self._current_idx = 0
        return self.states_df.loc[self._current_idx, self.state_columns].values.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take an action and return (next_state, reward, done, info).

        The pre-recorded reward from the dataset is adjusted based on whether
        the agent's chosen action matches the recorded optimal action.
        """
        row = self.states_df.iloc[self._current_idx]
        recorded_action = int(row["action"])
        base_reward = float(row["reward"])
        done = bool(row["done"])

        # Reward shaping: full reward if agent matches optimal, penalty otherwise
        if action == recorded_action:
            reward = base_reward
        else:
            reward = base_reward * 0.5 - 0.1

        self._current_idx += 1
        if self._current_idx >= len(self.states_df):
            done = True

        if done:
            next_state = np.zeros(len(self.state_columns), dtype=np.float32)
        else:
            next_state = self.states_df.loc[self._current_idx, self.state_columns].values.astype(
                np.float32
            )

        info = {"recorded_action": recorded_action, "step": self._current_idx}
        return next_state, reward, done, info

    def get_state_size(self) -> int:
        """Return the dimensionality of the state vector."""
        return len(self.state_columns)

    def get_action_size(self) -> int:
        """Return the number of distinct actions in the dataset."""
        return int(self.states_df["action"].nunique())


class QLearningAgent:
    """Tabular Q-Learning agent for discrete network optimization."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        n_bins: int = 10,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_bins = n_bins

        # Q-table: keys are (discretized_state_tuple, action)
        self.q_table: Dict[Tuple, float] = {}

        # Bin edges per state dimension (set during training)
        self._bin_edges = None

    def _init_bins(self, sample_states: np.ndarray):
        """Compute bin edges from a sample of continuous states."""
        self._bin_edges = []
        for dim in range(self.state_size):
            edges = np.linspace(
                sample_states[:, dim].min(),
                sample_states[:, dim].max(),
                self.n_bins + 1,
            )
            self._bin_edges.append(edges)

    def discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """Bin a continuous state vector into discrete bucket indices."""
        if self._bin_edges is None:
            raise RuntimeError("Bin edges not initialized. Call train() or _init_bins() first.")
        discrete = []
        for dim in range(self.state_size):
            idx = int(np.digitize(state[dim], self._bin_edges[dim]) - 1)
            idx = max(0, min(idx, self.n_bins - 1))
            discrete.append(idx)
        return tuple(discrete)

    def _get_q(self, state_key: Tuple, action: int) -> float:
        return self.q_table.get((state_key, action), 0.0)

    def choose_action(self, state: np.ndarray) -> int:
        """Select an action using an epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state_key = self.discretize_state(state)
        q_values = [self._get_q(state_key, a) for a in range(self.action_size)]
        return int(np.argmax(q_values))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
    ):
        """Update the Q-table using the Bellman equation."""
        state_key = self.discretize_state(state)
        next_key = self.discretize_state(next_state)

        best_next_q = max([self._get_q(next_key, a) for a in range(self.action_size)])
        current_q = self._get_q(state_key, action)
        new_q = current_q + self.lr * (reward + self.gamma * best_next_q - current_q)
        self.q_table[(state_key, action)] = new_q

    def train(
        self,
        env: NetworkEnvironment,
        n_episodes: int = 1000,
    ) -> list:
        """Train the agent over multiple episodes.

        Returns
        -------
        episode_rewards : list[float]
            Total reward collected in each episode.
        """
        # Initialise bins from the full state space in the environment
        all_states = env.states_df[env.state_columns].values.astype(np.float32)
        self._init_bins(all_states)

        episode_rewards = []

        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)

            if (ep + 1) % max(1, n_episodes // 10) == 0:
                avg = np.mean(episode_rewards[-100:])
                print(
                    f"Episode {ep + 1:>5}/{n_episodes}  |  "
                    f"Avg reward (last 100): {avg:8.2f}  |  "
                    f"Epsilon: {self.epsilon:.4f}"
                )

        print("Q-Learning training complete.")
        return episode_rewards

    def get_policy(self) -> Dict[Tuple, int]:
        """Return the learned greedy policy as a mapping from state to action.

        Returns
        -------
        policy : dict
            ``{discretized_state: best_action}`` for every state visited
            during training.
        """
        # Collect all unique state keys
        state_keys = set()
        for s_key, _a in self.q_table:
            state_keys.add(s_key)

        policy = {}
        for s_key in state_keys:
            q_values = [self._get_q(s_key, a) for a in range(self.action_size)]
            policy[s_key] = int(np.argmax(q_values))

        return policy


def main():
    """Main entry point for network optimization model training."""
    # TODO: Load processed state-transition data from PROCESSED_DATA_DIR
    # TODO: Create NetworkEnvironment and QLearningAgent
    # TODO: Train agent and visualise reward curve
    print("Network Optimization RL training — not yet implemented.")


if __name__ == "__main__":
    main()
