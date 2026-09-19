"""Tests for the Q-Learning agent and the network environment it trains against."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from network_optimization.data_generator import NetworkOptDataGenerator
from network_optimization.models import NetworkEnvironment, QLearningAgent

STATE_FEATURES = [
    "load",
    "sinr",
    "interference",
    "throughput",
    "latency",
    "connected_users",
    "prb_utilization",
]
TRAIN_EPISODES = 100
N_EVAL = 30
N_BINS = 10

# The README reports a 61% improvement over the random baseline, measured with
# 1,000 training episodes on the full 20,000-row generated dataset. This test
# trains for far fewer episodes on a small seeded dataset to stay fast, and
# measures roughly 75-77% improvement across a few training seeds. The floor
# below sits with clear room under both figures, so a real regression in
# learning shows up without the test being sensitive to run-to-run noise.
IMPROVEMENT_FLOOR_PCT = 40.0


def build_env(seed=42, n_cells=5, n_episodes=10, steps_per_episode=20):
    """Generate a small seeded dataset and wrap it the way the notebook does:
    map the string action column to the integer ids NetworkEnvironment expects.
    """
    generator = NetworkOptDataGenerator(
        seed=seed,
        n_cells=n_cells,
        n_episodes=n_episodes,
        steps_per_episode=steps_per_episode,
    )
    raw = generator.generate()
    action_map = {a: i for i, a in enumerate(sorted(raw["action"].unique()))}
    env_cols = STATE_FEATURES + ["action", "reward", "done"]
    df_env = raw[env_cols].copy()
    df_env["action"] = df_env["action"].map(action_map)
    return NetworkEnvironment(df_env)


def evaluate_agent(agent, env, n_episodes):
    """Run episodes without further learning and collect total reward per episode."""
    rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        total = 0.0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            total += reward
            state = next_state
        rewards.append(total)
    return rewards


@pytest.fixture(scope="module")
def env():
    return build_env()


@pytest.fixture(scope="module")
def trained_agent(env):
    """The agent as the project trains it: seed the global RNG, then train()."""
    np.random.seed(123)
    agent = QLearningAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
        n_bins=N_BINS,
    )
    agent.train(env, n_episodes=TRAIN_EPISODES)
    return agent


@pytest.fixture(scope="module")
def repeated_training(env):
    """Two independently trained agents from the same seed, for reproducibility checks."""

    def train_one():
        np.random.seed(123)
        agent = QLearningAgent(
            state_size=env.get_state_size(),
            action_size=env.get_action_size(),
            n_bins=N_BINS,
        )
        agent.train(env, n_episodes=TRAIN_EPISODES)
        return agent

    return train_one(), train_one()


@pytest.fixture(scope="module")
def greedy_agent(env, trained_agent):
    """The trained agent's greedy policy, evaluated without exploration."""
    agent = QLearningAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
        epsilon=0.0,
        n_bins=N_BINS,
    )
    agent.q_table = trained_agent.q_table.copy()
    agent._bin_edges = trained_agent._bin_edges
    return agent


@pytest.fixture(scope="module")
def random_agent(env, trained_agent):
    """A uniform-random policy over the same discretized state space, as the notebook's baseline."""
    agent = QLearningAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
        epsilon=1.0,
        epsilon_min=1.0,
        epsilon_decay=1.0,
        n_bins=N_BINS,
    )
    agent._bin_edges = trained_agent._bin_edges
    return agent


class TestRefusals:
    def test_discretize_state_refuses_before_training(self):
        agent = QLearningAgent(state_size=7, action_size=5)
        with pytest.raises(RuntimeError):
            agent.discretize_state(np.zeros(7, dtype=np.float32))

    def test_update_refuses_before_training(self):
        agent = QLearningAgent(state_size=7, action_size=5)
        state = np.zeros(7, dtype=np.float32)
        with pytest.raises(RuntimeError):
            agent.update(state, 0, 1.0, state)


class TestTrainingReproducibility:
    def test_same_seed_gives_same_q_table(self, repeated_training):
        agent_a, agent_b = repeated_training
        assert set(agent_a.q_table.keys()) == set(agent_b.q_table.keys())
        for key in agent_a.q_table:
            assert agent_a.q_table[key] == pytest.approx(agent_b.q_table[key])

    def test_same_seed_gives_same_greedy_policy(self, repeated_training):
        agent_a, agent_b = repeated_training
        assert agent_a.get_policy() == agent_b.get_policy()


class TestQTable:
    def test_q_table_is_populated_with_finite_values_in_range(self, env, trained_agent):
        assert len(trained_agent.q_table) > 0
        for (state_key, action), value in trained_agent.q_table.items():
            assert len(state_key) == env.get_state_size()
            assert all(0 <= b < N_BINS for b in state_key)
            assert 0 <= action < env.get_action_size()
            assert np.isfinite(value)


class TestActionAndStateSpaces:
    def test_greedy_actions_and_discretized_states_stay_in_range(self, env, greedy_agent):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            discretized = greedy_agent.discretize_state(state)
            assert len(discretized) == env.get_state_size()
            assert all(0 <= b < N_BINS for b in discretized)

            action = greedy_agent.choose_action(state)
            assert 0 <= action < env.get_action_size()

            state, _, done, _ = env.step(action)
            steps += 1
        assert steps > 0

    def test_random_policy_actions_also_stay_in_range(self, env, random_agent):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            action = random_agent.choose_action(state)
            assert 0 <= action < env.get_action_size()
            state, _, done, _ = env.step(action)
            steps += 1
        assert steps > 0

    def test_out_of_range_state_still_clips_into_the_state_space(self, env, greedy_agent):
        size = env.get_state_size()
        above = greedy_agent.discretize_state(np.full(size, 1e6, dtype=np.float32))
        below = greedy_agent.discretize_state(np.full(size, -1e6, dtype=np.float32))
        assert all(0 <= b < N_BINS for b in above)
        assert all(0 <= b < N_BINS for b in below)


class TestEvaluationAgainstRandomBaseline:
    def test_trained_agent_beats_random_policy(self, env, greedy_agent, random_agent):
        np.random.seed(999)
        trained_rewards = evaluate_agent(greedy_agent, env, N_EVAL)
        np.random.seed(999)
        random_rewards = evaluate_agent(random_agent, env, N_EVAL)

        trained_mean = np.mean(trained_rewards)
        random_mean = np.mean(random_rewards)
        assert random_mean != 0
        improvement_pct = (trained_mean - random_mean) / abs(random_mean) * 100
        assert improvement_pct >= IMPROVEMENT_FLOOR_PCT, (
            f"Improvement over random fell to {improvement_pct:.1f}%"
        )
