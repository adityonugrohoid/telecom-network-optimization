# Telecom Network Optimization

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)
![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## Business Context

Network engineers manually tune parameters (power, tilt, load balancing) across thousands of cells. RL-based optimization can automatically recommend optimal actions, improving KPIs while reducing manual effort.

## Problem Framing

Reinforcement Learning using Q-Learning.

- **Target:** KPI improvement
- **Primary Metric:** Cumulative Reward
- **Challenges:**
  - Continuous state space requiring discretization
  - Delayed reward signals
  - Action-effect uncertainty across cell configurations
  - Balancing exploration vs exploitation in a safety-critical environment

## Data Engineering

State-action-reward tuples (400 episodes x 50 steps = 20K rows):

- **7 state features** -- cell-level KPIs including load, throughput, SINR, latency, packet loss, handover rate, and congestion index
- **5 actions** -- domain-specific interventions: power adjustment, tilt change, load balancing, carrier reconfig, and capacity boost
- **Probabilistic outcomes** -- each action has domain-informed transition probabilities affecting KPI deltas

Domain physics: reward = weighted KPI delta, reflecting how real network optimization balances multiple competing objectives (throughput vs interference, capacity vs coverage).

## Methodology

- Tabular Q-Learning with epsilon-greedy exploration
- **Key components:**
  - State discretization (binning continuous KPIs into discrete states)
  - Epsilon decay schedule (exploration to exploitation over episodes)
  - Q-table update with configurable learning rate and discount factor
  - Episode-level cumulative reward tracking
- Comparison against random baseline policy
- Convergence analysis via reward moving average

## Key Findings

- **Convergence:** Agent trains over 500 episodes with epsilon decay from 1.0 to 0.08
- **Most effective action:** `load_balance` -- achieves the best mean reward per step among all 5 actions
- **Performance:** 61% improvement in cumulative reward over random baseline policy
- Learned Q-table contains 145 state-action entries, demonstrating efficient state-space coverage

## Quick Start

```bash
# Clone the repository
git clone https://github.com/adityonugrohoid/telecom-ml-portfolio.git
cd telecom-ml-portfolio/06-network-optimization

# Install dependencies
uv sync

# Generate synthetic data
uv run python -m network_optimization.data_generator

# Run the notebook
uv run jupyter lab notebooks/
```

## Project Structure

```
06-network-optimization/
├── README.md
├── pyproject.toml
├── notebooks/
│   └── 06_network_optimization.ipynb
├── src/
│   └── network_optimization/
│       ├── __init__.py
│       ├── config.py
│       ├── data_generator.py
│       ├── features.py
│       └── models.py
├── data/
│   └── .gitkeep
├── tests/
│   └── test_data_quality.py
└── docs/
```

## Related Projects

| # | Project | Description |
|---|---------|-------------|
| 1 | [Churn Prediction](../01-churn-prediction) | Binary classification to predict customer churn |
| 2 | [Root Cause Analysis](../02-root-cause-analysis) | Multi-class classification for network alarm RCA |
| 3 | [Anomaly Detection](../03-anomaly-detection) | Unsupervised detection of network anomalies |
| 4 | [QoE Prediction](../04-qoe-prediction) | Regression to predict quality of experience |
| 5 | [Capacity Forecasting](../05-capacity-forecasting) | Time-series forecasting for network capacity planning |
| 6 | **Network Optimization** (this repo) | Optimization of network resource allocation |

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

## Author

**Adityo Nugroho**
