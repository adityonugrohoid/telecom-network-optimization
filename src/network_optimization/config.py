"""
Configuration management for Telecom Network Optimization.

This module centralizes all configuration parameters, making it easy to
adjust settings without modifying core logic.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


# ============================================================================
# DATA GENERATION CONFIG
# ============================================================================

DATA_GEN_CONFIG = {
    "random_seed": 42,
    "n_samples": 20_000,
    "test_size": 0.2,
    "validation_size": 0.1,
    "use_case_params": {
        "n_cells": 50,
        "n_episodes": 400,
        "steps_per_episode": 50,
        "n_actions": 5,
        "actions": [
            "increase_power",
            "decrease_power",
            "adjust_tilt",
            "load_balance",
            "no_action",
        ],
        "state_features": ["load", "sinr", "interference", "throughput", "latency"],
    },
}


# ============================================================================
# FEATURE ENGINEERING CONFIG
# ============================================================================

FEATURE_CONFIG = {
    "categorical_features": [
        "action",
        "cell_type",
    ],
    "numerical_features": [
        "load",
        "sinr",
        "interference",
        "throughput",
        "latency",
        "connected_users",
        "prb_utilization",
    ],
    "datetime_features": [],
    "rolling_windows": [],
    "create_features": True,
}


# ============================================================================
# MODEL TRAINING CONFIG
# ============================================================================

MODEL_CONFIG = {
    "algorithm": "q_learning",
    "cv_folds": None,
    "cv_strategy": None,
    "hyperparameters": {
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "n_episodes": 1000,
        "max_steps_per_episode": 50,
    },
    "early_stopping_rounds": None,
    "verbose": True,
}


# ============================================================================
# EVALUATION CONFIG
# ============================================================================

EVAL_CONFIG = {
    "primary_metric": "cumulative_reward",
    "threshold": None,
    "compute_metrics": [
        "cumulative_reward",
        "avg_reward_per_step",
        "sinr_improvement",
        "throughput_improvement",
        "convergence_episode",
    ],
}


# ============================================================================
# VISUALIZATION CONFIG
# ============================================================================

VIZ_CONFIG = {
    "style": "whitegrid",
    "palette": "husl",
    "context": "notebook",
    "figure_size": (12, 6),
    "dpi": 100,
}


# ============================================================================
# UTILITIES
# ============================================================================


def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_custom_config(config_path: Path) -> Dict[str, Any]:
    """Load custom configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "data_gen": DATA_GEN_CONFIG,
        "features": FEATURE_CONFIG,
        "model": MODEL_CONFIG,
        "eval": EVAL_CONFIG,
        "viz": VIZ_CONFIG,
        "paths": {
            "root": PROJECT_ROOT,
            "data": DATA_DIR,
            "raw": RAW_DATA_DIR,
            "processed": PROCESSED_DATA_DIR,
            "notebooks": NOTEBOOKS_DIR,
        },
    }


if __name__ == "__main__":
    ensure_directories()
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Data directory: {DATA_DIR}")
    print(f"Random seed: {DATA_GEN_CONFIG['random_seed']}")
    print(f"Algorithm: {MODEL_CONFIG['algorithm']}")
