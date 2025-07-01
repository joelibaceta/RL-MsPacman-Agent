import numpy as np

import gymnasium as gym

class CustomNoopResetWrapper(gym.Wrapper):
    """
    Custom No-op Reset Wrapper for Fine-Tuning Atari Agent Initialization

    Purpose:
    --------
    This wrapper reproduces the classic "No-op Reset" strategy used in Atari environments,
    but is implemented manually to enable future customization.

    Justification:
    --------------
    Many standardized Atari wrappers (e.g., from Stable-Baselines3 or Gymnasium) include
    a hard-coded No-op reset step that executes a random number of 'do nothing' actions
    (action=0) after an environment reset. This helps decorrelate initial states and
    prevents the agent from overfitting to a static starting position.

    However, by recreating this logic explicitly, we retain **full control** over:
    - The specific no-op action (in case we later define custom action sets).
    - The sampling strategy (e.g., from uniform to other distributions).
    - Maximum or minimum range for no-ops (e.g., `noop_max=30`).
    - Dynamic adjustment based on curriculum learning or policy stages.

    Scientific Motivation:
    ----------------------
    Empirical results in the Atari Learning Environment (ALE) and reinforcement learning
    benchmarks have shown that agents can exploit deterministic resets. Introducing
    random no-op steps increases environment stochasticity, promotes robustness, and
    improves generalization.

    Keeping this wrapper modular and accessible supports **future fine-tuning** for
    domain-specific exploration strategies, multi-agent variants, or adaptive curriculum
    training setups.

    Example:
    --------
    To explore how sensitivity to initial states impacts early learning, one might vary
    `noop_max` dynamically based on episode number or performance plateauing.
    """
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0  # Acción de no-op típica

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info