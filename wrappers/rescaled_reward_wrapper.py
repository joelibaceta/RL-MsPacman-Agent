import gymnasium as gym
import numpy as np

class AutoRescaledRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that smoothly rescales rewards to a continuous range in [-1, 1]
    using a tanh transformation. Unlike the typical Atari clipping (-1, 0, +1),
    this approach preserves the reward signal richness while preventing the agent
    from over-focusing on large absolute rewards.

    During the first few episodes, the wrapper collects observed reward magnitudes,
    and estimates an appropriate scale based on the 95th percentile to avoid outliers.
    This value is then used as the denominator in the tanh scaling.

    Parameters:
    - env: The Gymnasium environment.
    - warmup_episodes (int): Number of episodes to observe before calibrating the scale.
    - default_scale (float): Fallback scale to use before calibration is complete, or in case of empty buffer.

    Example:
        env = AutoRescaledRewardWrapper(env, warmup_episodes=10)
    """

    def __init__(self, env, warmup_episodes=10, default_scale=100.0):
        super().__init__(env)
        self.scale = default_scale
        self.warmup_episodes = warmup_episodes
        self.reward_buffer = []
        self.episode_count = 0
        self.calibrated = False

    def reward(self, reward):
        if not self.calibrated:
            self.reward_buffer.append(abs(reward))
        return np.tanh(reward / self.scale)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not self.calibrated:
            self.episode_count += 1
            if self.episode_count >= self.warmup_episodes and len(self.reward_buffer) > 0:
                self.scale = np.percentile(self.reward_buffer, 95)
                self.scale = max(self.scale, 1.0)  # Prevent division by zero or too small
                self.calibrated = True
                print(f"[AutoRescaledRewardWrapper] Adjusted scale to: {self.scale:.2f}")
        return obs, info