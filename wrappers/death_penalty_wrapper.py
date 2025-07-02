import gymnasium as gym


class DeathPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-500):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            reward += self.penalty

        return obs, reward, terminated, truncated, info