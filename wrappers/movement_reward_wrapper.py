import gymnasium as gym

class MovementRewardWrapper(gym.RewardWrapper):
    """
    Premia al agente si se mueve respecto al paso anterior, penaliza si se queda quieto.
    Útil para evitar comportamiento pasivo o de espera en un solo lugar.
    """
    def __init__(self, env, move_reward=0.05, idle_penalty=0.05):
        super().__init__(env)
        self.move_reward = move_reward
        self.idle_penalty = idle_penalty
        self.prev_agent_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_agent_pos = self._get_agent_pos(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        agent_pos = self._get_agent_pos(info)

        if agent_pos == self.prev_agent_pos:
            reward -= self.idle_penalty
        else:
            reward += self.move_reward

        self.prev_agent_pos = agent_pos
        return obs, reward, terminated, truncated, info

    def _get_agent_pos(self, info):
        ram = info.get("ale-ram")
        if ram is None:
            return (0, 0)
        return (ram[75], ram[77])  # MsPacman: X = 75, Y = 77