import gymnasium as gym
import numpy as np

class GhostEscapeRewardWrapper(gym.RewardWrapper):
    """
    Da recompensa si el agente se aleja de fantasmas que estaban peligrosamente cerca,
    solo si NO está energizado (es decir, no puede comerse a los fantasmas).
    """
    def __init__(self, env, danger_radius=3, escape_reward=0.2):
        super().__init__(env)
        self.danger_radius = danger_radius
        self.escape_reward = escape_reward
        self.prev_agent_pos = None
        self.prev_ghost_positions = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_agent_pos = self._get_agent_pos(info)
        self.prev_ghost_positions = self._get_ghosts_pos(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        agent_pos = self._get_agent_pos(info)
        ghost_positions = self._get_ghosts_pos(info)

        bonus = 0.0
        if not self._is_energized(info):
            for ghost in ghost_positions:
                if self._is_close(self.prev_agent_pos, ghost):
                    if self._distance(agent_pos, ghost) > self._distance(self.prev_agent_pos, ghost):
                        bonus += self.escape_reward  # Se alejó

        # Actualizar estado previo
        self.prev_agent_pos = agent_pos
        self.prev_ghost_positions = ghost_positions

        return obs, reward + bonus, terminated, truncated, info

    def _get_agent_pos(self, info):
        # RAM offset para MsPacman: 75 (posición X), 77 (posición Y)
        ram = info.get("ale-ram")
        if ram is None:
            return (0, 0)
        return (ram[75], ram[77])

    def _get_ghosts_pos(self, info):
        # MsPacman: ghosts en RAM 13, 14, 15, 16 (X), y 27, 28, 29, 30 (Y)
        ram = info.get("ale-ram")
        if ram is None:
            return []
        x = ram[13:17]
        y = ram[27:31]
        return list(zip(x, y))

    def _is_energized(self, info):
        # RAM offset 123 indica cuántos frames de poder quedan
        ram = info.get("ale-ram")
        if ram is None:
            return False
        return ram[123] > 0

    def _distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def _is_close(self, a, b):
        return self._distance(a, b) <= self.danger_radius