import gymnasium as gym

class SurvivalBonusWrapper(gym.RewardWrapper):
    """
    Adds a small positive reward for each time step survived.
    Encourages exploration and discourages standing still or dying early.
    
    Parameters:
    - env: the Gymnasium environment
    - bonus_per_step (float): reward to add per step
    """
    
    def __init__(self, env, bonus_per_step=0.1):
        super().__init__(env)
        self.bonus = bonus_per_step

    def reward(self, reward):
        return reward + self.bonus