import gymnasium as gym

class RandomInitialMovementWrapper(gym.Wrapper):
    """
    Wrapper to randomize initial environment states via random actions.

    Purpose:
    --------
    Executes a random number of random actions after each reset to
    decorrelate initial states and promote exploration diversity.

    Note:
    -----
    This does **NOT** guarantee reaching advanced levels (e.g., level 2 in MsPacman),
    but provides variation within the same level.

    Parameters:
    -----------
    min_steps : int
        Minimum number of random steps after reset.
    max_steps : int
        Maximum number of random steps after reset.
    """
    def __init__(self, env, min_steps=30, max_steps=200, safe_retry=True):
        super().__init__(env)
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.safe_retry = safe_retry  # If True, retries silently on death

    def reset(self, **kwargs):
        while True:
            obs, info = self.env.reset(**kwargs)
            random_steps = self.env.unwrapped.np_random.integers(
                self.min_steps, self.max_steps + 1
            )
            died = False

            for _ in range(random_steps):
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)

                if terminated or truncated:
                    died = True
                    break  # Pacman died during random steps

            if not self.safe_retry or not died:
                # Return state only if Pacman survived (or if retries are disabled)
                return obs, info
            # Otherwise retry silently