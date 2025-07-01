import gymnasium as gym
import numpy as np
import cv2
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

class CropPlayfieldWrapper(ObservationWrapper):
    """
    Observation wrapper that crops and resizes Atari frames for more stable learning.

    Specifically designed for MsPacman (original frame size: 210x160x3), this wrapper:
    - Crops out the top 34 pixels, which usually contain the game score and static UI elements.
    - Crops out the bottom 16 pixels, which may contain flickering or non-informative graphics.
    - Extracts the 160x160 central playfield, where the actual gameplay occurs.
    - Resizes the cropped region to a square 84x84 RGB image, preserving the full color space.

    Motivation:
    ----------
    The goal of this preprocessing is to remove sources of visual noise and avoid distortion:
    - The top region (score) does not affect the game's dynamics and may introduce unnecessary variance.
    - The bottom region often fluctuates due to animations or game artifacts, adding irrelevant variability.
    - Directly resizing the full 210x160 frame to 84x84 distorts the aspect ratio, compressing the vertical dimension disproportionately.
      Cropping before resizing maintains a more uniform and visually coherent input.
    
    This ensures the CNN receives consistent and relevant input, especially useful for detecting fine color changes 
    (e.g. when ghosts in Pacman turn blue and return to normal).

    Parameters:
    -----------
    env : gym.Env
        The original Atari environment.
    size : int
        The output resolution (default is 84), producing (size x size x 3) RGB images.
    """
    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size

        # Original frame dimensions must match expected Atari resolution
        orig_h, orig_w, c = env.observation_space.shape
        self.y1, self.y2 = 34, 194  # Crop from row 34 to 194 (160px height)
        assert orig_h == 210 and orig_w == 160, "CropPlayfieldWrapper expects input size (210x160x3)"

        # Define new observation space
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.size, self.size, c),
            dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        """
        Applies cropping and resizing to the observation.

        Parameters:
        -----------
        obs : np.ndarray
            Raw RGB frame from the Atari environment with shape (210, 160, 3)

        Returns:
        --------
        np.ndarray
            Processed RGB frame with shape (size, size, 3)
        """
        # Crop to the 160x160 playable region
        play = obs[self.y1:self.y2, :, :]  # Shape: (160, 160, 3)

        # Resize to (size, size) using area interpolation for best quality
        resized = cv2.resize(play, (self.size, self.size), interpolation=cv2.INTER_AREA)

        return resized