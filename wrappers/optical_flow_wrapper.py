import gymnasium as gym
import numpy as np
from cv2 import calcOpticalFlowFarneback, resize, cvtColor, COLOR_RGB2GRAY

class OpticalFlowWrapper(gym.ObservationWrapper):
    """
    Adds optical flow (X, Y) as extra channels to observations.
    """
    def __init__(self, env, downscale_factor=0.25):
        super().__init__(env)
        self.downscale_factor = downscale_factor
        self.prev_frame = None

        # Update observation space: +2 channels (flow_x, flow_y)
        obs_shape = self.observation_space.shape
        channels = obs_shape[0] + 2  # original channels + flow_x + flow_y
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(channels, *obs_shape[1:]), dtype=np.float32
        )

    def observation(self, obs):
        curr_frame = obs.transpose(1, 2, 0)  # (C, H, W) ➝ (H, W, C)

        if self.prev_frame is None:
            # On first call, no flow available: use zeros
            flow_x = np.zeros(curr_frame.shape[:2], dtype=np.float32)
            flow_y = np.zeros(curr_frame.shape[:2], dtype=np.float32)
        else:
            # Compute flow
            prev_gray = cvtColor(self.prev_frame, COLOR_RGB2GRAY)
            curr_gray = cvtColor(curr_frame, COLOR_RGB2GRAY)

            # Downscale
            small_prev = resize(prev_gray, (0, 0), fx=self.downscale_factor, fy=self.downscale_factor)
            small_curr = resize(curr_gray, (0, 0), fx=self.downscale_factor, fy=self.downscale_factor)

            # Optical flow
            flow = calcOpticalFlowFarneback(small_prev, small_curr, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
            flow_x, flow_y = flow[..., 0], flow[..., 1]

            # Upscale back
            flow_x = resize(flow_x, prev_gray.shape[::-1])
            flow_y = resize(flow_y, prev_gray.shape[::-1])

        # Update prev_frame
        self.prev_frame = curr_frame.copy()

        # Add flow_x, flow_y as channels
        flow_stack = np.stack([flow_x, flow_y], axis=0)  # shape (2, H, W)
        obs_with_flow = np.concatenate([obs, flow_stack], axis=0)

        return obs_with_flow

    def reset(self, **kwargs):
        self.prev_frame = None
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info