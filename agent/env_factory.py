from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from typing import Literal
import gymnasium as gym
import multiprocessing
# Custom Wrappers
from wrappers.custom_noop_reset_wrapper import CustomNoopResetWrapper
from wrappers.crop_playfield_wrapper import CropPlayfieldWrapper
from wrappers.rescaled_reward_wrapper import AutoRescaledRewardWrapper

from gymnasium.wrappers import RecordVideo, FrameStackObservation

from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    EpisodicLifeEnv,
)

class MsPacmanEnvFactory:
    """
    Factory class for constructing customized Ms. Pac-Man environments for Deep Reinforcement Learning.
    It encapsulates a preprocessing pipeline with modular wrappers and supports optional vectorization
    for parallel environments.

    Parameters:
    -----------
    record_video : bool
        If True, records gameplay videos every 1000 episodes for later visual inspection.
    debug : bool
        If True, disables video recording to speed up training in debug mode.
    vec_type : str, optional
        Type of vectorized environment to use. Options:
            - 'dummy'   : Uses DummyVecEnv (single-process, useful on macOS or for debugging)
            - 'subproc' : Uses SubprocVecEnv (multi-process, better performance in training)
            - None      : Returns a single non-vectorized environment
    n_envs : int
        Number of parallel environments to create if vec_type is not None.
    """

    def __init__(
        self,
        record_video: bool = False,
        debug: bool = False,
        vec_type: Literal["dummy", "subproc", None] = None,
        n_envs: int = 1,
    ):
        self.record_video = record_video
        self.debug = debug
        self.vec_type = vec_type
        self.n_envs = n_envs

    def _make_single_env(self):
        """
        Returns a closure that creates a single environment instance with all preprocessing steps.
        Useful for VecEnv initialization.
        """
        def _init():
            # 1. Load Ms. Pac-Man with RGB rendering (used for custom preprocessing)
            env = gym.make("ALE/MsPacman-v5", render_mode=None)

            # 2. Optionally record videos every 1000 episodes
            if self.record_video:
                env = RecordVideo(
                    env,
                    video_folder="./videos",
                    episode_trigger=lambda episode_id: episode_id % 1000 == 0,
                )

            # 3. Apply random no-op actions at reset to introduce variability
            env = CustomNoopResetWrapper(env, noop_max=30)

            # 4. Skip frames and apply max-pooling to reduce computational load and preserve motion
            env = MaxAndSkipEnv(env, skip=4)

            # 5. End episodes on life loss to provide denser reward feedback
            env = EpisodicLifeEnv(env)

            # 6. Crop HUD and borders, resize to 84x84 while preserving aspect ratio
            env = CropPlayfieldWrapper(env, size=84)

            # 7. Stack last 4 frames to give agent a sense of motion
            env = FrameStackObservation(env, stack_size=4)

            # 8. Rescale reward values to a smooth [-1, 1] range using tanh
            env = AutoRescaledRewardWrapper(env, warmup_episodes=10)

            return env

        return _init

    def build(self):
        """
        Builds the final environment or vectorized set of environments according to the selected mode.

        Returns:
        --------
        - gym.Env or VecEnv: A single preprocessed environment, or a vectorized set of environments.
        """
        if self.vec_type == "dummy":
            # Use single-process vectorized environment (safer on macOS or in notebooks)
            return DummyVecEnv([self._make_single_env() for _ in range(self.n_envs)])
        elif self.vec_type == "subproc":
            # Use subprocessed vectorized environment for performance (requires fork-capable OS)
            return SubprocVecEnv([self._make_single_env() for _ in range(self.n_envs)])
        else:
            # Return a single non-vectorized environment
            return self._make_single_env()()