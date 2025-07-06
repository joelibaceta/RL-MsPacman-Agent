from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from typing import Literal
import gymnasium as gym
import multiprocessing
# Custom Wrappers
from wrappers.custom_noop_reset_wrapper import CustomNoopResetWrapper
from wrappers.crop_playfield_wrapper import CropPlayfieldWrapper
from wrappers.rescaled_reward_wrapper import AutoRescaledRewardWrapper
from wrappers.death_penalty_wrapper import DeathPenaltyWrapper
from wrappers.survival_bonus_wrapper import SurvivalBonusWrapper
from wrappers.ghost_escape_reward_wrapper import GhostEscapeRewardWrapper
from wrappers.movement_reward_wrapper import MovementRewardWrapper
from wrappers.random_initial_movement_wrapper import RandomInitialMovementWrapper
from wrappers.save_game_wrapper import SaveGameWrapper

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
        render_mode=None
    ):
        self.record_video = record_video
        self.debug = debug
        self.vec_type = vec_type
        self.n_envs = n_envs
        self.render_mode = render_mode

    def _make_single_env(self):
        """
        Returns a closure that creates a single environment instance with all preprocessing steps.
        Useful for VecEnv initialization.
        """
        def _init():
            # Load Ms. Pac-Man with RGB rendering (used for custom preprocessing)
            env = gym.make("ALE/MsPacman-v5", render_mode=self.render_mode)

            env = SaveGameWrapper(env, max_saves=10, resume_prob=0.2, pre_save_steps=10)

            env = RandomInitialMovementWrapper(env, min_steps=30, max_steps=50, safe_retry=True)
            # 1. Apply random no-op actions at reset to introduce initial state variability
            # env = CustomNoopResetWrapper(env, noop_max=30)

           
            # 2. Skip frames and apply max-pooling to reduce computation and preserve motion
            env = MaxAndSkipEnv(env, skip=4)

            # 8. Crop HUD and borders, then resize to 84x84 while preserving aspect ratio
            env = CropPlayfieldWrapper(env, size=84)

            # 9. Stack the last 4 frames to give the agent a sense of motion
            env = FrameStackObservation(env, stack_size=4)

            # 4. Explicitly penalize life loss to encourage survival behavior
            env = DeathPenaltyWrapper(env)

            # 10. Smoothly rescale all reward values to a consistent [-1, 1] range
            env = AutoRescaledRewardWrapper(env, warmup_episodes=10)

            # 5. Add a small bonus per step survived to promote longer episodes
            env = SurvivalBonusWrapper(env, bonus_per_step=0.05)

            # 6. Reward the agent for escaping nearby ghosts (only when not energized)
            env = GhostEscapeRewardWrapper(env, danger_radius=16, escape_reward=0.2, chase_reward=0.1)

            # 7. Encourage movement and penalize staying idle
            env = MovementRewardWrapper(env, move_reward=0.1, idle_penalty=0.1)

            # 3. End episodes on life loss to provide denser reward feedback
            env = EpisodicLifeEnv(env)

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
        
