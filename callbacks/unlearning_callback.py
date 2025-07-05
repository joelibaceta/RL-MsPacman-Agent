from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

class UnlearningCallback(BaseCallback):
    def __init__(self, len_threshold=90, step_threshold=5_200_000, high_eps=0.2, verbose=1):
        super().__init__(verbose)
        self.len_threshold = len_threshold
        self.step_threshold = step_threshold
        self.high_eps = high_eps

    def _on_step(self) -> bool:
        # Get ep_len_mean from TensorBoard logs
        ep_len_mean = self.logger.name_to_value.get('rollout/ep_len_mean', 0)

        # If critical phase reached, increase exploration
        if self.num_timesteps > self.step_threshold or ep_len_mean > self.len_threshold:
            if self.model.exploration_rate < self.high_eps:
                if self.verbose:
                    print(f"\n⚠️ Unlearning triggered at step {self.num_timesteps} - Increasing ε to {self.high_eps}")
                self.model.exploration_rate = self.high_eps
        return True