import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

from stable_baselines3.common.evaluation import evaluate_policy

import wandb
from wandb.integration.sb3 import WandbCallback

wandb.init(
    project="pacman-rl",
    config={"algo": "DQN", "env": "MsPacman-v5"},
    sync_tensorboard=True,  # Sincroniza con TensorBoard
    monitor_gym=True,
    save_code=True,
)

gym.register_envs(ale_py)

# Crea el entorno
env = gym.make('ALE/MsPacman-v5', render_mode="rgb_array")
env = AtariWrapper(env)

device = torch.device("mps")

# Crea el modelo DQN
model = DQN("CnnPolicy", env, verbose=1, tensorboard_log="./logs", device=device)

# Entrena por 5M timesteps
model.learn(total_timesteps=5_000_000, callback=WandbCallback())

mean_reward, std = evaluate_policy(model, env, n_eval_episodes=5)
print(f"Reward promedio: {mean_reward:.2f}")

# Guarda el modelo entrenado
model.save("dqn_mspacman")

env.close()
