import torch
from datetime import datetime

import ale_py
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import CheckpointCallback
from agent.env_factory import MsPacmanEnvFactory
from stable_baselines3.common.vec_env import VecMonitor

from stable_baselines3.common.utils import get_schedule_fn

# DEFAULT CNN


# Custom CNN Feature Extractor
from agent.rgb_cnn import RGBCNN
from agent.rgb_rnn import RGBCNNRNN

if __name__ == "__main__":
    DEBUG = True

    if DEBUG:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb.init(
            project="pacman-rl",
            config={"algo": "DQN", "env": "MsPacman-v5"},
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

    gym.register_envs(ale_py)
    torch.set_num_threads(12)

    # === Environment preprocessing pipeline for training Ms. Pac-Man agent ===
    # We use a vectorized environment with multiple parallel instances of the game (SubprocVecEnv).
    # This means the agent learns from several Ms. Pac-Man environments running at the same time,
    # each with its own random conditions. This speeds up training and improves generalization.
    # By leveraging multiprocessing (each environment in its own process), we can utilize multiple CPU cores.
    # On modern machines like Apple Silicon, this allows us to take full advantage of available hardware.
    env = MsPacmanEnvFactory(vec_type="subproc", n_envs=12).build()
    env = VecMonitor(env)

    device = torch.device(
        "mps"
    )  # Use MPS (Metal Performance Shaders) for GPU acceleration on MacOS

    # === Create and configure the DQN model ===
    # This setup defines the core hyperparameters for training a DQN agent using a custom CNN policy.
    # The chosen parameters are based on best practices for Atari environments, with some adjustments
    # for better convergence speed and training stability.

    # Custom policy configuration: Specifies a custom feature extractor (RGBCNN) to be used by the DQN policy.
    policy_kwargs = dict(
        features_extractor_class=RGBCNN, features_extractor_kwargs=dict(features_dim=512)
    )

    lr_schedule = get_schedule_fn(1e-4) 

    model = DQN(
        policy="CnnPolicy",  # Use a convolutional neural network policy (customizable via policy_kwargs)
        env=env,  # Preprocessed Atari environment
        learning_rate=lr_schedule,  # Learning rate (higher than the SB3 default of 2.5e-4 is *not* true here, default is 1e-4)
        buffer_size=100_000,  # Replay buffer size: number of transitions to store
        learning_starts=10_000,  # Delays training until 10k steps have been collected
        batch_size=64,  # Mini-batch size sampled from the replay buffer
        tau=1.0,  # Soft update coefficient for the target network (1.0 = hard copy every update)
        gamma=0.99,  # Discount factor for future rewards
        train_freq=4,  # Train the model every 4 environment steps
        target_update_interval=1_000,  # Update the target network every 1000 training steps
        exploration_fraction=0.2,  # Fraction of total timesteps over which ε is decayed
        exploration_final_eps=0.01,  # Final value of ε after decay
        max_grad_norm=10,  # Clip gradients to prevent exploding gradients
        verbose=1,  # Enable logging
        tensorboard_log="./logs",  # Path to save TensorBoard logs
        device=device,  # MPS because we are on MacOS
        policy_kwargs=policy_kwargs,  # Custom CNN architecture passed to the policy
    )

    # === Training Phase ===
    # We train the agent for 5 million timesteps.
    # This number is based on common practice in the literature when training DQN agents on Atari environments.
    # For reference, the original DQN paper (Mnih et al., 2015) trained for 200 million frames,
    # which translates to 50 million timesteps with a frame skip of 4.
    # Here, we opt for 5 million timesteps to strike a balance between training time and policy performance,
    # especially useful for early experiments, debugging, or benchmarking.
    # This is typically enough to see substantial improvement in simple environments like MsPacman,
    # while still allowing for extended training later (e.g., 10M, 20M+) if needed.
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,              # Cada 100k timesteps
        save_path="./checkpoints/",     # Carpeta donde se guardan
        name_prefix="dqn_mspacman",     # Prefijo de los archivos
        save_replay_buffer=False,        # Opcional: guarda el buffer de replay
        save_vecnormalize=True          # Opcional: si usas normalización
    )

    callbacks = [checkpoint_callback]
    if DEBUG:
        wandb_callback = WandbCallback(
            gradient_save_freq=1000, model_save_path="./models/", verbose=2, log="all",
        )
        callbacks.append(wandb_callback)

    model.learn(total_timesteps=10_000_000, callback=callbacks)

    # Evaluate the performance of the trained model on the current environment.
    # Runs 5 full episodes (n_eval_episodes=5) in deterministic mode (no exploration)
    # to obtain an estimate of the agent's average reward.

    n_eval_episodes = 5 if DEBUG else 20
    mean_reward, std = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    print(f"Reward promedio: {mean_reward:.2f}")


    # Generate a unique name with timestamp to avoid overwriting previous models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/dqn_mspacman_{timestamp}"

    # Save the trained model to the specified path
    model.save(model_path)
    print(f"Trained model saved at: {model_path}")

    # Close the environment to release resources
    env.close()
