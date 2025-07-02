import argparse
import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from agent.env_factory import MsPacmanEnvFactory
from agent.rgb_cnn import RGBCNN


def load_env(record=False, video_folder=None):
    def make_env():
        # Usamos la fábrica con vec_type=None para obtener el env plano
        if record and video_folder:
            os.makedirs(video_folder, exist_ok=True)
            return MsPacmanEnvFactory(vec_type=None, render_mode="rgb_array").build()
        else:
            return MsPacmanEnvFactory(vec_type=None, render_mode="human").build()

    env = DummyVecEnv([make_env])

    if record and video_folder:
        env = VecVideoRecorder(
            env,
            video_folder=video_folder,
            record_video_trigger=lambda step: step == 0,
            video_length=4000,
            name_prefix="ms_pacman_eval",
        )

    return env


def main():
    parser = argparse.ArgumentParser(description="Visualizar agente Ms. Pac-Man entrenado.")
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo guardado (.zip)")
    parser.add_argument("--episodes", type=int, default=1, help="Número de episodios a reproducir")
    parser.add_argument("--record", action="store_true", help="Grabar video del episodio (no render en vivo)")
    parser.add_argument("--video_dir", type=str, default="videos", help="Carpeta para guardar los videos")

    args = parser.parse_args()

    env = load_env(record=args.record, video_folder=args.video_dir)

    model = DQN.load(
        args.model,
        env=env,
        custom_objects={
            "features_extractor_class": RGBCNN,
            "features_extractor_kwargs": dict(features_dim=512),
        },
    )

    total_rewards = []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]

            if not args.record:
                env.render()

        total_rewards.append(ep_reward)
        print(f"Episode {ep + 1} reward: {ep_reward:.2f}")

    print(f"\nAverage reward over {args.episodes} episode(s): {sum(total_rewards) / len(total_rewards):.2f}")
    env.close()


if __name__ == "__main__":
    gym.register_envs(ale_py)
    main()