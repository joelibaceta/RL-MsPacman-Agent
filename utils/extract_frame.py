import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

from stable_baselines3.common.evaluation import evaluate_policy


env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")

obs, info = env.reset()

# Ejecuta un paso para obtener otro frame
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

# `obs` ya contiene el frame como array (shape: [210, 160, 3])
print("Shape:", obs.shape)  # RGB (alto, ancho, canales)

# Guarda el frame como imagen
from PIL import Image
img = Image.fromarray(obs)
img.save("frame.png")