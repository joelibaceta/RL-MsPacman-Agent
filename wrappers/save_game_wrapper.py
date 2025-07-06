import gymnasium as gym
import numpy as np
from collections import deque

class SaveGameWrapper(gym.Wrapper):
    """
    Guarda checkpoints N pasos antes de la muerte para poder reintentar
    justo antes del momento crítico.
    """
    def __init__(self, env, max_saves=10, resume_prob=0.2, pre_save_steps=10):
        super().__init__(env)
        self.max_saves      = max_saves
        self.resume_prob    = resume_prob
        self.pre_save_steps = pre_save_steps

        # Buffer de estados ALE: tamaño pre_save_steps+1
        self.state_buffer = deque(maxlen=pre_save_steps+1)
        # Lista de checkpoints “anteriores a la muerte”
        self.checkpoints  = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Reiniciar buffer y meter el estado inicial
        self.state_buffer.clear()
        initial_state = self.env.unwrapped.ale.cloneState()
        self.state_buffer.append(initial_state)
        return obs, info

    def step(self, action):
        # Antes de avanzar, clonamos el estado actual
        current_state = self.env.unwrapped.ale.cloneState()
        self.state_buffer.append(current_state)

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Si muere, guardamos N pasos atrás
        if terminated and len(self.checkpoints) < self.max_saves:
            # Si buffer lleno, el primer elemento es el estado N pasos antes
            idx = 0 if len(self.state_buffer) <= self.pre_save_steps else 0
            state_to_save = self.state_buffer[0]
            self.checkpoints.append(state_to_save)

        # Lógica de “resume silencioso” sin terminar episodio
        if terminated and self.checkpoints and np.random.rand() < self.resume_prob:
            # Restaurar un checkpoint al azar
            state = self.checkpoints[np.random.randint(len(self.checkpoints))]
            self.env.unwrapped.ale.restoreState(state)
            obs = self.env.unwrapped.getScreenRGB()
            # devolvemos done=False para que no cuente como episodio nuevo
            return obs, reward, False, False, info

        return obs, reward, terminated, truncated, info