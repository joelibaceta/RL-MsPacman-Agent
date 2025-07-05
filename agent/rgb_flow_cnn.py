import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box

class RGBFlowCNN(BaseFeaturesExtractor):
    """
    Dual-branch CNN: separa procesamiento de frames RGB apilados y optical flow,
    para luego combinar las características y producir una representación latente.

    Soporta observaciones en dos formatos:
      1. (stack_size, H, W, C) canales-ultimo (RGBCNN estilo)
      2. (C, H, W) canales-primero (flow+frames)
    """
    def __init__(
        self,
        observation_space: Box,
        features_dim: int = 512,
        flow_channels: int = 2,
    ):
        super().__init__(observation_space, features_dim)

        shape = observation_space.shape
        # Determinar número de canales de frames y flow según el formato
        if len(shape) == 4:
            # formato RGBCNN: (stack_size, height, width, channels)
            stack_size, height, width, channels = shape
            frame_channels = stack_size * channels
            total_channels = frame_channels + flow_channels
        elif len(shape) == 3:
            # formato canales-primero: (total_channels, height, width)
            total_channels, height, width = shape
            frame_channels = total_channels - flow_channels
            channels = None
        else:
            raise ValueError(f"Unsupported observation space shape: {shape}")

        self.frame_channels = frame_channels
        self.flow_channels = flow_channels

        # Rama para frames RGB apilados
        self.frames_branch = nn.Sequential(
            nn.Conv2d(frame_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Rama para optical flow
        self.flow_branch = nn.Sequential(
            nn.Conv2d(flow_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calcular dinámicamente el tamaño de salida de cada rama
        with th.no_grad():
            frames_dummy = th.zeros(1, frame_channels, height, width)
            flow_dummy = th.zeros(1, flow_channels, height, width)
            frames_feat = self.frames_branch(frames_dummy)
            flow_feat = self.flow_branch(flow_dummy)
            concat_size = frames_feat.shape[1] + flow_feat.shape[1]

        # Capa final para combinar
        self.linear = nn.Sequential(
            nn.Linear(concat_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Si viene en formato (B, stack_size, H, W, C), reordenar a canales-primero
        if observations.ndim == 5:
            B, S, H, W, C = observations.shape
            observations = observations.permute(0, 1, 4, 2, 3).reshape(B, S * C, H, W)

        # Ahora observations tiene forma (B, total_channels, H, W)
        # Separar frames y flow
        frames_tensor = observations[:, :self.frame_channels, :, :]
        flow_tensor = observations[:, self.frame_channels:, :, :]

        # Extraer características
        frames_features = self.frames_branch(frames_tensor)
        flow_features = self.flow_branch(flow_tensor)

        # Combinar y proyectar
        combined = th.cat([frames_features, flow_features], dim=1)
        return self.linear(combined)
