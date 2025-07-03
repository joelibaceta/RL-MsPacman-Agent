import torch as th
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class RGBCNNRNN(BaseFeaturesExtractor):
    """
    Hybrid CNN + GRU feature extractor for stacked RGB observations.

    1) Procesa cada uno de los S frames RGB (84×84×3) con un pequeño CNN.
    2) Toma la secuencia de embeddings (de tamaño cnn_out) y la pasa por un GRU.
    3) Proyecta el último estado oculto del GRU a un vector de salida de `features_dim`.

    Input:
        observations: Tensor con forma (batch_size, S, H, W, C)
    Output:
        Tensor con forma (batch_size, features_dim)
    """

    def __init__(
        self,
        observation_space: Box,
        features_dim: int = 512,
        rnn_hidden_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)
        # Extraemos dimensiones
        stack_size, height, width, channels = observation_space.shape

        # --- 1) CNN por frame RGB ---
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calcular dinámicamente el tamaño de salida del CNN
        with th.no_grad():
            dummy = th.zeros(1, channels, height, width)
            cnn_out_size = self.cnn(dummy).shape[1]

        # --- 2) GRU sobre la secuencia de embeddings ---
        self.gru = nn.GRU(
            input_size=cnn_out_size,
            hidden_size=rnn_hidden_dim,
            batch_first=True,
        )

        # --- 3) Proyección final a features_dim ---
        self.linear = nn.Sequential(
            nn.Linear(rnn_hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        observations: (B, S, H, W, C)
        """
        B, S, H, W, C = observations.shape

        # 1) Reajustar para procesar cada frame por separado:
        #    (B, S, H, W, C) -> (B*S, C, H, W)
        x = observations.permute(0, 1, 4, 2, 3).reshape(B * S, C, H, W)
        # CNN
        cnn_features = self.cnn(x)  # (B*S, cnn_out_size)
        # Volver a (B, S, cnn_out_size)
        seq_embed = cnn_features.view(B, S, -1)

        # 2) GRU
        seq_out, _ = self.gru(seq_embed)  # (B, S, rnn_hidden_dim)
        last_hidden = seq_out[:, -1, :]   # (B, rnn_hidden_dim)

        # 3) Proyección final
        return self.linear(last_hidden)   # (B, features_dim)