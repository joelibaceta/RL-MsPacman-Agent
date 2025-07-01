import torch as th
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class RGBCNN(BaseFeaturesExtractor):
    """
    Custom CNN for extracting spatial and temporal features from stacked RGB observations.

    This class extends `BaseFeaturesExtractor` from Stable-Baselines3 and is tailored for environments like MsPacman,
    where color information is semantically meaningful. Unlike classic architectures that convert frames to grayscale,
    this network retains the full RGB channels.

    Design Hypothesis:
    ------------------
    In environments such as MsPacman, empirical observations suggest that grayscale-based networks often fail to
    capture crucial visual cues — especially when game state information is encoded through color changes.
    A notable example is when ghosts turn blue (edible) and then revert to their original color. Grayscale conversion
    can remove this distinction, causing the agent to behave incorrectly, such as continuing to chase non-edible ghosts.

    This network is based on the hypothesis that color acts as a **semantic signal**, and preserving RGB channels:
        - Helps differentiate visually similar but semantically different states.
        - Enables richer representations, supporting more nuanced and reactive policies.
        - Increases sensitivity to subtle visual cues, such as sprite color transitions or background shifts.

    Technical Details:
    ------------------
    - Expects a stacked RGB observation shaped as `(stack_size, height, width, 3)`.
    - Reorders and flattens temporal information into the channel dimension (e.g., 4 RGB frames → 12 channels).
    - Uses three convolutional layers followed by a fully connected layer to project to a latent space of `features_dim`.

    Parameters:
    -----------
    observation_space : gym.spaces.Box
        The observation space of the environment, typically shaped like (stack_size, H, W, 3).
    features_dim : int
        Dimension of the extracted feature vector. Default: 512.

    Output:
    -------
    torch.Tensor of shape (batch_size, features_dim), representing the latent features extracted from RGB input.
    """

    def __init__(self, observation_space: Box, features_dim: int = 512):
        """
        Initialize the RGB CNN feature extractor.

        Parameters:
        -----------
        observation_space : gym.spaces.Box
            Observation space from the environment. This can be either:
            - A stacked RGB input of shape (stack_size, height, width, 3), or
            - A preprocessed tensor of shape (height, width, channels).

            The input is automatically reshaped to match the CNN input format (N, C, H, W).

        features_dim : int
            Size of the output feature vector from the CNN. This will be the input to the policy or value head.

        Architecture:
        -------------
        The network consists of:
        - Three convolutional layers with ReLU activations.
        - A flattening layer to produce a 1D feature vector.
        - A fully connected layer mapping to `features_dim`.

        Input Handling:
        ---------------
        - If the observation is of shape (stack_size, H, W, 3), the frames are merged into the channel dimension,
          resulting in C = 3 * stack_size.
        - If the observation is already (H, W, C), no reshaping is needed.

        CNN Input Shape:
        ----------------
        After preprocessing, the CNN expects inputs of shape: (batch_size, channels, height, width)

        Feature Dimension Estimation:
        -----------------------------
        A dummy forward pass is performed with a sampled observation from the space, reshaped and permuted appropriately.
        This is used to infer the flattened size after convolutions, allowing for dynamic input sizing without hardcoding.
        """
                
        super().__init__(observation_space, features_dim)

        if len(observation_space.shape) == 4:
            stack_size, height, width, channels = observation_space.shape
            n_input_channels = channels * stack_size
        elif len(observation_space.shape) == 3:
            height, width, n_input_channels = observation_space.shape
        else:
            raise ValueError("Unsupported observation space shape")

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            sample = observation_space.sample()  # (4, 84, 84, 3)
            if len(sample.shape) == 4:
                sample = sample.transpose(1, 2, 3, 0).reshape(height, width, -1)  # (84, 84, 12)
            sample_input = th.as_tensor(sample[None]).float().permute(0, 3, 1, 2)  # (1, 12, 84, 84)
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass of the RGB CNN feature extractor.

        Parameters:
        -----------
        observations : torch.Tensor
            Input observation tensor of shape (B, S, H, W, C), where:
            - B: Batch size
            - S: Stack size (number of frames)
            - H, W: Height and width of each frame
            - C: Number of channels (e.g., 3 for RGB)

        Processing Steps:
        -----------------
        1. The input is permuted to shape (B, S, C, H, W) so that channels are adjacent.
        2. The frames and channels are merged into a single channel dimension: (B, S*C, H, W),
           effectively stacking RGB frames depth-wise.
        3. The reshaped input is passed through the convolutional neural network.
        4. The resulting feature map is flattened and passed through a fully connected layer.

        Returns:
        --------
        torch.Tensor
            A feature tensor of shape (B, features_dim), where `features_dim` is the size defined
            during initialization. This tensor can be used as input for policy or value networks.
        """

        B, S, H, W, C = observations.shape
        out = observations.permute(0, 1, 4, 2, 3).reshape(B, S * C, H, W)
        out = self.cnn(out)
        out = self.linear(out)
        return out