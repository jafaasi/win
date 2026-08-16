import torch
import torch.nn as nn
from abc import abstractmethod

class SequenceNetwork(nn.Module):
    """
    Standard PyTorch base module for deep sequence architectures (Transformer, Mamba, S4).
    Accepts tensor shape [Batch, Sequence_Length, Features] -> [Batch, Output_Classes].
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 10,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F] -> project to hidden dimension
        proj = self.input_projection(x)
        representation = self.encode(proj)
        final_state = representation[:, -1, :] # Take representation of the latest step
        logits = self.output_projection(final_state)
        return logits

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Applies sequence encoder (Self-Attention / SSM / State Space convolution)."""
        pass
