from dataclasses import dataclass

@dataclass
class TransformerConfig:
    """Hyperparameter configuration for Neural Transformer sequence models."""
    input_size: int = 17
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    feedforward: int = 512
    dropout: float = 0.1
    context_length: int = 128
    horizons: int = 3
    classes: int = 10
