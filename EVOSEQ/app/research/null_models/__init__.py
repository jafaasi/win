from .iid import generate_iid
from .markov import estimate_transition, generate_markov
from .shuffle import block_shuffle, marginal_preserving_shuffle
from .surrogate import NullModelType, SurrogateHierarchy

iid_null = generate_iid
markov_null = generate_markov
block_shuffle_null = block_shuffle

__all__ = [
    "generate_iid",
    "iid_null",
    "estimate_transition",
    "generate_markov",
    "markov_null",
    "block_shuffle",
    "block_shuffle_null",
    "marginal_preserving_shuffle",
    "NullModelType",
    "SurrogateHierarchy"
]
