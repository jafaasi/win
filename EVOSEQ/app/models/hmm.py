from .hmm_model import DiscreteHMM, RegimeMonitor

class HMMModel(DiscreteHMM):
    """Alias for Discrete Hidden Markov Model with n_states."""
    name = "HMM"
    def __init__(self, states: int = 4, **kwargs):
        super().__init__(n_states=states, **kwargs)

__all__ = ["DiscreteHMM", "HMMModel", "RegimeMonitor"]
