import copy
import random
from typing import Dict, Any, Optional

def compute_adaptive_exploration(
    uncertainty: float,
    drift_score: float,
    recent_improvement: float,
    base_exploration: float = 0.20
) -> float:
    """
    Dynamically balances exploration vs exploitation:
    High drift or deteriorating performance -> High exploration.
    Stable performance and low drift -> Low exploration (exploit known architectures).
    """
    factor = base_exploration + (drift_score * 0.5)
    if recent_improvement < 0:
        factor += abs(recent_improvement) * 0.5
    return float(min(0.80, max(0.05, factor)))

def mutate_model_parameters(
    model_name: str,
    parameters: Dict[str, Any],
    mutation_rate: float = 0.30,
    exploration_factor: float = 0.20
) -> Dict[str, Any]:
    """
    Mutates model hyperparameters within architecture-specific valid parameter spaces.
    """
    child = copy.deepcopy(parameters)
    
    if model_name in ["Markov", "markov"]:
        if random.random() < mutation_rate:
            # Order: 1, 2, 3, 4, 5
            delta_ord = random.choice([-1, 1])
            child["order"] = max(1, min(5, child.get("order", 2) + delta_ord))
        if random.random() < mutation_rate:
            # Smoothing: [0.01, 2.0]
            mult = random.choice([0.5, 0.8, 1.25, 2.0])
            child["smoothing"] = round(max(0.01, min(2.0, child.get("smoothing", 0.5) * mult)), 4)

    elif model_name in ["DiscreteHMM", "hmm"]:
        if random.random() < mutation_rate:
            # States: 2, 3, 4, 6, 8
            child["n_states"] = random.choice([2, 3, 4, 6, 8])
        if random.random() < mutation_rate:
            mult = random.choice([0.5, 2.0])
            child["smoothing"] = max(1e-4, min(1e-1, child.get("smoothing", 1e-3) * mult))

    elif model_name in ["EchoStateNetwork", "esn"]:
        if random.random() < mutation_rate:
            # Reservoir size: 32, 64, 128, 256, 512
            child["reservoir_size"] = random.choice([32, 64, 128, 256, 512])
        if random.random() < mutation_rate:
            # Spectral radius: [0.5, 1.2]
            delta_r = random.choice([-0.1, 0.1])
            child["spectral_radius"] = round(max(0.5, min(1.2, child.get("spectral_radius", 0.9) + delta_r)), 2)
        if random.random() < mutation_rate:
            # Leak rate: [0.1, 0.9]
            child["leak_rate"] = round(max(0.1, min(0.9, child.get("leak_rate", 0.3) + random.choice([-0.1, 0.1]))), 2)
        if random.random() < mutation_rate:
            # Ridge: [1e-5, 1e-2]
            child["ridge"] = max(1e-5, min(1e-2, child.get("ridge", 1e-4) * random.choice([0.1, 10.0])))

    elif model_name in ["Transformer", "transformer"]:
        if random.random() < mutation_rate:
            # Hidden size: 16, 32, 64, 128
            child["hidden_size"] = random.choice([16, 32, 64, 128])
        if random.random() < mutation_rate:
            # Heads: 2, 4
            child["heads"] = random.choice([2, 4])
        if random.random() < mutation_rate:
            # Layers: 1, 2, 3
            child["layers"] = random.choice([1, 2, 3])
        if random.random() < mutation_rate:
            # Context length: 32, 64, 128, 256
            child["context_length"] = random.choice([32, 64, 128, 256])
        if random.random() < mutation_rate:
            # Temperature: [0.5, 2.0]
            child["temperature"] = round(max(0.5, min(2.0, child.get("temperature", 1.1) + random.choice([-0.2, 0.2]))), 2)

    elif model_name in ["S4", "s4", "Mamba", "mamba"]:
        if random.random() < mutation_rate:
            # Hidden size: 16, 32, 48, 64
            child["hidden_size"] = random.choice([16, 32, 48, 64])
        if random.random() < mutation_rate:
            # Layers: 1, 2, 3
            child["layers"] = random.choice([1, 2, 3])
        if random.random() < mutation_rate:
            # Context length: 32, 64, 128, 256
            child["context_length"] = random.choice([32, 64, 128, 256])
        if random.random() < mutation_rate:
            # Temperature: [0.5, 2.0]
            child["temperature"] = round(max(0.5, min(2.0, child.get("temperature", 1.1) + random.choice([-0.2, 0.2]))), 2)

    return child

