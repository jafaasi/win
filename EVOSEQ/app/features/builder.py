from typing import Sequence, List, Dict, Any, Optional
import numpy as np

from .types import SequenceFeatures
from .encoding import encode_outcome
from .entropy import categorical_entropy
from .conditional_entropy import conditional_entropy
from .information import information_gain
from .autocorrelation import autocorrelation
from .runs import current_run_length
from .transitions import transition_matrix, transition_entropy
from .lz import lz_complexity

def build_features(
    digits: Sequence[int],
    sizes: Optional[Sequence[int]] = None,
    colors: Optional[Sequence[int]] = None,
    parities: Optional[Sequence[int]] = None,
) -> SequenceFeatures:
    """
    Constructs a comprehensive canonical SequenceFeatures object and 36-dimensional feature vector.
    """
    digits = list(digits)
    n = len(digits)
    if n == 0:
        raise ValueError("Sequence must not be empty.")
        
    # Infer derived series if not provided
    if sizes is None:
        sizes = [1 if d >= 5 else 0 for d in digits]
    else:
        sizes = list(sizes)
        
    if colors is None:
        colors = [1 if d in [1, 3, 7, 9] else 2 if d in [0, 5] else 0 for d in digits]
    else:
        colors = list(colors)
        
    if parities is None:
        parities = [1 if d % 2 != 0 else 0 for d in digits]
    else:
        parities = list(parities)

    # 1. Base 17-dimensional One-Hot Categorical Vector
    base_cat = encode_outcome(digits[-1], sizes[-1], colors[-1], parities[-1])
    
    # 2. Marginal Entropies
    h_dig = categorical_entropy(digits, 10)
    h_siz = categorical_entropy(sizes, 2)
    h_col = categorical_entropy(colors, 3)
    h_par = categorical_entropy(parities, 2)
    
    # 3. Conditional Entropies
    h_cond_1 = conditional_entropy(digits, order=1, cardinality=10)
    h_cond_2 = conditional_entropy(digits, order=2, cardinality=10)
    h_cond_3 = conditional_entropy(digits, order=3, cardinality=10)
    
    # 4. Information Gains
    ig_1 = information_gain(digits, order=1, cardinality=10)
    ig_2 = information_gain(digits, order=2, cardinality=10)
    ig_3 = information_gain(digits, order=3, cardinality=10)
    
    # 5. Autocorrelations
    acf_1 = autocorrelation(digits, lag=1)
    acf_2 = autocorrelation(digits, lag=2)
    acf_3 = autocorrelation(digits, lag=3)
    
    # 6. Current Run Lengths
    run_d = current_run_length(digits)
    run_s = current_run_length(sizes)
    run_c = current_run_length(colors)
    run_p = current_run_length(parities)
    
    # 7. Transition Dynamics
    t_mat = transition_matrix(digits, cardinality=10)
    t_ent = transition_entropy(t_mat)
    
    # 8. LZ Algorithmic Complexity (normalized)
    lz_val = float(lz_complexity(digits))
    lz_norm = lz_val / float(max(1, len(digits)))
    
    # Assemble full 36-dimensional feature vector
    vector_elements = [
        *base_cat,
        h_dig, h_siz, h_col, h_par,
        h_cond_1, h_cond_2, h_cond_3,
        ig_1, ig_2, ig_3,
        acf_1, acf_2, acf_3,
        float(run_d), float(run_s), float(run_c), float(run_p),
        t_ent,
        lz_norm
    ]
    vector = np.asarray(vector_elements, dtype=np.float32)

    return SequenceFeatures(
        digit=digits[-1],
        size=sizes[-1],
        color=colors[-1],
        parity=parities[-1],
        entropy_digit=round(h_dig, 4),
        entropy_size=round(h_siz, 4),
        entropy_color=round(h_col, 4),
        entropy_parity=round(h_par, 4),
        conditional_entropy_1=round(h_cond_1, 4),
        conditional_entropy_2=round(h_cond_2, 4),
        conditional_entropy_3=round(h_cond_3, 4),
        information_gain_1=round(ig_1, 4),
        information_gain_2=round(ig_2, 4),
        information_gain_3=round(ig_3, 4),
        run_digit=run_d,
        run_size=run_s,
        run_color=run_c,
        run_parity=run_p,
        autocorrelation_1=round(acf_1, 4),
        autocorrelation_2=round(acf_2, 4),
        autocorrelation_3=round(acf_3, 4),
        transition_entropy=round(t_ent, 4),
        lz_complexity=lz_val,
        vector=vector
    )

def build_temporal_tensor(
    sequence: Sequence[int],
    context_length: int = 128,
    window_size: int = 64
) -> np.ndarray:
    """
    Constructs a causal temporal feature matrix of shape [context_length, feature_dim]
    X_t = [f_{t-L+1}, ..., f_{t-1}, f_t]
    """
    sequence = list(sequence)
    if len(sequence) < context_length:
        # Prepend pad with first symbol
        pad_size = context_length - len(sequence)
        sequence = [sequence[0]] * pad_size + sequence
        
    start_idx = len(sequence) - context_length
    temporal_rows = []
    
    for i in range(start_idx, len(sequence)):
        # Causal window ending at step i
        sub_seq = sequence[max(0, i + 1 - window_size):i + 1]
        feat = build_features(sub_seq)
        temporal_rows.append(feat.vector)
        
    tensor = np.asarray(temporal_rows, dtype=np.float32) # [context_length, feature_dim]
    return tensor
