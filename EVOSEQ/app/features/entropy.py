import numpy as np

def entropy(probabilities):
    """Calculates Shannon entropy for a given probability vector."""
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log2(probabilities)))

def shannon_entropy(sequence, alphabet_size=10):
    """Computes empirical Shannon entropy in bits for an integer sequence."""
    if not len(sequence):
        return float(np.log2(alphabet_size))
    counts = np.bincount(sequence, minlength=alphabet_size)
    probs = counts / float(len(sequence))
    return entropy(probs)

def conditional_entropy(sequence, order=1, alphabet_size=10):
    """Computes conditional entropy H(X_t | X_{t-1:t-order})."""
    if len(sequence) <= order:
        return 0.0
    joint_counts = {}
    context_counts = {}
    
    for i in range(order, len(sequence)):
        ctx = tuple(sequence[i-order:i])
        target = sequence[i]
        context_counts[ctx] = context_counts.get(ctx, 0) + 1
        joint_counts[(ctx, target)] = joint_counts.get((ctx, target), 0) + 1
        
    total_samples = float(len(sequence) - order)
    cond_h = 0.0
    
    for (ctx, target), j_count in joint_counts.items():
        p_joint = j_count / total_samples
        p_cond = j_count / float(context_counts[ctx])
        if p_cond > 0:
            cond_h -= p_joint * np.log2(p_cond)
            
    return float(cond_h)
