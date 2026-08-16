from typing import Sequence

def lz_complexity(sequence: Sequence[int]) -> int:
    """
    Computes Lempel-Ziv (LZ76) algorithmic complexity by parsing unique phrases.
    Higher values indicate greater randomness; lower values indicate repeating algorithmic patterns.
    """
    sequence = list(sequence)
    n = len(sequence)
    if n == 0:
        return 0
        
    phrases = 0
    i = 0
    while i < n:
        length = 1
        while (i + length <= n):
            candidate = tuple(sequence[i:i+length])
            # Check if candidate has appeared earlier in sequence[:i]
            is_seen = False
            for j in range(i - length + 1):
                if tuple(sequence[j:j+length]) == candidate:
                    is_seen = True
                    break
            if not is_seen:
                break
            length += 1
            
        phrases += 1
        i += length
        
    return phrases
