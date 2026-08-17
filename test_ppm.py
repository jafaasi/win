import math
import random

def advanced_sequence_intelligence(history):
    if len(history) < 10:
        return [0.1]*10

    h_str = "".join([str(int(x) % 10) for x in history])
    
    # 1. Prediction by Partial Matching (PPM) / Variable Order Markov
    # We look at contexts of length 1 to 15.
    max_order = min(15, len(history)-1)
    
    probabilities = [0.0] * 10
    total_weight = 0.0
    
    for order in range(1, max_order + 1):
        context = h_str[-order:]
        
        counts = [0.0] * 10
        found = False
        
        start = 0
        while True:
            idx = h_str.find(context, start, -1)
            if idx == -1:
                break
            
            next_char = h_str[idx + order]
            digit = int(next_char)
            
            # Recency weighting
            distance = len(h_str) - (idx + order)
            weight = math.exp(-distance / 200.0)
            
            counts[digit] += weight
            found = True
            start = idx + 1
            
        if found:
            s_counts = sum(counts)
            # Exponentially favor longer, deeper context matches
            order_weight = math.pow(3.0, order) 
            
            for k in range(10):
                probabilities[k] += (counts[k] / s_counts) * order_weight
            total_weight += order_weight

    # Add base empirical prior to prevent 0 probabilities
    for k in range(10):
        probabilities[k] += 0.05
    total_weight += 0.5
    
    h1 = [p / total_weight for p in probabilities]
    s = sum(h1)
    h1 = [round(p / s, 4) for p in h1]
    h1[-1] = round(1.0 - sum(h1[:-1]), 4)
        
    return h1

history = [random.randint(0,9) for _ in range(1000)]
# Let's plant a deep pattern
history += [1,2,3,4,5, 9] * 3
history += [1,2,3,4,5]

h1 = advanced_sequence_intelligence(history)
print('Target Pattern 1,2,3,4,5 -> Next should be 9')
print([round(x, 4) for x in h1])
print('Predicted max:', h1.index(max(h1)), 'Probability:', max(h1))

