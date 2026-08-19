#!/usr/bin/env python3
"""Quick sanity test for HighIntelligencePredictor without DB."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from backend.high_intelligence_predictor import (
    HighIntelligencePredictor,
    three_level_win_probability,
    recommend_strike_level,
)

def main():
    # Test 1: basic construction + small history
    hip = HighIntelligencePredictor()
    for d in [5, 3, 7, 2, 8, 1, 0, 5, 9, 4, 6, 3, 7, 2, 8, 1, 5, 0, 9, 4]:
        hip.add_observation(d)
    res = hip.predict()
    print('Test 1 (short history) OK:')
    print(f'  prediction={res.prediction} conf={res.confidence} target={res.targetNum} hedge={res.hedgeNum}')
    print(f'  P(single)={res.calibrated_p_single:.3f} P(win3)={res.calibrated_p_win_in_3:.3f} strike={res.strike_quality}')
    print(f'  entropy={res.entropy:.3f} change_p={res.change_probability:.3f} regime={res.regime_strength:.3f}')

    # Test 2: longer synthetic history with learnable 55% big bias
    np.random.seed(42)
    synthetic = []
    for _ in range(600):
        if np.random.random() < 0.55:
            synthetic.append(np.random.randint(5, 10))
        else:
            synthetic.append(np.random.randint(0, 5))
    hip2 = HighIntelligencePredictor.from_history(synthetic)
    res2 = hip2.predict()
    print()
    print('Test 2 (600-step biased history) OK:')
    print(f'  prediction={res2.prediction} conf%={res2.confidence:.1f}')
    print(f'  P(single)={res2.calibrated_p_single:.3f} P(win3)={res2.calibrated_p_win_in_3:.3f} strike={res2.strike_quality}')
    print(f'  CTW={res2.ctw_weight:.3f} Markov={res2.markov_weight:.3f} Streak={res2.streak_weight:.3f}')
    top5 = sorted(range(10), key=lambda i: -float(res2.digit_distribution[i]))[:5]
    print(f'  digit_dist_top5={top5}')

    # Test 3: joint probability + strike recommendation
    p = 0.60
    p3 = three_level_win_probability(p, 0.5 + 0.94 * (p - 0.5), 0.5 + 0.88 * (p - 0.5))
    strike, pct = recommend_strike_level(p3, p)
    print()
    print(f'Test 3 (pure math) OK: p={p:.2f} P(win in 3)={p3:.3f} strike={strike} (conf%={pct:.1f})')

    # Test 4: closed-loop reward
    hip3 = HighIntelligencePredictor()
    history3 = list(synthetic[:300])
    for d in history3[:-1]:
        hip3.add_observation(d)
    pred = hip3.predict()
    actual = history3[-1]
    hip3.reward(actual)
    actual_side = 'Big' if actual >= 5 else 'Small'
    print()
    print(f'Test 4 (reward loop) OK: predicted_side={pred.prediction} actual={actual} ({actual_side})')
    print(f'  calibrator_buffer_size={len(hip3.calibrator.buffer)} ensemble_n={len(hip3.ensemble.weights)}')

    print()
    print('=== ALL HIGH-INTELLIGENCE PREDICTOR TESTS PASSED ===')

if __name__ == '__main__':
    main()
