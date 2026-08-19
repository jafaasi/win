#!/usr/bin/env python3
"""Prediction intelligence tests with 3-level win metrics."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.prediction_intelligence import EvidencePolicy, evaluate_records


def test1():
    result = evaluate_records([], 0.81)
    assert result["action"] == "FORECAST"
    assert result["confidence"] == 0.5
    print("Test 1 PASSED: gate forecasts without out-of-sample history")


def test2():
    records = [
        {"probability_big": 0.9, "actual_size": "Small" if i % 2 else "Big"}
        for i in range(240)
    ]
    result = evaluate_records(records, 0.9)
    assert result["action"] == "FORECAST"
    assert result["brier_improvement"] < 0
    print(f"Test 2 PASSED: unskilled Brier Δ={result['brier_improvement']:.6f} reason={result['reason']}")
    print(f"  Per-round win rate: {result['per_round_win_rate']:.3f} | 3-level: {result['three_level_win_rate']:.3f}")


def test3():
    records = [
        {
            "probability_big": 0.8 if i % 2 else 0.2,
            "actual_size": (
                "Big" if (i % 2 and i % 5) or (not i % 2 and not i % 5) else "Small"
            ),
        }
        for i in range(250)
    ]
    result = evaluate_records(
        records, 0.8, EvidencePolicy(min_resolved=200, min_local_samples=40)
    )
    assert result["action"] == "FORECAST"
    assert result["validated_edge"] is True, (
        f"reason={result['reason']} per_round_lower={result['accuracy_lower_bound']:.3f} "
        f"3level_lower={result['three_level_lower_bound']:.3f} n3l_windows={result.get('three_level_win_rate', 0)}"
    )
    assert result["confidence"] < 1.0
    print(f"Test 3 PASSED: skillful validated_edge={result['validated_edge']} reason={result['reason']}")
    print(f"  calibrated conf p={result['confidence']:.3f} | joint3 p={result['joint3_probability']:.3f}")
    print(f"  per_round win={result['per_round_win_rate']:.3f} (lower {result['accuracy_lower_bound']:.3f})")
    print(f"  3-level win={result['three_level_win_rate']:.3f} (lower {result['three_level_lower_bound']:.3f})")


if __name__ == "__main__":
    test1()
    test2()
    test3()
    print()
    print("ALL PREDICTION INTELLIGENCE TESTS PASSED")
