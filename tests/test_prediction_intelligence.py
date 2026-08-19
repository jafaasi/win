from backend.prediction_intelligence import EvidencePolicy, evaluate_records


def test_gate_forecasts_without_out_of_sample_history():
    result = evaluate_records([], 0.81)
    assert result["action"] == "FORECAST"
    assert result["confidence"] == 0.5


def test_gate_keeps_forecasting_for_an_unskilled_model():
    records = [
        {"probability_big": 0.9, "actual_size": "Small" if i % 2 else "Big"}
        for i in range(240)
    ]
    result = evaluate_records(records, 0.9)
    assert result["action"] == "FORECAST"
    assert result["brier_improvement"] < 0


def test_gate_accepts_only_a_large_consistently_skillful_history():
    records = [
        {
            "probability_big": 0.8 if i % 2 else 0.2,
            "actual_size": (
                "Big" if (i % 2 and i % 5) or (not i % 2 and not i % 5) else "Small"
            ),
        }
        for i in range(250)
    ]
    result = evaluate_records(records, 0.8, EvidencePolicy(min_resolved=200, min_local_samples=40))
    assert result["action"] == "FORECAST"
    assert result["validated_edge"] is True
    assert result["confidence"] < 1.0
