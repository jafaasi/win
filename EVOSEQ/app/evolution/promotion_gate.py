from typing import Dict, Any

def promotion_gate(
    challenger_metrics: Dict[str, Any],
    champion_metrics: Dict[str, Any],
    min_robustness: float = 0.0
) -> bool:
    """
    Multi-Criteria Promotion Gate:
    Prevents self-reinforcing evolutionary overfitting by strictly gating promotions:
    1. Challenger temporal out-of-sample score > Champion temporal score
    2. Challenger calibration error <= Champion calibration error * 1.5
    3. Challenger null advantage > 0
    4. Challenger robustness score >= min_robustness
    """
    c_score = challenger_metrics.get("temporal_score", challenger_metrics.get("score", -float("inf")))
    champ_score = champion_metrics.get("temporal_score", champion_metrics.get("score", -float("inf")))
    
    if c_score <= champ_score:
        return False
        
    c_calib = challenger_metrics.get("calibration_error", 0.05)
    champ_calib = champion_metrics.get("calibration_error", 0.05)
    if c_calib > (champ_calib * 1.5 + 1e-4):
        return False
        
    null_adv = challenger_metrics.get("null_advantage", 0.0)
    if null_adv <= 0:
        return False
        
    robustness = challenger_metrics.get("robustness_score", 0.1)
    if robustness < min_robustness:
        return False
        
    return True
