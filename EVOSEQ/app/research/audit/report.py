from typing import Dict, Any

def generate_ascii_audit_hud(report_data: Dict[str, Any]) -> str:
    """Renders the ASCII Research & Audit Matrix HUD."""
    gen = report_data.get("generation", 1)
    obs = report_data.get("observations", 0)
    champ = report_data.get("champion", "None")
    recent_p = float(report_data.get("recent_performance", 0.0))
    hist_p = float(report_data.get("historical_performance", 0.0))
    delta_null = float(report_data.get("delta_vs_null", 0.0))
    drift = report_data.get("drift", "LOW")
    calib = report_data.get("calibration", "GOOD")
    disagree = report_data.get("disagreement", "LOW")
    null_exp = report_data.get("null_experiments", 0)
    cands = report_data.get("candidate_models", 0)
    retired = report_data.get("retired_models", 0)
    robust = report_data.get("robustness", "HIGH")

    recent_p_str = f"{recent_p:.4f}"
    hist_p_str = f"{hist_p:.4f}"
    delta_null_str = f"{delta_null:+.4f}"

    hud = f"""
╔══════════════════════════════════════════════════════════╗
║               🧠 EVOSEQ RESEARCH AUDIT                   ║
╠══════════════════════════════════════════════════════════╣
║ Generation:             {gen:<32} ║
║ Observations:           {obs:<32} ║
║ Champion:               {champ:<32} ║
║                                                          ║
║ Recent Performance:     {recent_p_str:<32} ║
║ Historical Performance: {hist_p_str:<32} ║
║ Δ vs Null Hypothesis:   {delta_null_str:<32} ║
║                                                          ║
║ Drift State:            {drift:<32} ║
║ Probability Calibration:{calib:<32} ║
║ Model Disagreement:     {disagree:<32} ║
║                                                          ║
║ Null Experiments Run:   {null_exp:<32} ║
║ Candidate Models:       {cands:<32} ║
║ Retired Models:         {retired:<32} ║
║                                                          ║
║ Generalization Robust:  {robust:<32} ║
╚══════════════════════════════════════════════════════════╝
"""
    return hud
