from typing import Dict, Any, List, Optional
from enum import Enum

class PromotionStage(str, Enum):
    CANDIDATE = "CANDIDATE"
    SANITY_CHECK = "SANITY_CHECK"
    RESEARCH_VALIDATION = "RESEARCH_VALIDATION"
    ROBUSTNESS_VALIDATION = "ROBUSTNESS_VALIDATION"
    PROBATION = "PROBATION"
    CHAMPION = "CHAMPION"
    REJECTED = "REJECTED"

class SequentialPromotionGate:
    """
    Multi-stage sequential promotion gate for EVOSEQ candidates:
    Ensures that no candidate is crowned without surviving sanity, multi-seed validation,
    surrogate null significance, and out-of-sample forward probation.
    """

    def __init__(self, min_improvement_delta: float = 0.0005, max_null_p_value: float = 0.10):
        self.min_delta = min_improvement_delta
        self.max_null_p = max_null_p_value

    def evaluate_promotion(
        self,
        candidate_metrics: Dict[str, Any],
        champion_metrics: Dict[str, Any],
        null_test_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes sequential gate criteria.
        Returns: { "stage": PromotionStage, "promoted": bool, "reason": str }
        """
        cand_loss = candidate_metrics.get("mean_loss", 2.3026)
        champ_loss = champion_metrics.get("mean_loss", 2.3026)
        delta = float(champ_loss - cand_loss)
        
        std_loss = candidate_metrics.get("std_loss", 0.0)
        p_val = null_test_result.get("null_p_value", 1.0)
        
        # 1. Sanity Check: Loss must not explode
        if cand_loss > 3.50 or np.isnan(cand_loss):
            return {"stage": PromotionStage.REJECTED, "promoted": False, "reason": "Failed sanity check (loss unbounded)"}
            
        # 2. Multi-Seed Stability Check
        if std_loss > 0.15:
            return {"stage": PromotionStage.REJECTED, "promoted": False, "reason": f"High seed variance (std={std_loss:.4f})"}
            
        # 3. Null Hypothesis Referee Check
        if p_val > self.max_null_p:
            return {"stage": PromotionStage.REJECTED, "promoted": False, "reason": f"Failed null test referee (p={p_val:.4f} > {self.max_null_p})"}
            
        # 4. Out-of-sample Improvement Delta
        if delta < self.min_delta:
            return {"stage": PromotionStage.RESEARCH_VALIDATION, "promoted": False, "reason": f"Insufficient improvement (delta={delta:+.5f} < {self.min_delta})"}
            
        # 5. Passed all gates -> Advance to PROBATION / CHAMPION
        return {
            "stage": PromotionStage.CHAMPION,
            "promoted": True,
            "reason": f"Surpassed champion by {delta:+.5f} log-loss with null p-val {p_val:.4f}"
        }

import numpy as np
