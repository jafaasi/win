from typing import List, Dict, Any
from ..database import SessionLocal
from ..schemas import MetaExperimentRecord, ResearchHypothesisRecord

class ModelKnowledgeGraph:
    """
    Model Knowledge Graph:
    Allows querying which model families have historically been robust under specific environments
    and which features reliably survive out-of-sample audits.
    """

    def query_robust_families_under_drift(self, drift_threshold: float = 0.05) -> Dict[str, float]:
        """Returns average performance of model families under high drift regimes."""
        with SessionLocal() as session:
            records = session.query(MetaExperimentRecord).all()
            family_scores: Dict[str, List[float]] = {}
            for r in records:
                env = r.environment
                if env.get("drift_score", 0.0) >= drift_threshold:
                    fam = r.model_descriptor.get("family", "Unknown")
                    score = (r.null_advantage or 0.0) - ((r.brier_score or 0.09) * 5.0)
                    if fam not in family_scores:
                        family_scores[fam] = []
                    family_scores[fam].append(score)
                    
            return {fam: round(float(sum(sc) / len(sc)), 4) for fam, sc in family_scores.items() if sc}

    def get_supported_scientific_hypotheses(self) -> List[Dict[str, Any]]:
        """Returns all hypotheses scientifically supported by evidence."""
        with SessionLocal() as session:
            supported = session.query(ResearchHypothesisRecord).filter(
                ResearchHypothesisRecord.status.in_(["SUPPORTED", "WEAK_EVIDENCE"])
            ).all()
            return [
                {
                    "id": h.id,
                    "title": h.title,
                    "status": h.status,
                    "evidence_score": h.evidence_score,
                    "evidence_summary": h.evidence_summary
                }
                for h in supported
            ]
