from typing import List, Dict, Any, Optional
from ..database import SessionLocal
from ..schemas import ResearchQuestionRecord, ResearchHypothesisRecord

class ResearchQuestionManager:
    """
    Manages the formal hypothesis lifecycle:
    Hypothesis -> Pre-registration -> Out-of-sample execution -> Null Comparison -> Evidence Status.
    """

    DEFAULT_QUESTIONS = [
        ("RQ-001", "Does additional context beyond 128 observations improve sequence generalization?", "Long-range context contains residual predictability."),
        ("RQ-002", "Does State Space (S4 / Mamba) recurrence outperform attention under regime shifts?", "Continuous state-space compression is more robust to drift than fixed attention."),
        ("RQ-003", "Does information-theoretic feature compression improve candidate convergence?", "Conditional entropy and mutual information features accelerate learning."),
        ("RQ-004", "Does ESN reservoir size scale monotonically with prediction accuracy?", "Reservoirs experience diminishing returns beyond size 128.")
    ]

    def __init__(self):
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        with SessionLocal() as session:
            for code, q, h in self.DEFAULT_QUESTIONS:
                existing = session.query(ResearchQuestionRecord).filter(ResearchQuestionRecord.question_code == code).first()
                if not existing:
                    rec = ResearchQuestionRecord(question_code=code, question=q, hypothesis=h, status="OPEN", priority=0.8)
                    session.add(rec)
                    session.commit()

    def register_hypothesis(self, question_code: str, title: str, description: str) -> int:
        with SessionLocal() as session:
            q = session.query(ResearchQuestionRecord).filter(ResearchQuestionRecord.question_code == question_code).first()
            hypo = ResearchHypothesisRecord(
                question_id=q.id if q else None,
                title=title,
                description=description,
                status="PRE_REGISTERED",
                evidence_score=0.0
            )
            session.add(hypo)
            session.commit()
            session.refresh(hypo)
            return hypo.id

    def update_evidence(self, hypothesis_id: int, p_value: float, delta_null: float, summary: Dict[str, Any]) -> str:
        with SessionLocal() as session:
            hypo = session.query(ResearchHypothesisRecord).filter(ResearchHypothesisRecord.id == hypothesis_id).first()
            if not hypo:
                return "NOT_FOUND"
                
            hypo.evidence_summary = summary
            if p_value < 0.01 and delta_null > 0.05:
                hypo.status = "SUPPORTED"
                hypo.evidence_score = 1.0
            elif p_value < 0.05 and delta_null > 0.0:
                hypo.status = "WEAK_EVIDENCE"
                hypo.evidence_score = 0.6
            elif p_value >= 0.20:
                hypo.status = "REJECTED"
                hypo.evidence_score = -1.0
            else:
                hypo.status = "INCONCLUSIVE"
                hypo.evidence_score = 0.0
                
            session.commit()
            return hypo.status

    def get_agenda(self) -> List[Dict[str, Any]]:
        with SessionLocal() as session:
            questions = session.query(ResearchQuestionRecord).all()
            agenda = []
            for q in questions:
                hypos = session.query(ResearchHypothesisRecord).filter(ResearchHypothesisRecord.question_id == q.id).all()
                agenda.append({
                    "code": q.question_code,
                    "question": q.question,
                    "status": q.status,
                    "hypotheses": [{"id": h.id, "title": h.title, "status": h.status, "evidence_score": h.evidence_score} for h in hypos]
                })
            return agenda
