from typing import Optional
from .base import BaseWorker
from ..database import SessionLocal
from ..schemas import SystemEventRecord, ModelVersionRecord, Outcome, ResearchExperimentRecord
from ..research.statistics.confidence import compare_model_to_null
from ..research.null_models.iid import generate_iid
from ..features.basic import digit_distribution
from ..models.markov import MarkovModel

class ResearchWorker(BaseWorker):
    """
    Research Worker: Executes deep null hypothesis comparisons and records long-term experiment logs.
    """

    def __init__(self, poll_interval: float = 5.0):
        super().__init__(worker_name="research_worker", poll_interval=poll_interval)

    def process_cycle(self) -> int:
        state = self.get_state()
        last_seq = state.last_processed_sequence or 0
        
        with SessionLocal() as session:
            events = session.query(SystemEventRecord).filter(
                SystemEventRecord.event_type == "EVOLUTION_DECISION",
                SystemEventRecord.sequence_no > last_seq
            ).order_by(SystemEventRecord.sequence_no.asc()).limit(5).all()
            
            if not events:
                self.update_state(last_seq, status="idle")
                return 0
                
            for evt in events:
                seq_no = evt.sequence_no
                champ = session.query(ModelVersionRecord).filter(ModelVersionRecord.status == "champion").order_by(ModelVersionRecord.id.desc()).first()
                if champ:
                    rows = session.query(Outcome).filter(Outcome.sequence_no <= seq_no).order_by(Outcome.sequence_no.desc()).limit(100).all()
                    rows.reverse()
                    digits = [r.digit for r in rows]
                    
                    if len(digits) >= 40:
                        mkv = MarkovModel(order=champ.parameters.get("order", 2))
                        null_probs = digit_distribution(digits)
                        null_res = compare_model_to_null(
                            mkv,
                            digits,
                            null_generator_fn=lambda l, s: generate_iid(null_probs, length=l, seed=s),
                            repetitions=10,
                            initial_train_size=20
                        )
                        
                        exp = ResearchExperimentRecord(
                            experiment_type="PERIODIC_RESEARCH_AUDIT",
                            model_version_id=champ.id,
                            null_model="iid_empirical_marginal",
                            sample_size=len(digits),
                            observed_score=null_res["observed_score"],
                            null_mean=null_res["null_mean"],
                            null_std=null_res["null_std"],
                            p_value=null_res["p_value"],
                            correction_method="benjamini_hochberg",
                            test_range_start=rows[0].sequence_no,
                            test_range_end=rows[-1].sequence_no,
                            metadata_json={"periodic": True}
                        )
                        session.add(exp)
                        
                self.publish_event("RESEARCH_AUDITED", seq_no, {"audited": True})
                last_seq = seq_no
                
            session.commit()
            self.update_state(last_seq, status="running")
            return len(events)

if __name__ == "__main__":
    ResearchWorker().run()
