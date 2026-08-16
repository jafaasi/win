import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.schemas import Outcome
from app.models.markov import MarkovModel
from app.research.null_models.surrogate import NullHypothesisReferee

def run_research():
    print("🔬 Running EVOSEQ Research Null-Hypothesis Audit...")
    with SessionLocal() as session:
        outcomes = session.query(Outcome).order_by(Outcome.sequence_no.asc()).all()
        digits = [o.digit for o in outcomes]
        
    if len(digits) < 40:
        print("⚠️ Insufficient observations for statistical audit.")
        return
        
    model = MarkovModel(order=2, smoothing=0.5)
    model.fit(digits[:len(digits)//2])
    
    referee = NullHypothesisReferee(num_surrogates=15, alpha=0.05)
    audit = referee.audit_model(model, digits, metric="brier")
    
    print("📋 NULL HYPOTHESIS AUDIT RESULTS:")
    print(f"  - Observed Brier Score:       {audit.observed_metric:.4f}")
    print(f"  - IID Null Brier Score:        {audit.null_metrics['iid_mean']:.4f} (p-val: {audit.p_values['p_iid']:.4f})")
    print(f"  - Markov Null Brier Score:     {audit.null_metrics['markov_mean']:.4f} (p-val: {audit.p_values['p_markov']:.4f})")
    print(f"  - Surrogate Null Brier Score:  {audit.null_metrics['surrogate_mean']:.4f} (p-val: {audit.p_values['p_surrogate']:.4f})")
    print(f"  - FDR Corrected Significance:  {audit.fdr_significant}")
    print(f"  - Null Advantage:              {audit.null_advantage:.4f}")
    print(f"  - Decision Status:             {audit.decision_status}")
    print("✅ Research Audit Complete.")

if __name__ == "__main__":
    run_research()
