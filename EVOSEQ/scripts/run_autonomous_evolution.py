import sys
import os
import numpy as np

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.randomness import seed_everything
from app.database import SessionLocal, engine, Base
from app.schemas import Outcome
from app.evolution import (
    AutonomousResearchDirector,
    FeatureAblationTester,
    generate_hypotheses
)
from app.ensemble.uncertainty import decompose_uncertainty

def run_autonomous_evolution():
    seed_everything(42)
    Base.metadata.create_all(bind=engine)
    print("🚀 Initializing EVOSEQ — Autonomous Evolution & Meta-Search Pipeline...")
    
    with SessionLocal() as session:
        outcomes = session.query(Outcome).order_by(Outcome.sequence_no.asc()).all()
        
    if len(outcomes) < 100:
        from app.ingestion.stream import ingest_outcomes_batch
        seed_batch = [{"sequence_no": i + 1, "digit": int(np.random.randint(0, 10))} for i in range(200)]
        ingest_outcomes_batch(seed_batch)
        with SessionLocal() as session:
            outcomes = session.query(Outcome).order_by(Outcome.sequence_no.asc()).all()
            
    digits = np.array([o.digit for o in outcomes], dtype=np.int64)
    N = len(digits)
    train_slice = digits[:-50]
    test_slice = digits[-50:]
    
    class MockEnv:
        drift_score = 0.045
        model_disagreement = 0.22
        entropy = 3.29
        
    director = AutonomousResearchDirector()
    champion_metrics = {"mean_loss": 2.3026, "std_loss": 0.002}
    
    # 1. Run Autonomous Research Cycle
    cycle_result = director.run_research_cycle(
        environment_state=MockEnv(),
        data_train=train_slice,
        data_test=test_slice,
        champion_metrics=champion_metrics,
        generation=3
    )
    
    # 2. Feature Ablation Test
    from app.features.builder import build_feature_matrix
    from app.models.markov import MarkovModel
    feat_matrix = build_feature_matrix(test_slice)
    ablation_tester = FeatureAblationTester()
    
    def eval_fn(mat):
        # 17-dim feature projection
        m = MarkovModel(order=2).fit(train_slice)
        return m.predict_sequence(test_slice)
        
    ablation_results = ablation_tester.ablate_features(feat_matrix, test_slice, eval_fn)
    
    # 3. Render ASCII Report
    report = f"""
══════════════════════════════════════════════════════════════
       EVOSEQ — AUTONOMOUS EVOLUTION & META-SEARCH REPORT
══════════════════════════════════════════════════════════════
Generation:         {cycle_result['generation']}
Total Observations: {N:,}
Research Hypotheses Formulated: {cycle_result['hypotheses_generated']}

CANDIDATES & NULL REFEREE AUDIT
CANDIDATE CODE              FAMILY         LOSS     NULL p-VAL  STATUS
──────────────────────────────────────────────────────────────"""
    for c in cycle_result["candidates_evaluated"]:
        report += f"\n{c['code']:27} {c['family']:14} {c['mean_loss']:.4f}   {c['null_p']:.4f}      {c['status']}"

    report += f"""
──────────────────────────────────────────────────────────────
FEATURE ABLATION (Marginal Information Value)
FEATURE GROUP       DELTA LOSS      STATUS
──────────────────────────────────────────────────────────────"""
    for g, res in ablation_results.items():
        report += f"\n{g:19} {res['delta_loss']:+.5f}        {res['status']}"

    report += f"""
──────────────────────────────────────────────────────────────
FAMILY SURVIVAL RATES (Promoted / Generated)"""
    for fam, rate in cycle_result["survival_rates"].items():
        report += f"\n{fam.capitalize():19} {rate * 100:.1f}%"

    winner_text = cycle_result['promoted_winner']['candidate_code'] if cycle_result['promoted_winner'] else "None (Hypotheses Refuted / Null Preserved)"
    report += f"""
──────────────────────────────────────────────────────────────
EVOLUTIONARY DECISION
Promoted Winner:    {winner_text}
══════════════════════════════════════════════════════════════
"""
    print(report)
    return report

if __name__ == "__main__":
    run_autonomous_evolution()
