import sys
sys.path.insert(0, '/Users/jaf/win')
sys.path.insert(0, '/Users/jaf/win/backend')

from backend.intelligence.state_fingerprint import compute_state_fingerprint, StateFingerprint
from backend.intelligence.similar_state import SimilarStateMemory, SimilarStateResult
from backend.intelligence.multi_model import (
    MultiModelEnsemble, FrequencyModel, BayesianModel, MarkovModel,
    NgramModel, RecencyWeightedModel, SimilarStateModel, RandomBaseline, MajorityBaseline
)
from backend.intelligence.meta_learner import MetaLearner, MetaLearnerOutput
from backend.intelligence.adversarial_critic import AdversarialCritic, CriticOutput
from backend.intelligence.three_level import ThreeLevelAnalysis, ThreeLevelProbabilities
from backend.intelligence.online_learning import FastMemory, OnlineUpdater
from backend.intelligence.concept_drift import ConceptDriftDetector, DriftResult
from backend.intelligence.calibration import ConfidenceCalibrator, CalibrationResult
from backend.intelligence.abstention import AbstentionEngine, AbstentionResult
from backend.intelligence.daily_evolution import DailyEvolution, WalkForwardEvaluator, BaselineEvaluator, GenerationRecord
from backend.intelligence.dashboard import IntelligenceDashboard, DashboardData
from backend.intelligence.daily_report import DailyIntelligenceReport, DailyReport
from backend.intelligence.engine import AdaptiveIntelligenceEngine
from backend.intelligence import AdaptiveIntelligenceEngine as AIE2
from backend.intelligence.models import (
    StateFingerprintRecord, SimilarStateRecord, ModelPerformanceRecord,
    CalibrationRecord, DriftRecord, AbstentionRecord, DailyReportRecord,
    ensure_intelligence_tables
)

print('ALL 20+ INTELLIGENCE MODULES IMPORTS: OK')
fp = compute_state_fingerprint([5,3,8,1,4,9,0,2,7,6]*20, 100)
print('Sample fingerprint id:', fp.fingerprint_id())
print('Sample entropy:', fp.entropy)

# Quick test: engine init with synthetic history
import random
random.seed(42)
hist = [random.randint(0,9) for _ in range(500)]
engine = AdaptiveIntelligenceEngine(generation=1)
init = engine.initialize_from_history(hist)
print('Engine init keys:', sorted(init.keys())[:10])

pred = engine.predict(hist[-200:], next_issue='1000', next_sequence_no=500)
print('Prediction keys:', sorted(pred.keys())[:20])
print('Legacy fields present:', all(k in pred for k in [
    'prediction','confidence','targetNum','hedgeNum','nextIssue','action',
    'strikeQuality','modelConsensus','martingaleLevel','driftLevel','patternName',
    'totalSamplesTrained','ensembleWeights','modelPBigVector'
]))
print('New fields present:', all(k in pred for k in [
    'generation','stateFingerprint','stateSimilarity','stateSampleSize','entropy',
    'regime','adversarialScore','contradictionScore','calibratedProbability',
    'calibrationError','oosScore','baselineScore','edgeStatus','learningStatus',
    'modelReliability','knowledgeVersion'
]))
print('action:', pred.get('action'), 'confidence:', pred.get('confidence'), 'edgeStatus:', pred.get('edgeStatus'))
