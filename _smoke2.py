import sys
sys.path.insert(0, '/Users/jaf/win')
sys.path.insert(0, '/Users/jaf/win/backend')

import random
random.seed(42)
hist = [random.randint(0,9) for _ in range(600)]

from backend.intelligence.engine import AdaptiveIntelligenceEngine
engine = AdaptiveIntelligenceEngine(generation=1)
init = engine.initialize_from_history(hist)
print('Engine init:', init)

# Predict with the correct signature
pred = engine.predict(hist[-250:], next_issue_number='1000', next_sequence_no=600)
print('\n=== LEGACY FIELDS ===')
legacy = ['prediction','confidence','targetNum','hedgeNum','nextIssue','action',
          'strikeQuality','modelConsensus','martingaleLevel','driftLevel','patternName',
          'totalSamplesTrained','ensembleWeights','modelPBigVector']
for k in legacy:
    v = pred.get(k)
    if isinstance(v, (list, dict)) and len(str(v)) > 120:
        v = f'<{type(v).__name__} len={len(v)}>'
    print(f'  {k}: {v}')
print('\n=== NEW FIELDS ===')
new_fields = ['generation','stateSimilarity','stateSampleSize','entropy',
              'regime','adversarialScore','contradictionScore','calibratedProbability',
              'calibrationError','oosScore','baselineScore','edgeStatus','learningStatus',
              'knowledgeVersion']
for k in new_fields:
    print(f'  {k}: {pred.get(k)}')
print(f'  stateFingerprint keys: {len(pred.get("stateFingerprint", {}))}')
print(f'  modelReliability keys: {list(pred.get("modelReliability", {}).keys())[:5]}')
print('\n=== RESOLVE OUTCOME TEST (fast online learning) ===')
actual_digit = random.randint(0,9)
print(f'  Resolving actual digit = {actual_digit}')
engine.resolve_outcome(actual_digit, hist[-10:], {})
print('  Resolve OK, recent_acc:', engine.fast_memory.recent_accuracy(20))
print('  Total resolved:', engine.fast_memory.total_resolved)

print('\n=== DAILY EVOLUTION (small, synthetic) ===')
import time
t0 = time.time()
gen_rec = engine.run_daily_evolution(hist)
dt = time.time() - t0
for k, v in gen_rec.items():
    if isinstance(v, (dict, list)) and len(str(v)) > 200:
        continue
    print(f'  {k}: {v}')
print(f'  Runtime: {dt:.2f}s')

print('\n=== DASHBOARD & REPORT ===')
dash = engine.build_dashboard()
dash_dict = dash.__dict__ if hasattr(dash, '__dict__') else dash
print(f'  Dashboard keys: {len(dash_dict)}')
report = engine.build_daily_report(historical_samples_total=len(hist), new_samples_today=24)
print('  DAILY REPORT TEXT:\n' + report.render_text())
