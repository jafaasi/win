import sys
import os
import time
import json
import logging
import numpy as np
from datetime import datetime

# Load environment variables from .env file BEFORE importing database module
try:
    from dotenv import load_dotenv
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(backend_dir, '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"[ULTRA] Loaded .env from {env_file}")
except Exception as e:
    print(f"[ULTRA] Warning: Could not load .env: {e}")

# Ensure local imports work in all environments
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import (
    SessionLocal, Draw, Outcome, AIBrainState, save_ai_brain_state,
    save_prediction, save_prediction_audit,
)
from backend.evoseq_loop import run_evoseq_cycle
from backend.ultra_intelligence import UltraIntelligenceEngine

# Structured logging
logging.basicConfig(
    format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger("ULTRA")

# ============================================================================
# 🧠 ULTRA MODE: Unified Intelligence with Exploit Gating
# ============================================================================
# Replaces BeastPredictor with UltraIntelligenceEngine which integrates:
#   - ExhaustiveExploitDetector (11 statistical tests)
#   - HighIntelligencePredictor (CTW, BOCPD, Bayesian streak, calibration)
#   - EVOSEQ (Transformer + Mamba + baselines)
#   - Adaptive Hedge ensemble (online regret-minimizing)
#   - Decay-weighted Markov transitions
#   - Session-position temporal features
#   - Honest confidence bands with SKIP signal
#   - Persisted win/loss streak state
# ============================================================================

POLL_INTERVAL = 2  # seconds — faster for 30s game cycle alignment


def run_local_engine():
    logger.info("🚀 🧠 LAUNCHING ULTRA MODE: Unified Intelligence Engine")
    logger.info("Polling Supabase every %ds with exploit-gated predictions", POLL_INTERVAL)
    logger.info("8-model Hedge ensemble + ExhaustiveExploitDetector + honest confidence bands")

    engine = UltraIntelligenceEngine(max_history=50000)

    # Load persisted streak state from database
    try:
        db = SessionLocal()
        engine.load_streak(db)
        db.close()
        logger.info("Streak state loaded from database")
    except Exception as e:
        logger.warning("Could not load streak state: %s", e)

    last_processed_issue = None
    last_rewarded_issue = None  # never double-reward
    cycles_completed = 0

    while True:
        try:
            db = SessionLocal()

            from sqlalchemy import func
            # Check for the latest draw from the cloud scraper using numeric-safe sort
            latest_draw = db.query(Draw).order_by(
                func.length(Draw.issue_number).desc(),
                Draw.issue_number.desc()
            ).first()

            if latest_draw:
                latest_issue = str(latest_draw.issue_number)

                # --- Closed-loop reward: the draw we're seeing NOW resolves ---
                # the PREVIOUS cycle's prediction. Reward all sub-engines.
                if last_rewarded_issue != latest_issue and last_processed_issue is not None:
                    try:
                        actual_digit = int(latest_draw.number) % 10
                        engine.reward_resolved(actual_digit=actual_digit, issue_closed=latest_issue)
                        side = "Big" if actual_digit >= 5 else "Small"
                        logger.info(
                            "🔁 Reward: issue %s = digit %d (%s) | W-streak %d / L-streak %d | Session %.1f%%",
                            latest_issue, actual_digit, side,
                            engine.streak.win_streak, engine.streak.loss_streak,
                            engine.streak.session_win_rate * 100,
                        )
                        last_rewarded_issue = latest_issue

                        # Persist streak state every reward cycle
                        engine.save_streak(db)
                    except Exception as rew_err:
                        logger.warning("Reward step failed: %s", rew_err)

                if latest_issue != last_processed_issue:
                    logger.info(
                        "\nCycle #%d | New Draw: %s (Num: %s)",
                        cycles_completed + 1, latest_issue, latest_draw.number,
                    )
                    last_processed_issue = latest_issue
                    cycles_completed += 1

                    # 1. Extract full deep sequence history from Supabase (up to 50,000 outcomes)
                    outcomes_list = db.query(Outcome).order_by(Outcome.sequence_no.desc()).limit(50000).all()
                    if outcomes_list:
                        history = [int(o.digit) for o in reversed(outcomes_list)]
                    else:
                        db_draws = db.query(Draw).order_by(
                            func.length(Draw.issue_number).desc(),
                            Draw.issue_number.desc()
                        ).limit(50000).all()
                        if db_draws:
                            history = [int(d.number) for d in reversed(db_draws)]
                        else:
                            history = []

                    # 2. EVOSEQ Continuous Adaptive Evolution with Transformer + Mamba
                    if history and len(history) >= 10:
                        logger.info("Running EVOSEQ + Ultra Intelligence on %d samples...", len(history))
                        registry_state = run_evoseq_cycle(history, db)

                        # 3. Ultra Intelligence: exploit-gated, Hedge-ensemble, honest confidence
                        ai_result = engine.predict(history, registry_state, db)

                        # Only update if we have a valid prediction
                        if ai_result and ai_result.get("prediction"):
                            next_issue = str(int(latest_issue) + 1)
                            ai_result["currentIssue"] = latest_issue
                            ai_result["nextIssue"] = next_issue
                            ai_result["latestIssue"] = latest_issue
                            ai_result["generation"] = registry_state.get("generation", 1)
                            ai_result["championGenome"] = registry_state.get("champion_id", "Ultra-Ensemble")
                            ai_result["latentRegime"] = f"🔬 {registry_state.get('drift_level', 'STABLE')}"
                            ai_result["aleatoricEntropy"] = registry_state.get("entropy", 3.2)
                            ai_result["modelDisagreement"] = float(registry_state.get("disagreement_score", 0.03))
                            ai_result["totalSamplesTrained"] = len(history)
                            ai_result["regimeProbabilities"] = {
                                "momentum": 0.35,
                                "alternation": 0.25,
                                "harmonic": 0.20,
                                "equilibrium": 0.20,
                            }
                            ai_result["stochasticPrediction"] = {
                                "nextDigit": ai_result["targetNum"],
                                "nextSide": ai_result["prediction"],
                            }
                            ai_result.setdefault(
                                "martingale3Hint",
                                {
                                    "pWinIn3": ai_result.get("calibratedPWinIn3", 0.875),
                                    "pCorrectSingle": ai_result.get("calibratedPSingle", 0.55),
                                    "strike": ai_result.get("strikeQuality", "CONSERVATIVE"),
                                },
                            )

                            # Record the forecast before the result exists
                            save_prediction(
                                db, next_issue, ai_result["prediction"],
                                ai_result["confidence"], ai_result["patternName"],
                            )
                            save_prediction_audit(
                                db,
                                next_issue,
                                ai_result["championGenome"],
                                ai_result["probability_big"],
                                ai_result["targetNum"],
                                ai_result["entropy"],
                                ai_result["driftLevel"],
                                ai_result["driftScore"],
                                ai_result["nullAdvantage"],
                            )

                            action = ai_result.get("action", "FORECAST")
                            conf = ai_result["confidence"]
                            p3 = ai_result.get("calibratedPWinIn3", None)
                            strike = ai_result.get("strikeQuality", "?")
                            exploit_score = ai_result.get("exploitScore", 0)
                            agree = ai_result.get("enginesAgree", 0)

                            action_emoji = {
                                "SKIP": "⏭️",
                                "CAUTION": "⚠️",
                                "FORECAST": "📊",
                                "STRIKE": "⚡",
                            }.get(action, "📊")

                            logger.info(
                                "%s %s: %s (%.1f%%) for Issue #%s | Strike: %s | P(win3)=%.3f",
                                action_emoji, action, ai_result['prediction'], conf, next_issue,
                                strike, p3 if p3 is not None else 0,
                            )
                            logger.info(
                                "🎯 Target: %d | Hedge: %d | Exploit: %.2f | Agree: %d/4 | IID-reject: %s",
                                ai_result['targetNum'], ai_result['hedgeNum'],
                                exploit_score, agree,
                                'Y' if ai_result.get('rejectIID') else 'N',
                            )
                            logger.info(
                                "📈 Scorecard: %s | Session: %.1f%% | W%d/L%d",
                                ai_result.get('scorecard', {}).get('recent_20', ''),
                                engine.streak.session_win_rate * 100,
                                engine.streak.win_streak,
                                engine.streak.loss_streak,
                            )

                            # 4. Sync AI state directly to Supabase as Live_UI_State
                            # Add timestamp for freshness tracking
                            from datetime import datetime
                            ai_result["predictionCreatedAt"] = datetime.utcnow().isoformat() + "Z"
                            
                            save_ai_brain_state(
                                db=db,
                                model_name="Live_UI_State",
                                generation=ai_result["generation"],
                                total_samples=len(history),
                                weights_json=json.dumps(ai_result),
                                win_rate=registry_state.get("fitness", 50.0),
                            )
                            logger.info(
                                "✅ Live_UI_State saved | Issue: %s | %s %.1f%% | P(win3)=%s",
                                ai_result['currentIssue'], ai_result['prediction'],
                                ai_result['confidence'],
                                ai_result.get('calibratedPWinIn3', 'n/a'),
                            )

                            # 5. Evolution history tracking
                            save_ai_brain_state(
                                db=db,
                                model_name=f"EVOSEQ_Gen_{ai_result['generation']}",
                                generation=ai_result["generation"],
                                total_samples=len(history),
                                weights_json=json.dumps({
                                    "snapshot": ai_result,
                                    "registry": registry_state,
                                }),
                                win_rate=registry_state.get("fitness", 50.0),
                            )
                        else:
                            logger.warning("Ultra Intelligence returned no prediction")
                    else:
                        logger.info("Waiting for sufficient history (need >= 10 samples)...")

            db.close()

        except Exception as e:
            import traceback
            logger.error("Engine Error: %s", e)
            traceback.print_exc()

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    run_local_engine()
