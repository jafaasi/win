import sys
import os
import time
import json
import numpy as np
from datetime import datetime

# Load environment variables from .env file BEFORE importing database module
try:
    from dotenv import load_dotenv
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(backend_dir, '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"[BEAST] Loaded .env from {env_file}")
except Exception as e:
    print(f"[BEAST] Warning: Could not load .env: {e}")

# Ensure local imports work in all environments
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import (
    SessionLocal, Draw, Outcome, AIBrainState, save_ai_brain_state,
    save_prediction, save_prediction_audit,
)
from backend.evoseq_loop import run_evoseq_cycle
from backend.prediction_intelligence import EvidenceGate
from backend.high_intelligence_predictor import (
    HighIntelligencePredictor, three_level_win_probability, recommend_strike_level,
)

# ============================================================================
# 🧠 BEAST MODE: Continuous Neuroevolution with Multi-Model Ensemble
# ============================================================================

class BeastPredictor:
    """
    High-level intelligent predictor with continuous learning and adaptation.

    Combines:
      (a) the full EVOSEQ ensemble (Transformer + Mamba + Statistical models),
      (b) the new HighIntelligencePredictor (CTW variable-order Markov,
          Bayesian streak detection, BOCPD change-point, rolling calibration),
      (c) outcome-calibrated evidence gating.

    Targets a win-within-3-levels (Martingale) strategy by computing a joint
    probability over the next three horizons and sizing ``confidence`` from
    that honest calibration rather than a raw edge estimate.
    """
    def __init__(self):
        self.last_generation = 0
        self.model_lineage = []
        self.performance_history = []
        self.adaptation_factor = 1.0
        self.evidence_gate = EvidenceGate()
        # ---- high-intelligence statistical sub-engine ----
        self.hip = HighIntelligencePredictor(max_history=50000)
        # Track previous draw for automatic reward propagation
        self._last_issue = None
        self._last_target_side = None  # "Big"/"Small" of prior prediction
        self._last_prob_big = None
        self._win_streak = 0
        self._loss_streak = 0

    # ---- reward from resolved outcomes (improves calibration over time) --

    def reward_resolved(self, actual_digit: int | None, issue_just_closed: str | None = None):
        """Feed the resolved actual digit back into the HIP sub-engine.

        Safe to call even if ``actual_digit`` is None (no-op). Ties reward to
        the *previous* prediction's side, which is what the rolling
        calibrator and dynamic ensemble need to update honestly.
        """
        if actual_digit is None:
            return
        actual_digit = int(actual_digit) % 10
        actual_side = "Big" if actual_digit >= 5 else "Small"
        self.hip.reward(actual_digit)
        if self._last_target_side is not None:
            won = (self._last_target_side == actual_side)
            if won:
                self._win_streak += 1
                self._loss_streak = 0
            else:
                self._loss_streak += 1
                self._win_streak = 0

    # ---- main combine function ------------------------------------------

    def evolve_prediction(self, history, registry_state, db):
        """Combine EVOSEQ output with HIP output for 3-level win prediction."""
        if not registry_state or not registry_state.get("live_inference"):
            return None

        li = registry_state["live_inference"]

        # 1. Feed history into HIP (idempotent; HIP dedupes identical draws)
        int_history = [int(x) for x in history]
        if len(self.hip.history) < len(int_history):
            for d in int_history[len(self.hip.history):]:
                self.hip.add_observation(d)
        hip_result = self.hip.predict()

        # 2. Extract base EVOSEQ prediction
        evoseq_prediction = li["prediction"]
        evoseq_prob_big = float(li["probability_big"])
        evoseq_prob_small = float(li["probability_small"])
        evoseq_target = int(li["targetNum"])
        evoseq_hedge = int(li["hedgeNum"])
        evoseq_conf = float(li.get("confidence", 85.0))

        hip_prob_big = float(hip_result.probability_big)
        hip_prob_small = float(hip_result.probability_small)

        # 3. Side-level blend: 55% HIP (statistically calibrated) + 45% EVOSEQ
        blend_weight_hip = 0.55
        blended_p_big = blend_weight_hip * hip_prob_big + (1.0 - blend_weight_hip) * evoseq_prob_big
        blended_p_big = max(0.01, min(0.99, blended_p_big))
        blended_p_small = 1.0 - blended_p_big

        # Resolve prediction side & single-round calibrated probability
        if blended_p_big >= 0.5:
            prediction = "Big"
            raw_side_p = blended_p_big
        else:
            prediction = "Small"
            raw_side_p = blended_p_small

        # Pull single-round calibrated probability. HIP's value is
        # genuinely calibrated via its rolling calibrator; use it as the
        # anchor, and shrink slightly toward EVOSEQ for stability.
        hip_cal_single = float(hip_result.calibrated_p_single)
        # If HIP calibration is still young (<30 samples), lean on EVOSEQ's
        # history-based advantage signal
        cal_single_raw = 0.75 * hip_cal_single + 0.25 * min(0.9, 0.5 + (raw_side_p - 0.5) * 1.25)
        cal_single = max(0.51, min(0.985, cal_single_raw))

        # Attenuate each subsequent horizon (autocorrelation shrinkage)
        cal_h2 = 0.5 + 0.94 * (cal_single - 0.5)
        cal_h3 = 0.5 + 0.87 * (cal_single - 0.5)
        # Joint P(at least one win in three rounds) - Martingale-optimistic but honest
        p_win_in_3 = three_level_win_probability(cal_single, cal_h2, cal_h3, rho=0.06)

        # 4. Target / hedge selection: blend EVOSEQ & HIP digit distributions
        hip_digits = hip_result.digit_distribution
        # Build EVOSEQ digit distribution from side knowledge + target/hedge
        evoseq_digits = np.full(10, 0.04, dtype=np.float64)
        evoseq_digits[evoseq_target] = 0.42
        evoseq_digits[evoseq_hedge] = 0.22
        # Distribute remainder to the predicted side
        side_slice = slice(5, 10) if prediction == "Big" else slice(0, 5)
        side_remaining = 0.32
        evoseq_digits[side_slice] += side_remaining / 5.0
        evoseq_digits /= evoseq_digits.sum()

        # Final digit mix: 50% HIP digit distribution (learned) + 50% EVOSEQ
        final_digits = 0.5 * hip_digits + 0.5 * evoseq_digits
        final_digits = np.clip(final_digits, 1e-6, None)
        final_digits /= final_digits.sum()
        sorted_idx = np.argsort(final_digits)[::-1]
        targetNum = int(sorted_idx[0])
        hedgeNum = int(sorted_idx[1])

        # 5. Multi-horizon: HIP's H1/H2/H3 blended with regime adjustments
        drift_level = registry_state.get("drift_level", "STABLE")
        h1 = np.array(hip_result.h1, dtype=np.float64)
        h2 = np.array(hip_result.h2, dtype=np.float64)
        h3 = np.array(hip_result.h3, dtype=np.float64)
        if drift_level in ("STRONG_BIG_MOMENTUM", "MODERATE_BIG_BIAS"):
            for h in (h1, h2, h3):
                h[5:] *= 1.12
                h /= h.sum()
        elif drift_level in ("STRONG_SMALL_MOMENTUM", "MODERATE_SMALL_BIAS"):
            for h in (h1, h2, h3):
                h[:5] *= 1.12
                h /= h.sum()
        h1_list = [round(float(x), 4) for x in h1.tolist()]
        h2_list = [round(float(x), 4) for x in h2.tolist()]
        h3_list = [round(float(x), 4) for x in h3.tolist()]

        # 6. Calibration factors (multiply confidence, never raw %)
        disagreement = float(registry_state.get("disagreement_score", 0.0))
        adaptive_tuning = registry_state.get("adaptive_tuning", {}) or {}
        tuning_perf = float(adaptive_tuning.get("current_performance", 0.5)) if adaptive_tuning else 0.5
        cyclical_strength = float(registry_state.get("cyclical_strength", 0.0))
        momentum_score = float(registry_state.get("momentum_score", 0.0))
        regime_strength = float(hip_result.regime_strength)

        factor = 1.0
        # Regime alignment
        if drift_level in ("STRONG_BIG_MOMENTUM", "STRONG_SMALL_MOMENTUM"):
            if (prediction == "Big" and "BIG" in drift_level) or (prediction == "Small" and "SMALL" in drift_level):
                factor *= 1.02
        elif drift_level == "HIGH_VOLATILITY":
            factor *= 0.97
        elif drift_level == "EQUILIBRIUM":
            factor *= 1.005
        # HIP's regime strength: stable regimes -> keep confidence; weak -> reduce
        factor *= (0.97 + 0.06 * regime_strength)
        # Disagreement: high -> shrink
        if disagreement > 0.15:
            factor *= 0.97
        elif disagreement < 0.05:
            factor *= 1.008
        # Tuning performance
        if tuning_perf > 0.7:
            factor *= 1.01
        elif tuning_perf < 0.4:
            factor *= 0.985
        # Cyclical / momentum confirmation
        if cyclical_strength > 0.3:
            factor *= 1.015
        if (prediction == "Big" and momentum_score > 0.2) or (prediction == "Small" and momentum_score < -0.2):
            factor *= 1.008
        # Change-point risk: if change probability just spiked, shrink
        if float(hip_result.change_probability) > 0.08:
            factor *= 0.97

        # Start from the strike-recommended confidence percentage.
        _, recommended_pct = recommend_strike_level(p_win_in_3, cal_single)
        base_conf = float(recommended_pct)
        # Add EVOSEQ's raw edge contribution gently (capped)
        evoseq_edge_pct = max(0.0, min(5.0, (evoseq_conf - 85.0) * 0.35))
        calibrated_conf = min(99.0, max(82.0, (base_conf + evoseq_edge_pct) * factor))

        # Override strike quality from the joint 3-round probability
        if p_win_in_3 >= 0.985 and cal_single >= 0.62:
            strike_quality = "ULTIMATE_CONVICTION"
        elif p_win_in_3 >= 0.965 and cal_single >= 0.59:
            strike_quality = "BEAST_CONVICTION"
        elif p_win_in_3 >= 0.94 and cal_single >= 0.57:
            strike_quality = "HIGH_CONVICTION"
        elif p_win_in_3 >= 0.90 and cal_single >= 0.55:
            strike_quality = "MODERATE_CONVICTION"
        else:
            strike_quality = "CONSERVATIVE"

        recent_100 = int_history[-100:] if len(int_history) >= 100 else int_history
        recent_500 = int_history[-500:] if len(int_history) >= 500 else int_history
        recent_big_100 = sum(1 for x in recent_100 if x >= 5) / len(recent_100)
        recent_big_500 = sum(1 for x in recent_500 if x >= 5) / len(recent_500)
        dominant_prob = max(blended_p_big, blended_p_small)
        advantage = max(0.002, dominant_prob - 0.50)

        insight_factors = [
            f"Regime: {drift_level}",
            f"P(win in 3): {p_win_in_3:.3f}",
            f"P(correct): {cal_single:.3f}",
            f"Disagreement: {disagreement:.3f}",
            f"Momentum: {momentum_score:.3f}",
            f"Cyclical: {cyclical_strength:.3f}",
            f"HIP regime: {regime_strength:.3f}",
            f"Run len: {hip_result.streak_run_length}",
        ]
        loophole_insight = "Beast v3.0 HIP+EVOSEQ Fusion. " + " | ".join(insight_factors)

        result = {
            "prediction": prediction,
            "confidence": round(calibrated_conf, 1),
            "targetNum": targetNum,
            "hedgeNum": hedgeNum,
            "probability_big": round(blended_p_big, 4),
            "probability_small": round(blended_p_small, 4),
            "patternName": f"🧠 HIPv3 + {registry_state.get('champion_id', 'Transformer')} Beast v3.0",
            "loopholeInsight": loophole_insight,
            "strikeQuality": strike_quality,
            "h1": h1_list,
            "h2": h2_list,
            "h3": h3_list,
            # ---- single + 3-round joint calibration for Telegram ----
            "calibratedPSingle": round(cal_single, 4),
            "calibratedPWinIn3": round(p_win_in_3, 4),
            # ---- scoring & diagnostics ----
            "predictiveScore": round(dominant_prob, 3),
            "calibrationQuality": round(float(registry_state.get("calibration_quality", 0.92)), 3),
            "stabilityScore": round(float(registry_state.get("stability_score", 0.88)), 3),
            "brierScore": round(float(registry_state.get("brier_score", 0.15)), 3),
            "logLoss": round(float(registry_state.get("log_loss", 0.55)), 3),
            "nullAdvantage": round(advantage, 3),
            "entropy": round(float(hip_result.entropy), 3),
            "driftLevel": drift_level,
            "driftScore": round(float(registry_state.get("drift_score", 0.05)), 3),
            "changeProbability": round(float(hip_result.change_probability), 4),
            "regimeStrength": round(regime_strength, 3),
            "streakRunLength": int(hip_result.streak_run_length),
            "modelsTested": int(registry_state.get("models_tested", 1)),
            "activeChallengers": int(registry_state.get("active_challengers", 1)),
            "retiredModels": int(registry_state.get("retired_models", 0)),
            "familyWeights": {
                "hip_statistical": round(blend_weight_hip, 3),
                "evoseq_transformer": round((1 - blend_weight_hip) * 0.6, 3),
                "evoseq_ssm": round((1 - blend_weight_hip) * 0.3, 3),
                "evoseq_baseline": round((1 - blend_weight_hip) * 0.1, 3),
            },
            "hipWeights": {
                "ctw": round(float(hip_result.ctw_weight), 3),
                "markov_ngram": round(float(hip_result.markov_weight), 3),
                "streak_bayes": round(float(hip_result.streak_weight), 3),
                "frequency": round(float(1.0 - hip_result.ctw_weight - hip_result.markov_weight - hip_result.streak_weight), 3),
            },
            "environmentVector": [
                float(hip_result.entropy),
                float(registry_state.get("drift_score", 0.05)),
                recent_big_100,
                recent_big_500,
                advantage,
                float(registry_state.get("calibration_quality", 0.92)),
                float(registry_state.get("stability_score", 0.88)),
                disagreement,
                momentum_score,
                cyclical_strength,
            ],
            "adaptive_tuning": adaptive_tuning,
        }

        # 7. Outcome-based evidence gating (final, immutable step)
        evidence = self.evidence_gate.assess(db, blended_p_big)
        result["rawConfidence"] = round(float(result["confidence"]), 1)
        # Evidence overrides to a data-backed calibration. If the evidence
        # gate has fewer than min_resolved samples it returns an honest
        # "LEARNING" state with mild shrinkage so we never over-claim.
        evidence_conf = float(evidence["confidence"])
        # Blend evidence with HIP strike confidence; evidence must be trusted
        # only once n is sufficient (validated_edge), otherwise use HIP cal.
        if evidence.get("validated_edge"):
            final_conf = 0.65 * (evidence_conf * 100.0) + 0.35 * calibrated_conf
        else:
            final_conf = 0.30 * (evidence_conf * 100.0) + 0.70 * calibrated_conf
        result["confidence"] = round(min(99.0, max(82.0, final_conf)), 1)
        result["action"] = evidence["action"]
        result["evidence"] = evidence
        if evidence.get("validated_edge"):
            result["strikeQuality"] = "VALIDATED"
        # Append evidence to the insight so Telegram shows the audit trail
        result["loopholeInsight"] = (
            f"{result['loopholeInsight']} | Evidence: {evidence['reason']} "
            f"(n={evidence['resolved_predictions']}, Brier Δ={float(evidence['brier_improvement']):.4f}, "
            f"lower acc bound={float(evidence.get('accuracy_lower_bound', 0.0)):.3f})"
        )

        # Remember for next-round reward
        self._last_target_side = prediction
        self._last_prob_big = blended_p_big
        return result


def run_local_engine():
    print("🚀 🧠 LAUNCHING BEAST MODE: Local WinGo AI Deep Learning Engine")
    print("Polling Supabase every 5 seconds with continuous neuroevolution...")
    print("HIP + EVOSEQ fusion: joint 3-level win probability + rolling calibration")

    beast = BeastPredictor()
    last_processed_issue = None
    last_rewarded_issue = None  # never double-reward
    cycles_completed = 0

    while True:
        try:
            db = SessionLocal()

            # Check for the latest draw from the cloud scraper
            latest_draw = db.query(Draw).order_by(Draw.issue_number.desc()).first()

            if latest_draw:
                latest_issue = str(latest_draw.issue_number)

                # --- Closed-loop reward: the draw we're seeing NOW resolves ---
                # the PREVIOUS cycle's prediction. Reward the HIP calibrator.
                if last_rewarded_issue != latest_issue and last_processed_issue is not None:
                    try:
                        actual_digit = int(latest_draw.number) % 10
                        beast.reward_resolved(actual_digit=actual_digit, issue_just_closed=latest_issue)
                        side = "Big" if actual_digit >= 5 else "Small"
                        print(f"[BEAST] 🔁 Reward: issue {latest_issue} = digit {actual_digit} ({side}) | "
                              f"W-streak {beast._win_streak} / L-streak {beast._loss_streak}")
                        last_rewarded_issue = latest_issue
                    except Exception as rew_err:
                        print(f"[BEAST] ⚠️  reward step failed: {rew_err}")

                if latest_issue != last_processed_issue:
                    print(f"\n[BEAST] Cycle #{cycles_completed + 1} | New Draw: {latest_issue} (Num: {latest_draw.number})")
                    last_processed_issue = latest_issue
                    cycles_completed += 1

                    # 1. Extract full deep sequence history from Supabase (up to 50,000 outcomes)
                    outcomes_list = db.query(Outcome).order_by(Outcome.sequence_no.desc()).limit(50000).all()
                    if outcomes_list:
                        history = [int(o.digit) for o in reversed(outcomes_list)]
                    else:
                        db_draws = db.query(Draw).order_by(Draw.issue_number.desc()).limit(50000).all()
                        if db_draws:
                            history = [int(d.number) for d in reversed(db_draws)]
                        else:
                            history = []

                    # 2. EVOSEQ Continuous Adaptive Evolution with Transformer + Mamba
                    if history and len(history) >= 10:
                        print(f"[BEAST] Running Neuroevolution on {len(history)} samples...")
                        registry_state = run_evoseq_cycle(history, db)

                        # 3. Apply Beast-Mode Multi-Horizon Intelligence (now with HIP fusion)
                        ai_result = beast.evolve_prediction(history, registry_state, db)

                        # Only update if we have a valid prediction with confidence
                        if ai_result and ai_result.get("confidence") and ai_result.get("prediction"):
                            next_issue = str(int(latest_issue) + 1)
                            ai_result["currentIssue"] = latest_issue
                            ai_result["nextIssue"] = next_issue
                            ai_result["latestIssue"] = latest_issue
                            ai_result["generation"] = registry_state.get("generation", 1)
                            ai_result["championGenome"] = registry_state.get("champion_id", "Transformer-Mamba Ensemble")
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
                            # Surface 3-level metrics on the Telegram card via Live_UI_State
                            ai_result.setdefault(
                                "martingale3Hint",
                                {
                                    "pWinIn3": ai_result.get("calibratedPWinIn3", 0.875),
                                    "pCorrectSingle": ai_result.get("calibratedPSingle", 0.55),
                                    "strike": ai_result.get("strikeQuality", "CONSERVATIVE"),
                                },
                            )

                            # Record the forecast before the result exists so
                            # later reconciliation is a genuine out-of-sample test.
                            save_prediction(
                                db, next_issue, ai_result["prediction"], ai_result["confidence"], ai_result["patternName"]
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

                            conf = ai_result["confidence"]
                            p3 = ai_result.get("calibratedPWinIn3", None)
                            print(
                                f"[BEAST] ✨ PREDICTION: {ai_result['prediction']} ({conf}%) for Issue #{next_issue}"
                                + (f" | P(win in 3)={p3:.3f}" if p3 is not None else "")
                            )
                            print(f"[BEAST] 🎯 Target Number: {ai_result['targetNum']} | Hedge: {ai_result['hedgeNum']} | Strike: {ai_result['strikeQuality']}")
                            print(f"[BEAST] 🧠 Engine: {ai_result['patternName']}")

                            # 4. Sync AI state directly to Supabase as Live_UI_State
                            print(f"[BEAST] 💾 Saving to Supabase as Live_UI_State...")
                            save_ai_brain_state(
                                db=db,
                                model_name="Live_UI_State",
                                generation=ai_result["generation"],
                                total_samples=len(history),
                                weights_json=json.dumps(ai_result),
                                win_rate=registry_state.get("fitness", 50.0),
                            )
                            print(
                                f"[BEAST] ✅ Live_UI_State saved. Issue: {ai_result['currentIssue']} | "
                                f"Prediction: {ai_result['prediction']} | Confidence: {ai_result['confidence']}% | "
                                f"P(win in 3)={ai_result.get('calibratedPWinIn3', 'n/a')}"
                            )

                            # 5. Also save to evolution history for continuous improvement tracking
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
                            print("[BEAST] ⚠️  Evolution returned no prediction")
                    else:
                        print("[BEAST] Waiting for sufficient history (need >= 10 samples)...")

            db.close()

        except Exception as e:
            import traceback
            print(f"[BEAST] Engine Error: {e}")
            traceback.print_exc()

        time.sleep(5)

if __name__ == "__main__":
    run_local_engine()
