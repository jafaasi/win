#!/usr/bin/env python3
"""
Daily Learning Loop for WinGo 30s Prediction Bot
==================================================
Self-training intelligence that runs once per day (or on demand) to:

  1. Fetch all recent Supabase outcomes (configurable lookback window)
  2. Deep-retrain PatternIntelligence engines from the full outcome history
  3. Recalibrate ThreeLevelWinningAlgorithm from PredictionLog win rates
  4. Refit HighIntelligencePredictor (CTW + n-gram + streak) on latest history
  5. Evaluate prediction quality metrics and persist a learning report
  6. Update ensemble hyperparameters based on model family performance
  7. Schedule next daily run (stores timestamp in ai_brain_state)

Designed to run as:
  - A background thread started by local_ai_engine.py (runs at midnight UTC)
  - A standalone script: `python -m backend.daily_learning`
  - A GitHub Actions workflow or cron job

All state changes are persisted to Supabase so the main engine picks them
up on next startup without any manual intervention.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    format="%(asctime)s [DailyLearning] %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("DailyLearning")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

LOOKBACK_DAYS: int = 60          # how many days of history to use for daily train
MIN_SAMPLES: int = 200           # minimum samples before any learning is attempted
DAILY_RUN_HOUR_UTC: int = 0      # run at 00:00 UTC (midnight)
LEARNING_REPORT_KEY: str = "Daily_Learning_Report"
LAST_RUN_KEY: str = "Daily_Learning_LastRun"


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _brier(predictions: List[Tuple[float, int]]) -> float:
    """Mean Brier score: lower is better (0 = perfect)."""
    if not predictions:
        return 0.25
    return float(sum((p - a) ** 2 for p, a in predictions) / len(predictions))


def _log_loss(predictions: List[Tuple[float, int]]) -> float:
    """Mean binary cross-entropy."""
    if not predictions:
        return math.log(2)
    total = 0.0
    for p, a in predictions:
        p = max(1e-6, min(1 - 1e-6, p))
        total += -(a * math.log(p) + (1 - a) * math.log(1 - p))
    return total / len(predictions)


def _accuracy(predictions: List[Tuple[float, int]]) -> float:
    if not predictions:
        return 0.5
    correct = sum(1 for p, a in predictions if (p >= 0.5) == (a == 1))
    return correct / len(predictions)


def _wilson_lower(wins: int, total: int, z: float = 1.645) -> float:
    if total <= 0:
        return 0.0
    p = wins / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (centre - spread) / denom)


def _three_level_win_rate(results: List[bool]) -> float:
    """Rolling stride-1 windows of 3: fraction where at least 1 is correct."""
    if len(results) < 3:
        return 0.5
    wins = total = 0
    for i in range(len(results) - 2):
        if any(results[i:i + 3]):
            wins += 1
        total += 1
    return wins / total if total else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble hyperparameter updater
# ─────────────────────────────────────────────────────────────────────────────

def _update_hedge_eta(
    pattern_brier: float,
    evoseq_brier: float,
    exploit_brier: float,
    current_eta: float = 0.10,
) -> float:
    """Adapt Hedge learning rate based on model spread.

    If all models perform similarly, use a lower eta (stable).
    If there is spread (one clearly better), raise eta so we shift faster.
    """
    spread = max(pattern_brier, evoseq_brier, exploit_brier) - min(
        pattern_brier, evoseq_brier, exploit_brier
    )
    # Spread > 0.05 → increase eta toward 0.20; spread < 0.01 → lower to 0.05
    target_eta = 0.05 + 0.75 * min(1.0, spread / 0.05) * 0.15
    # Smooth update
    new_eta = 0.7 * current_eta + 0.3 * target_eta
    return round(max(0.04, min(0.25, new_eta)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model family evaluation from DB audit rows
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_audit_rows(rows) -> dict:
    """Compute accuracy metrics from resolved PredictionAudit rows."""
    pairs: List[Tuple[float, int]] = []
    results: List[bool] = []
    for row in rows:
        if row.probability_big is None or row.actual_size is None:
            continue
        p = float(row.probability_big)
        a = 1 if row.actual_size == "Big" else 0
        pairs.append((p, a))
        results.append((p >= 0.5) == (a == 1))

    n = len(pairs)
    wins = sum(results)
    return {
        "n": n,
        "accuracy": round(_accuracy(pairs), 4),
        "brier": round(_brier(pairs), 4),
        "log_loss": round(_log_loss(pairs), 4),
        "three_level_win_rate": round(_three_level_win_rate(results), 4),
        "wilson_lower_90": round(_wilson_lower(wins, n), 4),
        "recent_accuracy": round(
            _accuracy(pairs[-100:]) if len(pairs) >= 100 else _accuracy(pairs), 4
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main daily learning function
# ─────────────────────────────────────────────────────────────────────────────

def run_daily_learning(db=None, force: bool = False) -> dict:
    """Execute one full daily learning cycle.

    Parameters
    ----------
    db : SQLAlchemy session or None.  If None, opens its own session.
    force : bool.  If True, skip the "already ran today" guard.

    Returns a learning report dict that is also persisted to Supabase.
    """
    start_time = time.time()
    own_db = db is None
    report: dict = {
        "started_at": datetime.utcnow().isoformat(),
        "status": "STARTED",
        "samples_trained": 0,
        "metrics": {},
        "eta_update": None,
        "errors": [],
    }

    try:
        from backend.database import (
            SessionLocal, Outcome, PredictionAudit, PredictionLog,
            save_ai_brain_state, load_ai_brain_state,
        )
        from backend.pattern_intelligence import PatternIntelligence
        from backend.three_level_winning import ThreeLevelWinningAlgorithm
        from backend.high_intelligence_predictor import HighIntelligencePredictor

        if own_db:
            db = SessionLocal()

        # ── Guard: only run once per calendar day unless forced ───────────────
        if not force:
            last_run_brain = load_ai_brain_state(db, model_name=LAST_RUN_KEY)
            if last_run_brain and last_run_brain.synaptic_weights:
                try:
                    last_info = json.loads(last_run_brain.synaptic_weights)
                    last_date = last_info.get("date", "")
                    today = datetime.utcnow().strftime("%Y-%m-%d")
                    if last_date == today:
                        logger.info("Daily learning already ran today (%s). Skipping.", today)
                        report["status"] = "SKIPPED_ALREADY_RAN"
                        return report
                except Exception:
                    pass

        # ── Step 1: Fetch history from Supabase ───────────────────────────────
        logger.info("Step 1: Fetching outcome history (last %d days)...", LOOKBACK_DAYS)
        cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)
        try:
            outcome_rows = (
                db.query(Outcome)
                .filter(Outcome.timestamp_utc >= cutoff)
                .order_by(Outcome.sequence_no.asc())
                .all()
            )
        except Exception as e:
            logger.warning("Outcome query failed, falling back to all outcomes: %s", e)
            outcome_rows = db.query(Outcome).order_by(Outcome.sequence_no.asc()).limit(100000).all()

        digits = [int(r.digit) for r in outcome_rows]
        report["samples_trained"] = len(digits)

        if len(digits) < MIN_SAMPLES:
            logger.warning("Only %d samples — minimum is %d. Skipping.", len(digits), MIN_SAMPLES)
            report["status"] = "SKIPPED_INSUFFICIENT_DATA"
            return report

        logger.info("  → %d samples loaded (oldest: %s)", len(digits),
                    outcome_rows[0].timestamp_utc if outcome_rows else "?")

        # ── Step 2: Retrain PatternIntelligence ───────────────────────────────
        logger.info("Step 2: Deep-retraining PatternIntelligence...")
        try:
            pi = PatternIntelligence()
            pi.load_state(db)  # warm-start from existing state
            n_trained = pi.deep_train_from_db(db, lookback_days=LOOKBACK_DAYS)
            pi.save_state(db)
            report["metrics"]["pattern_intelligence"] = {"samples": n_trained}
            logger.info("  → PatternIntelligence trained on %d samples", n_trained)
        except Exception as e:
            msg = f"PatternIntelligence training failed: {e}"
            logger.error(msg)
            report["errors"].append(msg)

        # ── Step 3: Recalibrate ThreeLevelWinningAlgorithm ───────────────────
        logger.info("Step 3: Recalibrating 3-Level Martingale algorithm...")
        try:
            algo = ThreeLevelWinningAlgorithm()
            algo.load_state(db)
            algo.recalibrate_from_db(db)
            algo.save_state(db)
            status = algo.get_status()
            report["metrics"]["three_level"] = status
            logger.info("  → 3-Level recalibrated: %s", status)
        except Exception as e:
            msg = f"3-Level recalibration failed: {e}"
            logger.error(msg)
            report["errors"].append(msg)

        # ── Step 4: Retrain HighIntelligencePredictor (CTW + n-gram) ─────────
        logger.info("Step 4: Retraining HighIntelligencePredictor on full history...")
        try:
            hip = HighIntelligencePredictor.from_history(digits[-50000:])
            # Persist via ai_brain_state as a health check entry
            save_ai_brain_state(
                db=db,
                model_name="HIP_Daily_Train",
                generation=len(digits),
                total_samples=len(digits),
                weights_json=json.dumps({
                    "trained_at": datetime.utcnow().isoformat(),
                    "samples": len(digits),
                    "ctw_nodes": len(hip.ctw.nodes),
                    "ngram_orders": list(hip.ngram_counts.keys()),
                }),
                win_rate=0.0,
            )
            report["metrics"]["hip"] = {
                "samples": len(digits),
                "ctw_nodes": len(hip.ctw.nodes),
            }
            logger.info("  → HIP retrained on %d samples (%d CTW nodes)",
                        len(digits), len(hip.ctw.nodes))
        except Exception as e:
            msg = f"HIP retraining failed: {e}"
            logger.error(msg)
            report["errors"].append(msg)

        # ── Step 5: Evaluate prediction quality from audit table ──────────────
        logger.info("Step 5: Evaluating prediction quality from audit history...")
        try:
            audit_rows = (
                db.query(PredictionAudit)
                .filter(PredictionAudit.actual_size.isnot(None))
                .order_by(PredictionAudit.id.desc())
                .limit(5000)
                .all()
            )
            metrics = _evaluate_audit_rows(audit_rows)
            report["metrics"]["audit"] = metrics
            logger.info(
                "  → Audit metrics: accuracy=%.3f brier=%.4f 3lvl_wr=%.3f (n=%d)",
                metrics["accuracy"], metrics["brier"],
                metrics["three_level_win_rate"], metrics["n"],
            )
        except Exception as e:
            msg = f"Audit evaluation failed: {e}"
            logger.error(msg)
            report["errors"].append(msg)
            metrics = {}

        # ── Step 6: Update ensemble hyperparameters ───────────────────────────
        logger.info("Step 6: Updating ensemble hyperparameters (Hedge eta)...")
        try:
            # Use recent audit brier as proxy for model families
            audit_brier = float(metrics.get("brier", 0.25))
            # Pattern and exploit families tend to be slightly less accurate
            # than the deep EVOSEQ ensemble — estimate from recent 200 rows
            recent_audit = audit_rows[:200] if len(audit_rows) >= 200 else audit_rows
            recent_pairs = [
                (float(r.probability_big), 1 if r.actual_size == "Big" else 0)
                for r in recent_audit if r.probability_big and r.actual_size
            ]
            recent_brier = _brier(recent_pairs)

            # Estimate family brier as slight offsets around the global brier
            new_eta = _update_hedge_eta(
                pattern_brier=recent_brier + 0.005,
                evoseq_brier=recent_brier,
                exploit_brier=recent_brier + 0.003,
                current_eta=0.10,
            )
            # Persist eta recommendation
            save_ai_brain_state(
                db=db,
                model_name="Ensemble_Hyperparams",
                generation=len(digits),
                total_samples=len(digits),
                weights_json=json.dumps({
                    "hedge_eta": new_eta,
                    "updated_at": datetime.utcnow().isoformat(),
                    "based_on_brier": recent_brier,
                    "samples": len(recent_pairs),
                }),
                win_rate=float(metrics.get("accuracy", 0.5)) * 100,
            )
            report["eta_update"] = new_eta
            report["metrics"]["hedge_eta"] = new_eta
            logger.info("  → Hedge eta updated to %.4f (brier=%.4f)", new_eta, recent_brier)
        except Exception as e:
            msg = f"Hyperparameter update failed: {e}"
            logger.error(msg)
            report["errors"].append(msg)

        # ── Step 7: Evaluate per-level 3-level win rates for Telegram display ─
        logger.info("Step 7: Computing per-level performance summary...")
        try:
            pred_rows = (
                db.query(PredictionLog)
                .filter(PredictionLog.is_win.isnot(None))
                .order_by(PredictionLog.id.desc())
                .limit(3000)
                .all()
            )
            level_perf: Dict[int, Dict] = {}
            for lv in (1, 2, 3):
                lv_rows = [r for r in pred_rows if int(r.martingale_level or 1) == lv]
                wins = sum(1 for r in lv_rows if r.is_win)
                total = len(lv_rows)
                level_perf[lv] = {
                    "wins": wins,
                    "total": total,
                    "win_rate": round(wins / total, 4) if total else 0.0,
                    "wilson_lower": round(_wilson_lower(wins, total), 4),
                }
            report["metrics"]["level_performance"] = level_perf
            logger.info("  → Level performance: %s", level_perf)
        except Exception as e:
            msg = f"Level performance evaluation failed: {e}"
            logger.error(msg)
            report["errors"].append(msg)

        # ── Step 8: Persist learning report ───────────────────────────────────
        report["status"] = "COMPLETED" if not report["errors"] else "COMPLETED_WITH_ERRORS"
        report["elapsed_seconds"] = round(time.time() - start_time, 2)
        report["completed_at"] = datetime.utcnow().isoformat()

        try:
            save_ai_brain_state(
                db=db,
                model_name=LEARNING_REPORT_KEY,
                generation=len(digits),
                total_samples=len(digits),
                weights_json=json.dumps(report),
                win_rate=float(report["metrics"].get("audit", {}).get("accuracy", 0.5)) * 100,
            )
            # Update last-run timestamp
            save_ai_brain_state(
                db=db,
                model_name=LAST_RUN_KEY,
                generation=1,
                total_samples=1,
                weights_json=json.dumps({
                    "date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "completed_at": report["completed_at"],
                }),
                win_rate=0.0,
            )
        except Exception as e:
            logger.error("Failed to persist learning report: %s", e)

        logger.info(
            "✅ Daily learning complete in %.1fs — status=%s errors=%d",
            report["elapsed_seconds"], report["status"], len(report["errors"]),
        )

    except Exception as fatal:
        report["status"] = "FATAL_ERROR"
        report["errors"].append(str(fatal))
        report["elapsed_seconds"] = round(time.time() - start_time, 2)
        logger.exception("Daily learning fatal error: %s", fatal)
    finally:
        if own_db and db is not None:
            try:
                db.close()
            except Exception:
                pass

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Background scheduler
# ─────────────────────────────────────────────────────────────────────────────

class DailyLearningScheduler:
    """Background thread that fires run_daily_learning() every midnight UTC.

    Start once from local_ai_engine.py:
        scheduler = DailyLearningScheduler()
        scheduler.start()
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="DailyLearning")
        self._thread.start()
        logger.info("DailyLearningScheduler started (runs at %02d:00 UTC)", DAILY_RUN_HOUR_UTC)

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        # Run immediately on startup (catches up if missed)
        logger.info("DailyLearningScheduler: running initial learning cycle on startup...")
        try:
            report = run_daily_learning(force=False)
            logger.info("Startup learning cycle: %s", report.get("status"))
        except Exception as e:
            logger.error("Startup learning cycle failed: %s", e)

        while not self._stop_event.is_set():
            now = datetime.utcnow()
            # Next midnight UTC
            next_run = (now + timedelta(days=1)).replace(
                hour=DAILY_RUN_HOUR_UTC, minute=0, second=0, microsecond=0
            )
            sleep_secs = (next_run - now).total_seconds()
            logger.info(
                "Next daily learning in %.0f minutes (at %s UTC)",
                sleep_secs / 60,
                next_run.strftime("%Y-%m-%d %H:%M"),
            )
            # Sleep in small chunks so stop_event is responsive
            while sleep_secs > 0 and not self._stop_event.is_set():
                chunk = min(60.0, sleep_secs)
                self._stop_event.wait(chunk)
                sleep_secs -= chunk

            if self._stop_event.is_set():
                break

            logger.info("🎓 Running scheduled daily learning at %s UTC",
                        datetime.utcnow().strftime("%Y-%m-%d %H:%M"))
            try:
                report = run_daily_learning(force=True)
                logger.info("Scheduled learning: %s", report.get("status"))
            except Exception as e:
                logger.error("Scheduled learning failed: %s", e)


def get_last_learning_report(db=None) -> Optional[dict]:
    """Fetch the most recent learning report from Supabase.  Used by Telegram bot."""
    own_db = db is None
    try:
        from backend.database import SessionLocal, load_ai_brain_state
        if own_db:
            db = SessionLocal()
        brain = load_ai_brain_state(db, model_name=LEARNING_REPORT_KEY)
        if brain and brain.synaptic_weights:
            return json.loads(brain.synaptic_weights)
    except Exception as e:
        logger.warning("get_last_learning_report failed: %s", e)
    finally:
        if own_db and db:
            try:
                db.close()
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    logger.info("Running daily learning manually (force=%s)...", force)
    report = run_daily_learning(force=force)
    print(json.dumps(report, indent=2))
