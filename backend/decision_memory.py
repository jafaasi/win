#!/usr/bin/env python3
"""
Decision Memory — TAIE Component 2
====================================
Stores the full reasoning state behind every prediction so the system
can later ask: "Under what conditions was I historically successful?"

Every prediction record captures:
  - The raw prediction and confidence
  - The complete 12-model P(Big) vector
  - Signal agreement metrics
  - Adversarial report summary
  - Entropy and regime state
  - Streak state at decision time
  - The TAIE decision tier (STRONG/MODERATE/WEAK/ABSTAIN)
  - Outcome (filled after resolution)

OOS-safe split helpers
-----------------------
A critical rule: never evaluate on the same window you trained on.
The split helpers enforce a strict temporal ordering:

  Full history  ──►  [  TRAIN (60%)  |  VALIDATION (20%)  |  TEST (20%)  ]
                                                              ↑
                                              Only this window is shown
                                              to the reality check engine.

The test window is always the most recent N rounds — never shuffled.

Persistence
-----------
Records are written to the `decision_memory` table in Supabase (defined
in database.py).  In-memory deque provides fast recent-window access
without DB round-trips during every 30-second cycle.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Record structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DecisionRecord:
    """Full reasoning state for one prediction."""

    # Identity
    issue_number: str
    timestamp_utc: datetime

    # Core prediction
    prediction: str                      # "Big" | "Small"
    confidence: float                    # final displayed confidence %
    probability_big: float               # raw blended P(Big)

    # TAIE tier
    taie_tier: str                       # STRONG | MODERATE | WEAK | ABSTAIN
    action: str                          # STRIKE | FORECAST | CAUTION | SKIP

    # 12-model feature vector
    model_p_big_vector: List[float]      # len=12, P(Big) per model
    model_weights: List[float]           # len=12, Hedge weights at decision time
    model_consensus: float               # fraction on majority side

    # Signal agreement
    signal_agreement: float              # 0-1, how many signals align
    engines_agree: int                   # exploit sub-engines agreeing
    reject_iid: bool

    # Adversarial summary
    adversarial_score: float
    adversarial_verdict: str             # HOLD | CAUTION | OVERRIDE | ABSTAIN
    adversarial_net_score: float

    # Statistical state
    entropy: float
    exploit_score: float
    drift_level: str
    drift_score: float
    change_probability: float
    streak_run_length: int

    # Martingale state
    martingale_level: int
    martingale_loss_streak: int

    # Evidence gate
    validated_edge: bool
    three_level_lower_bound: float
    brier_improvement: float

    # Reasoning narrative
    prediction_reason: str               # human-readable summary

    # Outcome (filled after resolution)
    actual_size: Optional[str] = None
    actual_digit: Optional[int] = None
    is_correct: Optional[bool] = None
    log_loss: Optional[float] = None
    brier_score: Optional[float] = None

    def to_db_dict(self) -> dict:
        """Serialize for Supabase insert."""
        return {
            "issue_number": self.issue_number,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "prediction": self.prediction,
            "confidence": round(self.confidence, 2),
            "probability_big": round(self.probability_big, 6),
            "taie_tier": self.taie_tier,
            "action": self.action,
            "model_p_big_vector": json.dumps([round(x, 6) for x in self.model_p_big_vector]),
            "model_weights": json.dumps([round(x, 6) for x in self.model_weights]),
            "model_consensus": round(self.model_consensus, 4),
            "signal_agreement": round(self.signal_agreement, 4),
            "engines_agree": self.engines_agree,
            "reject_iid": bool(self.reject_iid),
            "adversarial_score": round(self.adversarial_score, 4),
            "adversarial_verdict": self.adversarial_verdict,
            "adversarial_net_score": round(self.adversarial_net_score, 4),
            "entropy": round(self.entropy, 6),
            "exploit_score": round(self.exploit_score, 4),
            "drift_level": self.drift_level,
            "drift_score": round(self.drift_score, 4),
            "change_probability": round(self.change_probability, 4),
            "streak_run_length": self.streak_run_length,
            "martingale_level": self.martingale_level,
            "martingale_loss_streak": self.martingale_loss_streak,
            "validated_edge": bool(self.validated_edge),
            "three_level_lower_bound": round(self.three_level_lower_bound, 4),
            "brier_improvement": round(self.brier_improvement, 6),
            "prediction_reason": self.prediction_reason,
            "actual_size": self.actual_size,
            "actual_digit": self.actual_digit,
            "is_correct": self.is_correct,
            "log_loss": round(self.log_loss, 6) if self.log_loss is not None else None,
            "brier_score": round(self.brier_score, 6) if self.brier_score is not None else None,
        }

    def resolve(self, actual_size: str, actual_digit: int) -> None:
        """Fill in outcome fields after the draw resolves."""
        self.actual_size = actual_size
        self.actual_digit = actual_digit
        self.is_correct = (self.prediction == actual_size)
        target = 1.0 if actual_size == "Big" else 0.0
        p = max(1e-6, min(1 - 1e-6, self.probability_big))
        self.log_loss = -(target * math.log(p) + (1 - target) * math.log(1 - p))
        self.brier_score = (self.probability_big - target) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# OOS-safe split helpers
# ─────────────────────────────────────────────────────────────────────────────

def temporal_train_val_test_split(
    records: List[DecisionRecord],
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> Tuple[List[DecisionRecord], List[DecisionRecord], List[DecisionRecord]]:
    """
    Strict temporal split — never shuffle.  Oldest records in train,
    newest in test.  Returns (train, validation, test).

    This prevents look-ahead bias: when we evaluate the reality check,
    we only use the test window (most recent records never seen during
    weight optimisation).
    """
    n = len(records)
    if n < 10:
        return records, [], []
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))
    return records[:train_end], records[train_end:val_end], records[val_end:]


def resolved_only(records: List[DecisionRecord]) -> List[DecisionRecord]:
    """Filter to only records with known outcomes."""
    return [r for r in records if r.is_correct is not None]


def to_pairs(records: List[DecisionRecord]) -> List[Tuple[float, int]]:
    """Convert resolved records to (p_big, actual_side_0_1) pairs."""
    return [
        (r.probability_big, 1 if r.actual_size == "Big" else 0)
        for r in records if r.actual_size is not None
    ]


def accuracy(records: List[DecisionRecord]) -> float:
    res = resolved_only(records)
    if not res:
        return 0.5
    return sum(1 for r in res if r.is_correct) / len(res)


def brier_score(records: List[DecisionRecord]) -> float:
    pairs = to_pairs(records)
    if not pairs:
        return 0.25
    return sum((p - a) ** 2 for p, a in pairs) / len(pairs)


def log_loss_mean(records: List[DecisionRecord]) -> float:
    res = resolved_only(records)
    if not res:
        return math.log(2)
    return sum(r.log_loss for r in res if r.log_loss is not None) / len(res)


def three_level_win_rate(records: List[DecisionRecord]) -> float:
    """Stride-1 rolling windows of 3: fraction where at least 1 is correct."""
    res = resolved_only(records)
    if len(res) < 3:
        return 0.5
    wins = total = 0
    for i in range(len(res) - 2):
        if any(res[i + j].is_correct for j in range(3)):
            wins += 1
        total += 1
    return wins / total if total else 0.5


def taie_tier_breakdown(records: List[DecisionRecord]) -> Dict[str, Dict]:
    """Per-tier accuracy, Brier, and count."""
    tiers: Dict[str, List[DecisionRecord]] = {}
    for r in resolved_only(records):
        tiers.setdefault(r.taie_tier, []).append(r)
    return {
        tier: {
            "n": len(recs),
            "accuracy": round(accuracy(recs), 4),
            "brier": round(brier_score(recs), 4),
            "three_level_wr": round(three_level_win_rate(recs), 4),
        }
        for tier, recs in tiers.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# DecisionMemory — main class
# ─────────────────────────────────────────────────────────────────────────────

class DecisionMemory:
    """
    In-memory decision store with Supabase persistence.

    Per-cycle usage (fast path):
        memory.store(record)           # write before draw resolves
        memory.resolve(issue, actual)  # backfill after draw resolves

    Daily analysis (slow path):
        train, val, test = memory.oos_split()
        report = memory.analyse_conditions()  # "when was I successful?"
    """

    DB_KEY = "Decision_Memory_State"

    def __init__(self, maxlen: int = 5000):
        self._records: Deque[DecisionRecord] = deque(maxlen=maxlen)
        # issue_number → index position for fast resolve lookups
        self._index: Dict[str, DecisionRecord] = {}

    # ── Storage ──────────────────────────────────────────────────────────────

    def store(self, record: DecisionRecord) -> None:
        """Store a new prediction record (before outcome known)."""
        self._records.append(record)
        self._index[record.issue_number] = record

    def resolve(self, issue_number: str, actual_size: str, actual_digit: int) -> bool:
        """Fill in outcome for a previously stored record. Returns True if found."""
        rec = self._index.get(str(issue_number))
        if rec is None:
            return False
        rec.resolve(actual_size, actual_digit)
        return True

    # ── Query ─────────────────────────────────────────────────────────────────

    def recent(self, n: int = 200) -> List[DecisionRecord]:
        return list(self._records)[-n:]

    def all_resolved(self) -> List[DecisionRecord]:
        return resolved_only(list(self._records))

    def oos_split(
        self, train_frac: float = 0.60, val_frac: float = 0.20
    ) -> Tuple[List[DecisionRecord], List[DecisionRecord], List[DecisionRecord]]:
        """Return (train, val, test) with strict temporal ordering."""
        return temporal_train_val_test_split(
            self.all_resolved(), train_frac, val_frac
        )

    # ── Analysis: "Under what conditions was I successful?" ───────────────────

    def analyse_conditions(self) -> Dict[str, Any]:
        """
        Interrogates the resolved decision memory to find the feature
        conditions under which the engine performed best.

        Returns a report with:
          - per_tier performance (STRONG / MODERATE / WEAK)
          - best_conditions: feature ranges where accuracy was highest
          - worst_conditions: feature ranges where accuracy was lowest
          - oos_accuracy: accuracy on the test split only
          - recommendations: actionable thresholds to apply going forward
        """
        _, _, test = self.oos_split()
        all_res = self.all_resolved()

        if len(all_res) < 20:
            return {"status": "INSUFFICIENT_DATA", "n": len(all_res)}

        # Per-tier breakdown (on full resolved history)
        tier_breakdown = taie_tier_breakdown(all_res)

        # OOS test performance
        oos_acc = accuracy(test)
        oos_brier = brier_score(test)
        oos_3lvl = three_level_win_rate(test)

        # Find conditions where accuracy > 60% vs < 50%
        good_records = [r for r in all_res if r.is_correct]
        bad_records  = [r for r in all_res if not r.is_correct]

        def safe_mean(lst, attr):
            vals = [getattr(r, attr) for r in lst if getattr(r, attr) is not None]
            return round(float(np.mean(vals)), 4) if vals else None

        best_conditions = {
            "avg_model_consensus":    safe_mean(good_records, "model_consensus"),
            "avg_adversarial_score":  safe_mean(good_records, "adversarial_score"),
            "avg_exploit_score":      safe_mean(good_records, "exploit_score"),
            "avg_entropy":            safe_mean(good_records, "entropy"),
            "pct_validated_edge":     (
                sum(1 for r in good_records if r.validated_edge) / len(good_records)
                if good_records else 0.0
            ),
        }
        worst_conditions = {
            "avg_model_consensus":    safe_mean(bad_records, "model_consensus"),
            "avg_adversarial_score":  safe_mean(bad_records, "adversarial_score"),
            "avg_exploit_score":      safe_mean(bad_records, "exploit_score"),
            "avg_entropy":            safe_mean(bad_records, "entropy"),
            "pct_validated_edge":     (
                sum(1 for r in bad_records if r.validated_edge) / len(bad_records)
                if bad_records else 0.0
            ),
        }

        # Derive actionable thresholds
        recommendations = []
        if best_conditions["avg_model_consensus"] and worst_conditions["avg_model_consensus"]:
            threshold = round(
                0.6 * best_conditions["avg_model_consensus"]
                + 0.4 * worst_conditions["avg_model_consensus"], 3
            )
            recommendations.append(
                f"Require model_consensus ≥ {threshold:.2f} for STRONG tier"
            )
        if best_conditions["avg_adversarial_score"] and worst_conditions["avg_adversarial_score"]:
            threshold = round(
                0.4 * best_conditions["avg_adversarial_score"]
                + 0.6 * worst_conditions["avg_adversarial_score"], 3
            )
            recommendations.append(
                f"Reject predictions with adversarial_score ≥ {threshold:.2f}"
            )

        return {
            "status": "OK",
            "n_total": len(all_res),
            "n_test": len(test),
            "oos_accuracy": round(oos_acc, 4),
            "oos_brier": round(oos_brier, 4),
            "oos_three_level_wr": round(oos_3lvl, 4),
            "tier_breakdown": tier_breakdown,
            "best_conditions": best_conditions,
            "worst_conditions": worst_conditions,
            "recommendations": recommendations,
            "analysed_at": datetime.utcnow().isoformat(),
        }

    # ── Supabase persistence ──────────────────────────────────────────────────

    def persist_record(self, record: DecisionRecord, db) -> None:
        """Write one DecisionRecord to the decision_memory table."""
        try:
            from backend.database import DecisionMemoryRow
            existing = (
                db.query(DecisionMemoryRow)
                .filter(DecisionMemoryRow.issue_number == record.issue_number)
                .first()
            )
            if existing:
                # Update outcome fields only
                if record.actual_size is not None:
                    existing.actual_size = record.actual_size
                    existing.actual_digit = record.actual_digit
                    existing.is_correct = record.is_correct
                    existing.log_loss = record.log_loss
                    existing.brier_score_val = record.brier_score
            else:
                d = record.to_db_dict()
                row = DecisionMemoryRow(
                    issue_number=d["issue_number"],
                    timestamp_utc=datetime.fromisoformat(d["timestamp_utc"]),
                    prediction=d["prediction"],
                    confidence=d["confidence"],
                    probability_big=d["probability_big"],
                    taie_tier=d["taie_tier"],
                    action=d["action"],
                    model_p_big_vector=d["model_p_big_vector"],
                    model_weights=d["model_weights"],
                    model_consensus=d["model_consensus"],
                    signal_agreement=d["signal_agreement"],
                    engines_agree=d["engines_agree"],
                    reject_iid=d["reject_iid"],
                    adversarial_score=d["adversarial_score"],
                    adversarial_verdict=d["adversarial_verdict"],
                    adversarial_net_score=d["adversarial_net_score"],
                    entropy=d["entropy"],
                    exploit_score=d["exploit_score"],
                    drift_level=d["drift_level"],
                    drift_score=d["drift_score"],
                    change_probability=d["change_probability"],
                    streak_run_length=d["streak_run_length"],
                    martingale_level=d["martingale_level"],
                    martingale_loss_streak=d["martingale_loss_streak"],
                    validated_edge=d["validated_edge"],
                    three_level_lower_bound=d["three_level_lower_bound"],
                    brier_improvement=d["brier_improvement"],
                    prediction_reason=d["prediction_reason"],
                )
                db.add(row)
            db.commit()
        except Exception as e:
            try:
                db.rollback()
            except Exception:
                pass
            print(f"[DecisionMemory] persist_record failed: {e}")

    def load_recent_from_db(self, db, limit: int = 2000) -> int:
        """
        Hydrate in-memory deque from Supabase on startup.
        Returns number of records loaded.
        """
        try:
            from backend.database import DecisionMemoryRow
            rows = (
                db.query(DecisionMemoryRow)
                .order_by(DecisionMemoryRow.id.desc())
                .limit(limit)
                .all()
            )
            loaded = 0
            for row in reversed(rows):
                try:
                    vec = json.loads(row.model_p_big_vector) if row.model_p_big_vector else []
                    wts = json.loads(row.model_weights) if row.model_weights else []
                    rec = DecisionRecord(
                        issue_number=row.issue_number,
                        timestamp_utc=row.timestamp_utc or datetime.utcnow(),
                        prediction=row.prediction or "Big",
                        confidence=float(row.confidence or 60),
                        probability_big=float(row.probability_big or 0.5),
                        taie_tier=row.taie_tier or "MODERATE",
                        action=row.action or "FORECAST",
                        model_p_big_vector=vec,
                        model_weights=wts,
                        model_consensus=float(row.model_consensus or 0.5),
                        signal_agreement=float(row.signal_agreement or 0.5),
                        engines_agree=int(row.engines_agree or 0),
                        reject_iid=bool(row.reject_iid),
                        adversarial_score=float(row.adversarial_score or 0),
                        adversarial_verdict=row.adversarial_verdict or "HOLD",
                        adversarial_net_score=float(row.adversarial_net_score or 0),
                        entropy=float(row.entropy or 2.3),
                        exploit_score=float(row.exploit_score or 0),
                        drift_level=row.drift_level or "STABLE",
                        drift_score=float(row.drift_score or 0),
                        change_probability=float(row.change_probability or 0),
                        streak_run_length=int(row.streak_run_length or 0),
                        martingale_level=int(row.martingale_level or 1),
                        martingale_loss_streak=int(row.martingale_loss_streak or 0),
                        validated_edge=bool(row.validated_edge),
                        three_level_lower_bound=float(row.three_level_lower_bound or 0),
                        brier_improvement=float(row.brier_improvement or 0),
                        prediction_reason=row.prediction_reason or "",
                        actual_size=row.actual_size,
                        actual_digit=row.actual_digit,
                        is_correct=row.is_correct,
                        log_loss=row.log_loss,
                        brier_score=row.brier_score_val,
                    )
                    self.store(rec)
                    loaded += 1
                except Exception:
                    continue
            print(f"[DecisionMemory] Loaded {loaded} records from Supabase")
            return loaded
        except Exception as e:
            print(f"[DecisionMemory] load_recent_from_db failed: {e}")
            return 0

    # ── Factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def build_record(
        issue_number: str,
        result: dict,
        adversarial_report,
        taie_tier: str,
    ) -> "DecisionRecord":
        """
        Construct a DecisionRecord from the UltraIntelligenceEngine result dict
        and an AdversarialReport.  Called once per prediction cycle.
        """
        vec = result.get("modelPBigVector", {})
        model_vec = list(vec.values()) if isinstance(vec, dict) else []
        ew = result.get("ensembleWeights", {})
        weights = list(ew.values()) if isinstance(ew, dict) else []

        evidence = result.get("evidence", {})
        scorecard = result.get("scorecard", {})

        # Build human-readable reason
        parts = [
            f"TAIE:{taie_tier}",
            f"Consensus:{result.get('modelConsensus', 0):.0%}",
            f"Exploit:{result.get('exploitScore', 0):.2f}",
            f"Adversarial:{adversarial_report.verdict}({adversarial_report.adversarial_score:.2f})",
            f"Drift:{result.get('driftLevel', 'STABLE')}",
            f"Edge:{'Y' if evidence.get('validated_edge') else 'N'}",
            f"Streak:W{scorecard.get('win_streak',0)}/L{scorecard.get('loss_streak',0)}",
        ]
        reason = " | ".join(parts)

        return DecisionRecord(
            issue_number=str(issue_number),
            timestamp_utc=datetime.utcnow(),
            prediction=result.get("prediction", "Big"),
            confidence=float(result.get("confidence", 60)),
            probability_big=float(result.get("probability_big", 0.5)),
            taie_tier=taie_tier,
            action=result.get("action", "FORECAST"),
            model_p_big_vector=model_vec,
            model_weights=weights,
            model_consensus=float(result.get("modelConsensus", 0.5)),
            signal_agreement=float(result.get("enginesAgree", 0)) / 4.0,
            engines_agree=int(result.get("enginesAgree", 0)),
            reject_iid=bool(result.get("rejectIID", False)),
            adversarial_score=float(adversarial_report.adversarial_score),
            adversarial_verdict=adversarial_report.verdict,
            adversarial_net_score=float(adversarial_report.net_score),
            entropy=float(result.get("entropy", 2.3)),
            exploit_score=float(result.get("exploitScore", 0)),
            drift_level=result.get("driftLevel", "STABLE"),
            drift_score=float(result.get("driftScore", 0)),
            change_probability=float(result.get("changeProbability", 0)),
            streak_run_length=int(result.get("streakRunLength", 0)),
            martingale_level=int(result.get("martingaleLevel", 1)),
            martingale_loss_streak=int(result.get("martingaleLossStreak", 0)),
            validated_edge=bool(evidence.get("validated_edge", False)),
            three_level_lower_bound=float(evidence.get("three_level_lower_bound", 0)),
            brier_improvement=float(evidence.get("brier_improvement", 0)),
            prediction_reason=reason,
        )
