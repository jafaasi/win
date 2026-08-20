from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Sequence, Tuple

from .dashboard import DashboardData
from .daily_evolution import GenerationRecord


@dataclass
class DailyReport:
    generation: int
    report_date: str
    historical_samples: int
    new_samples_today: int
    current_regime: str
    champion_model: str
    strongest_model: Optional[str]
    weakest_model: Optional[str]
    model_disagreement: float
    calibration_ece: float
    mean_brier: float
    mean_log_loss: float
    baseline_accuracy: float
    oos_accuracy: Optional[float]
    drift_detected_today: bool
    models_rejected: int
    models_promoted: int
    conclusion: str
    full_report: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)

    def render_text(self) -> str:
        lines = []
        lines.append("=" * 56)
        lines.append("  EVOSEQ DAILY INTELLIGENCE REPORT")
        lines.append("=" * 56)
        lines.append(f"  Generation:            {self.generation}")
        lines.append(f"  Report Date:           {self.report_date}")
        lines.append(f"  Historical samples:    {self.historical_samples:,}")
        lines.append(f"  New samples today:     {self.new_samples_today:,}")
        lines.append("")
        lines.append(f"  Current regime:        {self.current_regime}")
        lines.append(f"  Model champion:        {self.champion_model}")
        if self.strongest_model:
            lines.append(f"  Strongest model:       {self.strongest_model}")
        if self.weakest_model:
            lines.append(f"  Weakest model:         {self.weakest_model}")
        lines.append("")
        lines.append(f"  Model disagreement:    {self.model_disagreement:.4f}")
        lines.append(f"  Calibration (ECE):     {self.calibration_ece:.4f}")
        lines.append(f"  Mean Brier score:      {self.mean_brier:.4f}")
        lines.append(f"  Mean Log loss:         {self.mean_log_loss:.4f}")
        lines.append("")
        lines.append(f"  Baseline accuracy:     {self.baseline_accuracy:.4f}")
        if self.oos_accuracy is not None:
            lines.append(f"  OOS performance:       {self.oos_accuracy:.4f}")
        lines.append("")
        lines.append(f"  Drift detected today:  {'YES' if self.drift_detected_today else 'NO'}")
        lines.append(f"  Models rejected:       {self.models_rejected}")
        lines.append(f"  Models promoted:       {self.models_promoted}")
        lines.append("")
        lines.append("  Conclusion:")
        lines.append(f"    {self.conclusion}")
        lines.append("")
        lines.append("  Evidence-based language only.")
        lines.append("  No guaranteed winning claims.")
        lines.append("=" * 56)
        return "\n".join(lines)


class DailyIntelligenceReport:
    """
    Generates the Daily Intelligence Report:
      - Samples/history, champion, strongest/weakest models
      - Disagreement, calibration, Brier, Log-loss
      - Baselines, OOS performance
      - Drift, rejects, promotions
      - Final conclusion (NO_VERIFIED_EDGE, MODEST_OOS_IMPROVEMENT, etc.)
    """

    def generate(
        self,
        dashboard: DashboardData,
        generation_record: Optional[GenerationRecord],
        historical_samples_total: int,
        new_samples_today: int,
        mean_brier_recent: float,
        mean_log_loss_recent: float,
        strongest_model: Optional[str] = None,
        weakest_model: Optional[str] = None,
        drift_detected_today: bool = False,
    ) -> DailyReport:
        baseline_acc = max(dashboard.baseline_performance.values()) if dashboard.baseline_performance else 0.5

        oos_acc = dashboard.champion_accuracy

        # Conclusion
        if oos_acc is None or oos_acc <= baseline_acc + 0.002:
            conclusion = "NO VERIFIED EDGE DETECTED"
        elif dashboard.edge_status == "VERIFIED_EDGE" and generation_record is not None and generation_record.statistically_significant:
            conclusion = "MODEST OUT-OF-SAMPLE IMPROVEMENT DETECTED"
        elif dashboard.edge_status == "MODEST_EDGE":
            conclusion = "MODEST OUT-OF-SAMPLE SIGNAL — MONITORING"
        elif dashboard.edge_status == "TENTATIVE_EDGE":
            conclusion = "TENTATIVE SIGNAL — FURTHER EVIDENCE REQUIRED"
        else:
            conclusion = "NO VERIFIED EDGE DETECTED"

        report_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        promoted = len(generation_record.promoted_models) if generation_record else 0
        rejected = len(generation_record.rejected_models) if generation_record else 0

        full = {
            "dashboard": dashboard.to_dict(),
            "generation": generation_record.to_dict() if generation_record else {},
            "drift_detected_today": drift_detected_today,
        }

        return DailyReport(
            generation=dashboard.current_generation,
            report_date=report_date,
            historical_samples=int(historical_samples_total),
            new_samples_today=int(new_samples_today),
            current_regime=dashboard.current_regime,
            champion_model=dashboard.champion_model,
            strongest_model=strongest_model,
            weakest_model=weakest_model,
            model_disagreement=float(dashboard.model_disagreement),
            calibration_ece=float(dashboard.calibration_ece),
            mean_brier=float(mean_brier_recent),
            mean_log_loss=float(mean_log_loss_recent),
            baseline_accuracy=float(baseline_acc),
            oos_accuracy=oos_acc,
            drift_detected_today=bool(drift_detected_today),
            models_rejected=int(rejected),
            models_promoted=int(promoted),
            conclusion=conclusion,
            full_report=full,
        )
