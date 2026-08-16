from dataclasses import dataclass
from typing import Dict, Any, Optional

class SystemEventType:
    OUTCOME_ARRIVED = "OUTCOME_ARRIVED"
    FEATURES_UPDATED = "FEATURES_UPDATED"
    PREDICTION_GENERATED = "PREDICTION_GENERATED"
    OUTCOME_RECONCILED = "OUTCOME_RECONCILED"
    DRIFT_EVALUATED = "DRIFT_EVALUATED"
    CHALLENGER_SPAWNED = "CHALLENGER_SPAWNED"
    RESEARCH_AUDITED = "RESEARCH_AUDITED"
    CHAMPION_PROMOTED = "CHAMPION_PROMOTED"
    CHAMPION_ROLLBACK = "CHAMPION_ROLLBACK"
    EVOLUTION_DECISION = "EVOLUTION_DECISION"

@dataclass
class SystemEvent:
    event_type: str
    sequence_no: int
    payload: Dict[str, Any]
    model_version_id: Optional[int] = None
