from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

@dataclass
class LineageNode:
    model_code: str
    generation: int
    family: str
    parent_code: Optional[str]
    status: str
    historical_log_loss: float
    promotions_count: int = 0
    children: List[str] = field(default_factory=list)

class ArchitectureSurvivalAnalyzer:
    """
    Tracks lineage genealogy and calculates family survival statistics:
    SurvivalRate = Promoted Candidates / Generated Candidates
    """

    def __init__(self):
        self.lineage_tree: Dict[str, LineageNode] = {}
        self.family_counts = {
            "markov": {"generated": 0, "promoted": 0},
            "hmm": {"generated": 0, "promoted": 0},
            "esn": {"generated": 0, "promoted": 0},
            "transformer": {"generated": 0, "promoted": 0},
            "mamba": {"generated": 0, "promoted": 0},
            "s4d": {"generated": 0, "promoted": 0},
        }

    def register_candidate(
        self,
        model_code: str,
        generation: int,
        family: str,
        parent_code: Optional[str],
        log_loss: float,
        promoted: bool = False
    ) -> LineageNode:
        fam_key = family.lower()
        if fam_key in self.family_counts:
            self.family_counts[fam_key]["generated"] += 1
            if promoted:
                self.family_counts[fam_key]["promoted"] += 1
                
        node = LineageNode(
            model_code=model_code,
            generation=generation,
            family=family,
            parent_code=parent_code,
            status="PROMOTED" if promoted else "RETAINED",
            historical_log_loss=log_loss,
            promotions_count=1 if promoted else 0
        )
        self.lineage_tree[model_code] = node
        if parent_code and parent_code in self.lineage_tree:
            self.lineage_tree[parent_code].children.append(model_code)
            
        return node

    def get_survival_rates(self) -> Dict[str, float]:
        """Calculates survival rate percentage per architecture family."""
        rates = {}
        for fam, counts in self.family_counts.items():
            gen = counts["generated"]
            prom = counts["promoted"]
            rates[fam] = round(float(prom / gen) if gen > 0 else 0.0, 4)
        return rates
