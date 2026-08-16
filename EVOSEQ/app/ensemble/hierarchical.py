from typing import Dict, List, Sequence, Any, Optional
import numpy as np
from .tracker import softmax_weights, combine_predictions

class HierarchicalEnsemble:
    """
    Hierarchical Multi-Family Ensemble Architecture:
    1. Aggregates models within families:
       - Statistical (Frequency, Markov 1/2/3, HMM)
       - Recurrent (ESN)
       - Neural (Transformer, Mamba, S4D)
    2. Combines family-level distributions via meta-family weights:
       P = w_stat * P_stat + w_rec * P_rec + w_neural * P_neural
    """

    def __init__(self, family_definitions: Optional[Dict[str, List[str]]] = None):
        self.families = family_definitions or {
            "statistical": ["frequency", "markov1", "markov2", "markov3", "hmm"],
            "recurrent": ["esn", "esn256"],
            "neural": ["transformer", "mamba", "mamba2", "s4", "s4d"],
        }
        self.family_weights = {"statistical": 0.3333, "recurrent": 0.3333, "neural": 0.3334}

    def set_family_weights(self, weights: Dict[str, float]) -> None:
        total = sum(weights.values())
        if total > 0:
            self.family_weights = {k: v / total for k, v in weights.items()}

    def combine_hierarchical(
        self,
        predictions_by_model: Dict[str, np.ndarray],
        losses_by_model: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Computes hierarchical combination of model predictions.
        Returns:
        {
            "family_distributions": { "statistical": np.ndarray, ... },
            "family_weights": { "statistical": 0.3, ... },
            "ensemble_prediction": np.ndarray
        }
        """
        family_preds = {}
        
        for fam_name, member_keys in self.families.items():
            active_members = [k for k in member_keys if k in predictions_by_model]
            if not active_members:
                continue
                
            member_preds = [predictions_by_model[k] for k in active_members]
            if losses_by_model:
                member_losses = [losses_by_model.get(k, 2.3026) for k in active_members]
                w = softmax_weights(member_losses, beta=2.0)
            else:
                w = np.full(len(active_members), 1.0 / len(active_members))
                
            fam_dist = combine_predictions(member_preds, w)
            family_preds[fam_name] = fam_dist
            
        if not family_preds:
            return {"ensemble_prediction": np.full(10, 0.1), "family_distributions": {}}
            
        # Combine across active families
        active_fams = list(family_preds.keys())
        w_fams = [self.family_weights.get(k, 1.0 / len(active_fams)) for k in active_fams]
        total_w = sum(w_fams)
        w_fams_norm = [wf / total_w for wf in w_fams] if total_w > 0 else [1.0 / len(active_fams)] * len(active_fams)
        
        final_prediction = combine_predictions(
            [family_preds[k] for k in active_fams],
            w_fams_norm
        )
        
        return {
            "family_distributions": family_preds,
            "family_weights": {k: float(w_fams_norm[i]) for i, k in enumerate(active_fams)},
            "ensemble_prediction": final_prediction
        }
