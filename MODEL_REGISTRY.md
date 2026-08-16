# EVOSEQ Model Registry & Lineage Management

Every candidate, challenger, and champion model in EVOSEQ is versioned, auditable, and tracked through its evolutionary genealogy.

---

## 🏛️ Model Lifecycle Stages

1. `CANDIDATE`: Synthesized by Candidate Factory via single-variable mutation over `SEARCH_SPACE`.
2. `SANITY_CHECK`: Verified for output shape, deterministic inference, and valid probability simplex ($\sum p = 1.0, p_k \ge 0$).
3. `RESEARCH_VALIDATION`: Evaluated on locked inner research folds across seeds (1, 7, 42, 123, 999).
4. `ROBUSTNESS_VALIDATION`: Tested against empirical surrogate null models (IID, Markov, Shuffle).
5. `PROBATION`: Deployed in parallel forward A/B testing against current Champion.
6. `CHAMPION`: Crowned active production champion upon passing all promotion gates.
7. `RETIRED`: Archived with complete historical results and artifacts intact.

---

## 🌳 Phylogenetic Lineage Tree

```
Generation 0 (Baselines)
      ↓
Generation 1 (First Evolved SSM & Transformer Candidates)
      ↓
Generation 2 (Meta-Gated Dynamic Hierarchical Ensembles)
      ↓
Generation 3+ (Autonomous Hypotheses & Evolved Architectures)
```

The system preserves all historical nodes to calculate **Architecture Survival Rates**:
$$\text{SurvivalRate} = \frac{\text{Promoted Models}}{\text{Generated Candidates}}$$
