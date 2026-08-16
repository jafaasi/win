# EVOSEQ — Autonomous Evolutionary Sequence Intelligence Engine

EVOSEQ evaluates sequence predictability through rigorous out-of-sample temporal validation against surrogate null baselines.

---

## 🧬 Model Populations

1. **Statistical Family**:
   - `Uniform`: Baseline maximum entropy reference ($H = \log_2(10) = 3.322\text{ bits}$).
   - `FrequencyBaseline`: Empirical unconditional marginal frequencies.
   - `MarkovModel` (Orders 1, 2, 3): Transition probability matrices with Laplace smoothing.
2. **Recurrent Family**:
   - `DiscreteHMM`: Latent Markov state clustering.
   - `ESN`: Echo State Network with random orthogonal reservoir matrix and ridge regression readout.
3. **Neural Family**:
   - `NeuralTransformer`: Multi-horizon causal self-attention network with upper-triangular lookahead masking.
4. **State-Space Family**:
   - `S4D`: Diagonal structured state-space model with HiPPO continuous-time initialization and recurrent step forwarding.
   - `MambaSSM`: Selective state-space model featuring pure-PyTorch CPU fallback and hardware-accelerated CUDA execution.

---

## 🎯 Continuous Research Loop

```
New Observations -> Data Validator -> Ingestion & Reconciliation -> Multi-Horizon Inference -> Temperature Calibration -> Environment & Drift Fingerprint -> Hypothesis Generator -> Candidate Factory (Single-Variable Mutation) -> Temporal Validation Lab (Multi-Seed + Null Referee) -> Sequential Promotion Gate -> Lineage & Genealogy Ledger
```
