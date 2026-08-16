# EVOSEQ Database Schema & Migrations

All migrations in EVOSEQ are purely additive, ensuring zero data loss and 100% backward compatibility with existing historical draw sequences.

---

## 🗄️ Core Tables

1. **`draws`**:
   - `id`, `issue_number` (unique), `number` (0–9), `color`, `size`, `created_at`.
2. **`prediction_logs`**:
   - `id`, `issue_number` (unique), `predicted_size`, `confidence`, `actual_size`, `is_win`, `martingale_level`, `pattern_detected`, `created_at`.
3. **`model_versions`**:
   - `id`, `model_name`, `version`, `parameters` (JSON), `training_end_sequence`, `validation_score`, `log_loss`, `brier_score`, `status`, `created_at`.
4. **`prediction_audit`**:
   - `id`, `sequence_no`, `model_version`, `probability_big`, `predicted_digit`, `actual_number`, `is_correct`, `log_loss`, `brier_score`, `entropy`, `regime_id`, `drift_score`, `null_advantage`, `created_at`.
5. **`ensemble_observations`**:
   - `id`, `sequence_no`, `environment` (JSON), `model_predictions` (JSON), `model_weights` (JSON), `ensemble_prediction` (JSON), `actual_digit`, `ensemble_log_loss`, `disagreement`, `created_at`.
6. **`research_hypotheses`**:
   - `id`, `hypothesis_code` (unique), `category`, `parent_model_id`, `description`, `configuration` (JSON), `expected_effect`, `priority`, `budget`, `status`, `created_at`.
7. **`model_candidates`**:
   - `id`, `candidate_code` (unique), `hypothesis_id`, `parent_model_id`, `generation`, `family`, `configuration` (JSON), `status`, `created_at`.
8. **`experiment_results`**:
   - `id`, `candidate_id`, `fold`, `seed`, `horizon`, `log_loss`, `brier_score`, `accuracy`, `calibration_error`, `null_p_value`, `runtime_seconds`, `created_at`.

---

## 📜 Migrations Ledger

- `migrations/001_initial_schema.sql`: Initial outcome tables.
- `migrations/002_model_registry.sql`: Initial model registry.
- `migrations/003_evaluation_tables.sql`: Walk-forward folds.
- `migrations/004_baselines.sql`: Frequency & Markov baselines.
- `migrations/005_recurrent.sql`: HMM & ESN tables.
- `migrations/006_feature_store.sql`: 17-dimensional causal feature vectors.
- `migrations/007_transformer.sql`: Neural Transformer configurations.
- `migrations/008_ssm.sql`: S4D & Mamba state monitoring.
- `migrations/009_meta_learning.sql`: Environment fingerprints & drift logs.
- `migrations/010_ensemble_observations.sql`: Dynamic ensemble observations.
- `migrations/011_autonomous_evolution.sql`: Hypotheses, candidates, experiments.
- `migrations/012_production_integration.sql`: Production integration harmonization.
