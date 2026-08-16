-- EVOSEQ Migration 010: Ensemble Observations Table

CREATE TABLE IF NOT EXISTS ensemble_observations (
    id BIGSERIAL PRIMARY KEY,
    sequence_no BIGINT NOT NULL,
    environment JSONB NOT NULL,
    model_predictions JSONB NOT NULL,
    model_weights JSONB NOT NULL,
    ensemble_prediction JSONB NOT NULL,
    actual_digit SMALLINT,
    ensemble_log_loss DOUBLE PRECISION,
    disagreement DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ensemble_obs_seq ON ensemble_observations(sequence_no);
