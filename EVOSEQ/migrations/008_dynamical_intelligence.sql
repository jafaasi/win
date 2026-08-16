-- EVOSEQ Migration 008: Hidden-State & Dynamical-System Intelligence Tables

CREATE TABLE IF NOT EXISTS hidden_state_experiments (
    id BIGSERIAL PRIMARY KEY,
    model_version_id BIGINT REFERENCES model_versions(id),
    generator_type TEXT NOT NULL,
    observation_count BIGINT,
    state_dimension INTEGER,
    state_recovery_score DOUBLE PRECISION,
    parameter_recovery_score DOUBLE PRECISION,
    predictive_score DOUBLE PRECISION,
    runtime_seconds DOUBLE PRECISION,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hidden_exp_gen ON hidden_state_experiments(generator_type);

CREATE TABLE IF NOT EXISTS environment_fingerprints (
    sequence_no BIGINT PRIMARY KEY,
    entropy DOUBLE PRECISION,
    conditional_entropy_1 DOUBLE PRECISION,
    conditional_entropy_2 DOUBLE PRECISION,
    information_gain_1 DOUBLE PRECISION,
    information_gain_2 DOUBLE PRECISION,
    lz_complexity DOUBLE PRECISION,
    lz_zscore DOUBLE PRECISION,
    autocorrelation_1 DOUBLE PRECISION,
    autocorrelation_2 DOUBLE PRECISION,
    drift_score DOUBLE PRECISION,
    recurrence_rate DOUBLE PRECISION,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
