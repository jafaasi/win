-- EVOSEQ Migration 001: Initial Schema

CREATE TABLE IF NOT EXISTS outcomes (
    id BIGSERIAL PRIMARY KEY,
    sequence_no BIGINT NOT NULL UNIQUE,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    digit SMALLINT NOT NULL CHECK (digit >= 0 AND digit <= 9),
    size SMALLINT NOT NULL CHECK (size IN (0, 1)),
    color SMALLINT NOT NULL CHECK (color IN (0, 1, 2)),
    parity SMALLINT NOT NULL CHECK (parity IN (0, 1)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_outcomes_sequence ON outcomes(sequence_no);
CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp ON outcomes(timestamp_utc);

CREATE TABLE IF NOT EXISTS model_versions (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}',
    training_start_sequence BIGINT,
    training_end_sequence BIGINT,
    validation_accuracy DOUBLE PRECISION,
    validation_log_loss DOUBLE PRECISION,
    validation_brier DOUBLE PRECISION,
    status TEXT NOT NULL CHECK (status IN ('candidate', 'challenger', 'champion', 'retired')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(model_name, version)
);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    model_version_id BIGINT REFERENCES model_versions(id),
    sequence_no BIGINT NOT NULL,
    probability_vector DOUBLE PRECISION[] NOT NULL,
    predicted_class SMALLINT,
    actual_class SMALLINT,
    log_loss DOUBLE PRECISION,
    brier_score DOUBLE PRECISION,
    entropy DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_sequence ON predictions(sequence_no);

CREATE TABLE IF NOT EXISTS model_events (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    model_version_id BIGINT REFERENCES model_versions(id),
    details JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    id BIGSERIAL PRIMARY KEY,
    sequence_no BIGINT NOT NULL,
    features JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
