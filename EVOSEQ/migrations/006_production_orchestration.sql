-- EVOSEQ Migration 006: Production Orchestration & Continuous Learning Tables

CREATE TABLE IF NOT EXISTS system_events (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    sequence_no BIGINT,
    model_version_id BIGINT,
    payload JSONB NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'completed',
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_system_events_sequence ON system_events(sequence_no);
CREATE UNIQUE INDEX IF NOT EXISTS idx_event_idempotency ON system_events(event_type, sequence_no, model_version_id);

CREATE TABLE IF NOT EXISTS worker_state (
    worker_name TEXT PRIMARY KEY,
    last_processed_sequence BIGINT DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'idle',
    records_processed BIGINT DEFAULT 0,
    last_success_at TIMESTAMPTZ,
    last_failure_at TIMESTAMPTZ,
    last_error TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS champion_health (
    id BIGSERIAL PRIMARY KEY,
    model_version_id BIGINT NOT NULL REFERENCES model_versions(id),
    sequence_no BIGINT NOT NULL,
    health_score DOUBLE PRECISION,
    drift_score DOUBLE PRECISION,
    calibration_error DOUBLE PRECISION,
    disagreement DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_champ_health_seq ON champion_health(sequence_no);
