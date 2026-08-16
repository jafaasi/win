-- EVOSEQ Migration 004: Model Runtime State Persistence Table

CREATE TABLE IF NOT EXISTS model_runtime_state (
    model_version_id BIGINT PRIMARY KEY REFERENCES model_versions(id),
    last_sequence_no BIGINT NOT NULL,
    state_path TEXT,
    optimizer_state_path TEXT,
    feature_state JSONB DEFAULT '{}',
    runtime_metadata JSONB DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_runtime_seq ON model_runtime_state(last_sequence_no);
