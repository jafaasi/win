-- EVOSEQ Migration 007: Meta-Learning & Research Intelligence Tables

CREATE TABLE IF NOT EXISTS meta_experiments (
    id BIGSERIAL PRIMARY KEY,
    model_version_id BIGINT REFERENCES model_versions(id),
    environment JSONB NOT NULL,
    model_descriptor JSONB NOT NULL,
    log_loss DOUBLE PRECISION,
    brier_score DOUBLE PRECISION,
    calibration_error DOUBLE PRECISION,
    stability_score DOUBLE PRECISION,
    null_advantage DOUBLE PRECISION,
    inference_latency DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_meta_exp_model ON meta_experiments(model_version_id);

CREATE TABLE IF NOT EXISTS research_questions (
    id BIGSERIAL PRIMARY KEY,
    question_code TEXT UNIQUE NOT NULL,
    question TEXT NOT NULL,
    hypothesis TEXT,
    status TEXT NOT NULL DEFAULT 'OPEN',
    priority DOUBLE PRECISION DEFAULT 0.5,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS research_hypotheses (
    id BIGSERIAL PRIMARY KEY,
    question_id BIGINT REFERENCES research_questions(id),
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'PRE_REGISTERED', -- PRE_REGISTERED, SUPPORTED, WEAK_EVIDENCE, INCONCLUSIVE, REJECTED
    evidence_score DOUBLE PRECISION DEFAULT 0.0,
    evidence_summary JSONB DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hypo_status ON research_hypotheses(status);
