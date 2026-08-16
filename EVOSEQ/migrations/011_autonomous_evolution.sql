-- EVOSEQ Migration 011: Autonomous Evolution & Meta-Architecture Search Tables

CREATE TABLE IF NOT EXISTS research_hypotheses (
    id BIGSERIAL PRIMARY KEY,
    hypothesis_code TEXT UNIQUE,
    category TEXT NOT NULL,
    parent_model_id BIGINT,
    description TEXT NOT NULL,
    configuration JSONB NOT NULL,
    expected_effect TEXT,
    priority DOUBLE PRECISION DEFAULT 0.0,
    budget INTEGER DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'PENDING',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_candidates (
    id BIGSERIAL PRIMARY KEY,
    candidate_code TEXT UNIQUE,
    hypothesis_id BIGINT REFERENCES research_hypotheses(id),
    parent_model_id BIGINT,
    generation INTEGER NOT NULL,
    family TEXT NOT NULL,
    configuration JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'CANDIDATE',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiment_results (
    id BIGSERIAL PRIMARY KEY,
    candidate_id BIGINT REFERENCES model_candidates(id),
    fold INTEGER,
    seed INTEGER,
    horizon INTEGER DEFAULT 1,
    log_loss DOUBLE PRECISION,
    brier_score DOUBLE PRECISION,
    accuracy DOUBLE PRECISION,
    calibration_error DOUBLE PRECISION,
    null_p_value DOUBLE PRECISION,
    runtime_seconds DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hypotheses_code ON research_hypotheses(hypothesis_code);
CREATE INDEX IF NOT EXISTS idx_candidates_code ON model_candidates(candidate_code);
CREATE INDEX IF NOT EXISTS idx_candidates_gen ON model_candidates(generation);
