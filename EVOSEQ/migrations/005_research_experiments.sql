-- EVOSEQ Migration 005: Research Experiments Table

CREATE TABLE IF NOT EXISTS research_experiments (
    id BIGSERIAL PRIMARY KEY,
    experiment_type TEXT NOT NULL,
    model_version_id BIGINT REFERENCES model_versions(id),
    null_model TEXT,
    sample_size BIGINT,
    observed_score DOUBLE PRECISION,
    null_mean DOUBLE PRECISION,
    null_std DOUBLE PRECISION,
    p_value DOUBLE PRECISION,
    correction_method TEXT,
    test_range_start BIGINT,
    test_range_end BIGINT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_research_exp_type ON research_experiments(experiment_type);
CREATE INDEX IF NOT EXISTS idx_research_model_id ON research_experiments(model_version_id);
