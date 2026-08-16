-- EVOSEQ Migration 009: Model Scores Table

CREATE TABLE IF NOT EXISTS model_scores (
    id BIGSERIAL PRIMARY KEY,
    model_version_id BIGINT NOT NULL REFERENCES model_versions(id),
    fold_id INTEGER NOT NULL,
    horizon INTEGER NOT NULL DEFAULT 1,
    accuracy DOUBLE PRECISION,
    log_loss DOUBLE PRECISION,
    brier_score DOUBLE PRECISION,
    calibration_error DOUBLE PRECISION,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_scores_ver ON model_scores(model_version_id);
CREATE INDEX IF NOT EXISTS idx_model_scores_fold ON model_scores(fold_id);
