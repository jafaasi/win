-- EVOSEQ Migration 003: Model Genealogy Table

CREATE TABLE IF NOT EXISTS model_genealogy (
    id BIGSERIAL PRIMARY KEY,
    model_version_id BIGINT NOT NULL REFERENCES model_versions(id),
    parent_model_version_id BIGINT REFERENCES model_versions(id),
    generation INTEGER NOT NULL,
    mutation JSONB NOT NULL DEFAULT '{}',
    selection_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_genealogy_model ON model_genealogy(model_version_id);
CREATE INDEX IF NOT EXISTS idx_genealogy_generation ON model_genealogy(generation);
