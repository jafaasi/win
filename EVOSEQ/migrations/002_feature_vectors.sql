-- EVOSEQ Migration 002: Feature Vectors Table

CREATE TABLE IF NOT EXISTS feature_vectors (
    id BIGSERIAL PRIMARY KEY,
    sequence_no BIGINT NOT NULL UNIQUE,
    window_size INTEGER NOT NULL,
    feature_vector DOUBLE PRECISION[] NOT NULL,
    digit_entropy DOUBLE PRECISION,
    conditional_entropy_1 DOUBLE PRECISION,
    conditional_entropy_2 DOUBLE PRECISION,
    conditional_entropy_3 DOUBLE PRECISION,
    information_gain_1 DOUBLE PRECISION,
    information_gain_2 DOUBLE PRECISION,
    information_gain_3 DOUBLE PRECISION,
    lz_complexity DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feature_sequence ON feature_vectors(sequence_no);
