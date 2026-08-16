-- EVOSEQ Migration 012: Production Integration Harmonization

-- 1. Ensure draws table has all necessary columns
ALTER TABLE draws ADD COLUMN IF NOT EXISTS color VARCHAR(32);
ALTER TABLE draws ADD COLUMN IF NOT EXISTS size VARCHAR(32);
ALTER TABLE draws ADD COLUMN IF NOT EXISTS parity SMALLINT;

-- 2. Ensure prediction_logs table has multi-horizon columns
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS h1_probs JSONB;
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS h2_probs JSONB;
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS h3_probs JSONB;
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS model_disagreement DOUBLE PRECISION;
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS aleatoric_entropy DOUBLE PRECISION;

-- 3. Ensure prediction_audit table has full EVOSEQ columns
ALTER TABLE prediction_audit ADD COLUMN IF NOT EXISTS feature_version VARCHAR(32) DEFAULT 'v17';
ALTER TABLE prediction_audit ADD COLUMN IF NOT EXISTS ensemble_weights JSONB;
ALTER TABLE prediction_audit ADD COLUMN IF NOT EXISTS brier_score DOUBLE PRECISION;
ALTER TABLE prediction_audit ADD COLUMN IF NOT EXISTS log_loss DOUBLE PRECISION;

-- 4. Create indexes for high-throughput temporal lookups
CREATE INDEX IF NOT EXISTS idx_draws_issue_num ON draws(issue_number);
CREATE INDEX IF NOT EXISTS idx_pred_logs_issue ON prediction_logs(issue_number);
