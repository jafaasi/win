# 🧠 EVOSEQ v1: Sequence Intelligence & PRNG Forensic Engine

EVOSEQ is an evolving sequence-analysis and predictability research platform built for analyzing non-stationary discrete streams, detecting distribution drift, evaluating sequence models, and running continuous population evolution.

## Architecture

```
EVOSEQ/
├── app/
│   ├── config.py         # Application configuration & thresholds
│   ├── database.py       # SQLAlchemy engine & session manager
│   ├── schemas.py        # Database ORM models
│   ├── ingestion/        # Incremental streaming cursor ingestion
│   ├── features/         # Empirical distributions, entropy, N-grams, run statistics
│   ├── models/           # SequenceModel ABC, UniformModel, MarkovModel
│   ├── evaluation/       # Multi-class Log-Loss, Brier score, ECE, walk-forward audit
│   ├── evolution/        # ModelRegistry, Jensen-Shannon Drift, Daily Orchestrator
│   └── main.py           # CLI entrypoint
├── migrations/           # PostgreSQL migration scripts
└── tests/                # Automated test suite
```

## Key Commands

```bash
# Seed 1,000 synthetic observations
python -m app.main --seed-synthetic 1000

# Run Daily Evolution Cycle (Challenger Spawning & Champion Selection)
python -m app.main --evolve

# Check Model Registry & Current Champion Status
python -m app.main --status

# Run Test Suite
pytest tests/
```
