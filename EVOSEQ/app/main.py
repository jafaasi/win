import sys
import os
import argparse

# Ensure EVOSEQ root is on python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.stream import get_latest_sequence, ingest_outcomes_batch
from app.evolution.orchestrator import daily_evolution
from app.evolution.registry import ModelRegistry

def main():
    parser = argparse.ArgumentParser(description="EVOSEQ Sequence Intelligence Research Engine CLI")
    parser.add_argument("--evolve", action="store_true", help="Trigger an immediate evolution cycle")
    parser.add_argument("--status", action="store_true", help="Display current champion and population registry")
    parser.add_argument("--seed-synthetic", type=int, default=0, help="Seed database with N synthetic pseudo-random outcomes for testing")
    args = parser.parse_args()

    registry = ModelRegistry()

    if args.seed_synthetic > 0:
        import random
        from datetime import datetime
        print(f"🌱 Generating {args.seed_synthetic} synthetic sequence outcomes...")
        start_seq = get_latest_sequence() + 1
        synthetic_batch = []
        for i in range(args.seed_synthetic):
            digit = random.choices(range(10), weights=[0.12, 0.08, 0.11, 0.09, 0.10, 0.13, 0.07, 0.11, 0.09, 0.10])[0]
            synthetic_batch.append({
                "sequence_no": start_seq + i,
                "timestamp_utc": datetime.utcnow(),
                "digit": digit
            })
        inserted = ingest_outcomes_batch(synthetic_batch)
        print(f"✅ Ingested {inserted} synthetic outcomes into database.")

    if args.evolve:
        daily_evolution(last_seq_cursor=0)

    if args.status or (not args.evolve and not args.seed_synthetic):
        champion = registry.get_champion()
        summary = registry.get_population_summary()
        latest_seq = get_latest_sequence()
        print("\n==================================================")
        print("          🧠 EVOSEQ RESEARCH CORE STATUS         ")
        print("==================================================")
        print(f"Total Observations Recorded : {latest_seq}")
        print(f"Current Active Champion     : {champion['version'] if champion else 'None (Run --evolve)'}")
        if champion:
            print(f"Champion Validation Score   : {champion.get('validation_accuracy')}")
            print(f"Champion Brier Score        : {champion.get('validation_brier')}")
        print(f"Population - Tested         : {summary['models_tested']}")
        print(f"Population - Challengers    : {summary['active_challengers']}")
        print(f"Population - Retired        : {summary['retired_models']}")
        print("==================================================\n")

if __name__ == "__main__":
    main()
