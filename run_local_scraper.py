#!/usr/bin/env python3
"""
Local Scraper Runner - Free Tier Optimized

This script runs the data collection scraper on your local machine.
It stores outcomes to your cloud database (Supabase/PostgreSQL).
The AI prediction engine runs separately via local_ai_engine.py

Usage:
    python3 run_local_scraper.py           # Run continuous scraper
    python3 run_local_scraper.py --once    # Run single test cycle
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.scraper import run_scraper_daemon

async def main():
    print("🏠 Starting Local Data Collection Scraper")
    print("💾 Free Tier Optimized: Stores to cloud database")
    print("🧠 AI Engine: Run separately via 'python3 backend/local_ai_engine.py'")
    print("⏱️  Poll interval: 2.0 seconds for database efficiency")
    print("-" * 50)
    
    # Check for --once flag
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        print("🧪 Running single test cycle (15 seconds)...")
        await run_scraper_daemon(max_duration_seconds=15)
    else:
        print("🔄 Starting continuous scraper (infinite)...")
        print("Press Ctrl+C to stop")
        try:
            await run_scraper_daemon(max_duration_seconds=999999999)
        except KeyboardInterrupt:
            print("\n✅ Scraper stopped by user")

if __name__ == "__main__":
    asyncio.run(main())