#!/usr/bin/env python3
"""
Daily Evolution System
Combined script for daily EVOSEQ retraining and data cleanup
"""

import sys
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(backend_dir)
sys.path.append(os.path.dirname(backend_dir))


def run_daily_evolution():
    """Run the complete daily evolution cycle"""
    logger.info("=" * 70)
    logger.info("STARTING DAILY EVOLUTION SYSTEM")
    logger.info(f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info("=" * 70)
    
    # Step 1: Run evolving intelligence
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: EVOLVING INTELLIGENCE")
    logger.info("=" * 70)
    
    try:
        from evolving_intelligence import run_evolutionary_cycle
        evolution_success = run_evolutionary_cycle()
        
        if evolution_success:
            logger.info("✅ Evolving intelligence cycle completed successfully")
        else:
            logger.error("❌ Evolving intelligence cycle failed")
            
    except Exception as e:
        logger.error(f"Error in evolving intelligence: {e}")
        evolution_success = False
    
    # Step 2: Run data cleanup
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: DATA CLEANUP")
    logger.info("=" * 70)
    
    try:
        from data_cleanup import run_cleanup
        cleanup_success = run_cleanup()
        
        if cleanup_success:
            logger.info("✅ Data cleanup completed successfully")
        else:
            logger.error("❌ Data cleanup failed")
            
    except Exception as e:
        logger.error(f"Error in data cleanup: {e}")
        cleanup_success = False
    
    # Step 3: Summary
    logger.info("\n" + "=" * 70)
    logger.info("DAILY EVOLUTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Evolving Intelligence: {'✅ SUCCESS' if evolution_success else '❌ FAILED'}")
    logger.info(f"Data Cleanup: {'✅ SUCCESS' if cleanup_success else '❌ FAILED'}")
    
    if evolution_success and cleanup_success:
        logger.info("\n🎉 DAILY EVOLUTION COMPLETED SUCCESSFULLY")
        return True
    else:
        logger.error("\n⚠️ DAILY EVOLUTION COMPLETED WITH ERRORS")
        return False


if __name__ == "__main__":
    success = run_daily_evolution()
    sys.exit(0 if success else 1)
