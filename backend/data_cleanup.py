#!/usr/bin/env python3
"""
Automatic Data Cleanup for Supabase
Deletes data older than 2 days to manage storage
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(backend_dir, '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
except Exception as e:
    logger.warning(f"Could not load .env: {e}")

# Database connection
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    # Use hardcoded Supabase URL as fallback
    DATABASE_URL = "postgresql://postgres.zyryxnifpduwsulglhdq:JafAasi1517@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"

if DATABASE_URL:
    DATABASE_URL = DATABASE_URL.strip().strip('"').strip("'").strip()
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

logger.info(f"Connecting to database: {DATABASE_URL[:50]}...")


def delete_old_data(days_old=2):
    """Delete data older than specified days"""
    try:
        engine = create_engine(DATABASE_URL)
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        with engine.connect() as conn:
            # Count records to be deleted
            count_query = text("""
                SELECT COUNT(*) FROM outcomes
                WHERE timestamp_utc < :cutoff_date
            """)
            count_result = conn.execute(count_query, {"cutoff_date": cutoff_date})
            count = count_result.scalar()
            
            if count == 0:
                logger.info(f"No records older than {days_old} days found.")
                return 0
            
            logger.info(f"Found {count} records older than {days_old} days.")
            
            # Delete old records
            delete_query = text("""
                DELETE FROM outcomes
                WHERE timestamp_utc < :cutoff_date
            """)
            
            delete_result = conn.execute(delete_query, {"cutoff_date": cutoff_date})
            conn.commit()
            
            logger.info(f"Deleted {delete_result.rowcount} old records.")
            
            # Also clean up draws table
            draw_count_query = text("""
                SELECT COUNT(*) FROM draws
                WHERE created_at < :cutoff_date
            """)
            draw_count_result = conn.execute(draw_count_query, {"cutoff_date": cutoff_date})
            draw_count = draw_count_result.scalar()
            
            if draw_count > 0:
                delete_draws_query = text("""
                    DELETE FROM draws
                    WHERE created_at < :cutoff_date
                """)
                delete_draws_result = conn.execute(delete_draws_query, {"cutoff_date": cutoff_date})
                conn.commit()
                logger.info(f"Deleted {delete_draws_result.rowcount} old draw records.")
            
            return delete_result.rowcount
            
    except Exception as e:
        logger.error(f"Error deleting old data: {e}")
        return -1


def cleanup_prediction_logs(days_old=7):
    """Clean up old prediction logs"""
    try:
        engine = create_engine(DATABASE_URL)
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        with engine.connect() as conn:
            # Check if prediction_logs table exists
            table_check = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'prediction_logs'
                )
            """)
            table_exists = conn.execute(table_check).scalar()
            
            if not table_exists:
                logger.info("prediction_logs table doesn't exist, skipping cleanup.")
                return 0
            
            # Delete old prediction logs
            delete_query = text("""
                DELETE FROM prediction_logs
                WHERE created_at < :cutoff_date
            """)
            
            delete_result = conn.execute(delete_query, {"cutoff_date": cutoff_date})
            conn.commit()
            
            logger.info(f"Deleted {delete_result.rowcount} old prediction log records.")
            return delete_result.rowcount
            
    except Exception as e:
        logger.error(f"Error cleaning prediction logs: {e}")
        return -1


def get_storage_stats():
    """Get current database storage statistics"""
    try:
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as conn:
            # Count records in each table
            outcomes_count = conn.execute(text("SELECT COUNT(*) FROM outcomes")).scalar()
            draws_count = conn.execute(text("SELECT COUNT(*) FROM draws")).scalar()
            
            # Get oldest record date
            oldest_date = conn.execute(text("""
                SELECT MIN(timestamp_utc) FROM outcomes
            """)).scalar()
            
            stats = {
                'outcomes_count': outcomes_count,
                'draws_count': draws_count,
                'oldest_date': oldest_date,
                'database_size': 'N/A'  # Would need additional queries for actual size
            }
            
            return stats
            
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        return None


def run_cleanup():
    """Main cleanup function"""
    logger.info("=" * 50)
    logger.info("Starting Data Cleanup")
    logger.info("=" * 50)
    
    # Get current stats
    logger.info("Current database statistics:")
    stats = get_storage_stats()
    if stats:
        logger.info(f"  Outcomes: {stats['outcomes_count']}")
        logger.info(f"  Draws: {stats['draws_count']}")
        logger.info(f"  Oldest record: {stats['oldest_date']}")
    
    # Delete 2-day old data
    logger.info("\nDeleting data older than 2 days...")
    deleted = delete_old_data(days_old=2)
    
    if deleted > 0:
        logger.info(f"Successfully deleted {deleted} records.")
    elif deleted == 0:
        logger.info("No old records to delete.")
    else:
        logger.error("Cleanup failed.")
    
    # Clean up prediction logs (7 days)
    logger.info("\nCleaning up prediction logs older than 7 days...")
    cleanup_prediction_logs(days_old=7)
    
    # Get updated stats
    logger.info("\nUpdated database statistics:")
    updated_stats = get_storage_stats()
    if updated_stats:
        logger.info(f"  Outcomes: {updated_stats['outcomes_count']}")
        logger.info(f"  Draws: {updated_stats['draws_count']}")
        logger.info(f"  Oldest record: {updated_stats['oldest_date']}")
    
    logger.info("=" * 50)
    logger.info("Data Cleanup Completed")
    logger.info("=" * 50)
    
    return deleted >= 0


if __name__ == "__main__":
    success = run_cleanup()
    sys.exit(0 if success else 1)
