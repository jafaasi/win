#!/usr/bin/env python3
"""
Robust Scraper for WinGo 30S
- Polls official API every 2 seconds with correct headers
- Stores outcomes in Supabase with correct schema
- Handles duplicates gracefully
- Comprehensive logging
"""

import requests
import time
import os
import sys
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scraper.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("❌ Missing SUPABASE_URL or SUPABASE_KEY in environment")
    sys.exit(1)

# Official API endpoint
API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"

# Headers that match browser requests exactly (from network capture)
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://bdgwin888.com",
    "Referer": "https://bdgwin888.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "sec-ch-ua": '"Chromium";v="120", "Not=A?Brand";v="8"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "cross-site",
}

def init_supabase():
    """Initialize Supabase client"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info(f"✅ Connected to Supabase: {SUPABASE_URL[:30]}...")
        return supabase
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        sys.exit(1)

def fetch_official_data():
    """Fetch latest data from official API"""
    try:
        params = {"ts": str(int(time.time() * 1000))}
        response = requests.get(API_URL, headers=HEADERS, params=params, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"⚠️ API returned status {response.status_code}")
            return None
        
        # Parse JSON - response format: {"data": {"list": [...]}}
        data = response.json()
        
        # Extract the list from nested structure
        if isinstance(data, dict) and "data" in data:
            result_list = data["data"].get("list", [])
        elif isinstance(data, list):
            result_list = data
        else:
            logger.warning(f"⚠️ Unexpected API response structure: {type(data)}")
            return None
        
        if not result_list:
            logger.warning("⚠️ Empty list from API")
            return None
            
        logger.info(f"✅ Fetched {len(result_list)} records from API")
        return result_list
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Parse error: {e}")
        return None

def parse_outcome(record):
    """Parse API record into our schema"""
    try:
        issue_number = record.get("issueNumber") or record.get("issue_number")
        number_str = record.get("number") or record.get("Number") or "0"
        color = record.get("color") or ""
        
        if not issue_number:
            return None
            
        number = int(number_str)
        size = 1 if number > 4 else 0  # BIG=1 (5-9), SMALL=0 (0-4)
        
        # Parse color
        color_code = 0  # default
        if "red" in color.lower():
            color_code = 1
        elif "green" in color.lower():
            color_code = 2
        elif "violet" in color.lower() or "blue" in color.lower():
            color_code = 3
            
        # Parity: 1=odd, 0=even
        parity = 1 if number % 2 == 1 else 0
        
        return {
            "sequence_no": issue_number,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "digit": number,
            "size": size,
            "color": color_code,
            "parity": parity
        }
    except Exception as e:
        logger.error(f"❌ Parse error for record {record}: {e}")
        return None

def save_outcome(supabase, outcome):
    """Save outcome to Supabase if not exists"""
    try:
        # Check if already exists
        existing = supabase.table("outcomes").select("id").eq("sequence_no", outcome["sequence_no"]).execute()
        
        if existing.data:
            logger.debug(f"⏭️  Duplicate: {outcome['sequence_no']}")
            return False
        
        # Insert new record
        result = supabase.table("outcomes").insert(outcome).execute()
        logger.info(f"💾 Saved: {outcome['sequence_no']} | Digit: {outcome['digit']} | Size: {'BIG' if outcome['size'] else 'SMALL'}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Save error: {e}")
        return False

def run_scraper():
    """Main scraper loop"""
    logger.info("🚀 Starting Robust Scraper...")
    logger.info(f"Supabase: {SUPABASE_URL[:30]}...")
    logger.info(f"API: {API_URL}")
    
    supabase = init_supabase()
    
    while True:
        try:
            # Fetch data from API
            records = fetch_official_data()
            
            if not records:
                logger.warning("⚠️ No data from API, waiting...")
                time.sleep(2)
                continue
            
            # Process records (newest first)
            new_count = 0
            for record in records:
                outcome = parse_outcome(record)
                if outcome:
                    if save_outcome(supabase, outcome):
                        new_count += 1
            
            if new_count > 0:
                logger.info(f"✅ Added {new_count} new outcomes")
            else:
                logger.info(f"⏸️  No new outcomes (total known: {len(records)})")
            
            # Wait before next poll
            time.sleep(2)
            
        except KeyboardInterrupt:
            logger.info("👋 Stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Cycle error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    run_scraper()
