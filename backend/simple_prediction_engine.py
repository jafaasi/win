#!/usr/bin/env python3
"""
Simple Prediction Engine - Minimal, Reliable, No Frills
Polls official API, generates predictions, saves to live_ui_state
"""

import os
import sys
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/simple_engine.log')
    ]
)
logger = logging.getLogger('SIMPLE_ENGINE')

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import requests
    from supabase import create_client
    from dotenv import load_dotenv
    load_dotenv()
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    sys.exit(1)


class SimplePredictionEngine:
    """Minimal prediction engine that just works."""
    
    def __init__(self):
        self.supabase = None
        self.last_issue = None
        self.last_prediction = None
        self.cycle_count = 0
        
        # Initialize Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env")
        
        self.supabase = create_client(supabase_url, supabase_key)
        logger.info("✅ Connected to Supabase")
        
        # Load historical data once
        self.history = self._load_history()
        logger.info(f"📚 Loaded {len(self.history)} historical records")
    
    def _load_history(self) -> List[Dict]:
        """Load historical outcomes from Supabase."""
        try:
            response = self.supabase.table("outcomes").select(
                "issue_number, number"
            ).order("issue_number", desc=True).limit(1000).execute()
            
            if response.data:
                return response.data
            return []
        except Exception as e:
            logger.warning(f"Could not load history: {e}")
            return []
    
    def _fetch_latest_from_api(self) -> Optional[Dict]:
        """Fetch latest issues from official API."""
        url = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
        params = {"ts": int(time.time() * 1000)}
        
        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Origin": "https://bdgwin888.com",
            "Referer": "https://bdgwin888.com/"
        }
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Parse API response (adjust based on actual structure)
            if isinstance(data, dict) and 'data' in data:
                issues = data['data']
            elif isinstance(data, list):
                issues = data
            else:
                logger.warning(f"Unexpected API response structure: {type(data)}")
                return None
            
            if not issues:
                return None
            
            # Return most recent issue
            latest = issues[0] if isinstance(issues, list) else issues
            return {
                'issue_number': str(latest.get('issueNumber', latest.get('issue_number', ''))),
                'number': int(latest.get('number', latest.get('num', 0)))
            }
            
        except Exception as e:
            logger.warning(f"API fetch failed: {e}")
            return None
    
    def _save_outcome_to_db(self, issue_data: Dict):
        """Save new outcome to database if not exists."""
        try:
            # Check if already exists
            existing = self.supabase.table("outcomes").select("id").eq(
                "issue_number", issue_data['issue_number']
            ).execute()
            
            if existing.data:
                return  # Already saved
            
            # Insert new outcome
            self.supabase.table("outcomes").insert({
                "issue_number": issue_data['issue_number'],
                "number": issue_data['number'],
                "created_at": datetime.utcnow().isoformat()
            }).execute()
            
            logger.info(f"💾 Saved outcome: {issue_data['issue_number']} = {issue_data['number']}")
            
            # Update history
            self.history.insert(0, issue_data)
            if len(self.history) > 1000:
                self.history = self.history[:1000]
                
        except Exception as e:
            logger.warning(f"Could not save outcome: {e}")
    
    def _calculate_next_issue(self, current_issue: str) -> str:
        """Calculate next issue number."""
        try:
            # Issue format: YYYYMMDDHHMMSSXXX
            # Increment by 1 (or appropriate interval)
            issue_num = int(current_issue)
            return str(issue_num + 1)
        except:
            return str(int(current_issue) + 1)
    
    def _generate_prediction(self, history: List[Dict], next_issue: str) -> Dict[str, Any]:
        """Generate prediction using simple ensemble."""
        
        # Default prediction if no history
        if len(history) < 10:
            return {
                'prediction': 'BIG',
                'probability': 0.50,
                'confidence': 0.50,
                'targetNum': 5,
                'hedgeNum': 3,
                'action': 'SKIP',
                'modelConsensus': 0.50
            }
        
        # Simple frequency-based prediction
        last_30 = [h['number'] for h in history[:30]]
        last_100 = [h['number'] for h in history[:100]]
        
        # Count BIG (5-9) vs SMALL (0-4)
        big_count_30 = sum(1 for n in last_30 if n >= 5)
        big_count_100 = sum(1 for n in last_100 if n >= 5)
        
        freq_30 = big_count_30 / len(last_30) if last_30 else 0.5
        freq_100 = big_count_100 / len(last_100) if last_100 else 0.5
        
        # Weighted average
        p_big = (freq_30 * 0.6) + (freq_100 * 0.4)
        
        # Determine prediction
        if p_big > 0.52:
            prediction = 'BIG'
            target_digits = [5, 6, 7, 8, 9]
        elif p_big < 0.48:
            prediction = 'SMALL'
            target_digits = [0, 1, 2, 3, 4]
        else:
            prediction = 'BIG' if p_big >= 0.5 else 'SMALL'
            target_digits = list(range(10))
        
        # Confidence based on deviation from 0.5
        confidence = abs(p_big - 0.5) * 2  # 0.0 to 1.0
        
        # Only act if confidence > 0.6
        action = 'ENTER' if confidence > 0.6 else 'SKIP'
        
        # Select target and hedge numbers
        targetNum = target_digits[0] if target_digits else 5
        hedgeNum = target_digits[1] if len(target_digits) > 1 else 3
        
        return {
            'prediction': prediction,
            'probability': round(p_big, 4),
            'confidence': round(confidence, 4),
            'targetNum': targetNum,
            'hedgeNum': hedgeNum,
            'action': action,
            'modelConsensus': round(p_big, 4),
            'strikeQuality': 'HIGH' if confidence > 0.7 else 'MEDIUM' if confidence > 0.5 else 'LOW'
        }
    
    def _save_prediction(self, next_issue: str, prediction: Dict):
        """Save prediction to live_ui_state table."""
        try:
            payload = {
                'next_issue': next_issue,
                'prediction': prediction['prediction'],
                'probability': prediction['probability'],
                'confidence': prediction['confidence'],
                'target_digit': prediction['targetNum'],
                'hedge_digit': prediction['hedgeNum'],
                'action': prediction['action'],
                'model_consensus': prediction['modelConsensus'],
                'strike_quality': prediction.get('strikeQuality', 'MEDIUM'),
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Check if prediction for this issue already exists
            existing = self.supabase.table("live_ui_state").select("id").eq(
                "next_issue", next_issue
            ).execute()
            
            if existing.data:
                # Update existing
                self.supabase.table("live_ui_state").update(payload).eq(
                    "next_issue", next_issue
                ).execute()
                logger.info(f"🔄 Updated prediction for {next_issue}")
            else:
                # Insert new
                self.supabase.table("live_ui_state").insert(payload).execute()
                logger.info(f"✨ Saved new prediction for {next_issue}")
                
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
    
    def run_cycle(self):
        """Run one prediction cycle."""
        self.cycle_count += 1
        
        # Fetch latest from API
        latest = self._fetch_latest_from_api()
        
        if not latest:
            logger.warning("⚠️ No data from API")
            time.sleep(2)
            return
        
        current_issue = latest['issue_number']
        current_number = latest['number']
        
        # Check if new outcome
        if current_issue != self.last_issue:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 Cycle #{self.cycle_count} | New Draw: {current_issue} (Num: {current_number})")
            
            # Save to database
            self._save_outcome_to_db(latest)
            
            # Resolve previous prediction if exists
            if self.last_prediction:
                actual_size = 'BIG' if current_number >= 5 else 'SMALL'
                was_correct = (self.last_prediction['prediction'] == actual_size)
                logger.info(f"📊 Previous prediction: {self.last_prediction['prediction']} | Actual: {actual_size} | {'✅ WIN' if was_correct else '❌ LOSS'}")
            
            # Calculate next issue
            next_issue = self._calculate_next_issue(current_issue)
            
            # Generate prediction
            prediction = self._generate_prediction(self.history, next_issue)
            
            # Save prediction
            self._save_prediction(next_issue, prediction)
            
            # Update state
            self.last_issue = current_issue
            self.last_prediction = prediction
            
            logger.info(f"🔮 Next Issue: {next_issue} | Prediction: {prediction['prediction']} ({prediction['action']})")
            logger.info(f"📈 Confidence: {prediction['confidence']:.2%} | Probability: {prediction['probability']:.2%}")
        else:
            # Same issue, just log
            logger.debug(f"Waiting for new issue... Current: {current_issue}")
        
        time.sleep(2)
    
    def run(self):
        """Main loop."""
        logger.info("🚀 Simple Prediction Engine Starting...")
        logger.info("📡 Polling official API every 2 seconds")
        logger.info("💾 Saving predictions to live_ui_state\n")
        
        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("👋 Shutting down...")
                break
            except Exception as e:
                logger.error(f"❌ Cycle error: {e}", exc_info=True)
                time.sleep(5)


if __name__ == "__main__":
    try:
        engine = SimplePredictionEngine()
        engine.run()
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        sys.exit(1)
