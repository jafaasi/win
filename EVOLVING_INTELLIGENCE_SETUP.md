# Evolving Intelligence System Setup

## Overview
True evolving intelligence system that daily feeds WinGo data from Supabase to EVOSEQ, retrains models, and automatically cleans up old data.

## Components

### 1. Daily Data Feeding (`backend/daily_learning.py`)
- Fetches recent data from Supabase (last 7 days)
- Processes data for EVOSEQ training format
- Trains EVOSEQ models with fresh data
- Saves training metrics to database

### 2. Data Cleanup (`backend/data_cleanup.py`)
- Automatically deletes data older than 2 days
- Cleans up prediction logs older than 7 days
- Provides storage statistics
- Prevents database bloat

### 3. Evolving Intelligence (`backend/evolving_intelligence.py`)
- Comprehensive model retraining with fresh data
- Retrains EVOSEQ ensemble with updated weights
- Tracks model versions and performance
- Compares new vs old accuracy

### 4. Daily Evolution (`backend/daily_evolution.py`)
- Combined script that runs all daily tasks
- Runs evolving intelligence cycle
- Runs data cleanup
- Provides summary of results

## Setup on AWS

### Step 1: Make Scripts Executable
```bash
cd ~/win
chmod +x backend/daily_learning.py
chmod +x backend/data_cleanup.py
chmod +x backend/evolving_intelligence.py
chmod +x backend/daily_evolution.py
chmod +x setup_cron.sh
```

### Step 2: Test Scripts Manually
```bash
# Test data cleanup
cd ~/win
source .venv/bin/activate
python backend/data_cleanup.py

# Test daily learning
python backend/daily_learning.py

# Test evolving intelligence
python backend/evolving_intelligence.py

# Test combined daily evolution
python backend/daily_evolution.py
```

### Step 3: Set Up Cron Job
```bash
# Run setup script
./setup_cron.sh

# Or manually edit crontab
crontab -e
```

**Add this line to run daily at 2 AM UTC:**
```cron
0 2 * * * cd /home/ubuntu/win && /home/ubuntu/win/.venv/bin/python backend/daily_evolution.py >> /home/ubuntu/win/logs/daily_evolution.log 2>&1
```

### Step 4: Create Log Directory
```bash
mkdir -p ~/win/logs
```

### Step 5: Enable Cron Service
```bash
sudo systemctl enable cron
sudo systemctl start cron
```

## How It Works

### Daily Evolution Cycle
1. **Data Fetching**: Fetches last 7 days of WinGo data from Supabase
2. **Sequence Preparation**: Converts data to training sequences
3. **Model Retraining**: Retrains EVOSEQ ensemble with fresh data
4. **Ensemble Creation**: Creates new ensemble with updated weights
5. **Performance Tracking**: Saves model version and accuracy
6. **Data Cleanup**: Deletes data older than 2 days

### Continuous Learning
- **Daily Retraining**: Models learn from recent patterns
- **Version Tracking**: Each training cycle creates a new model version
- **Performance Monitoring**: Tracks accuracy improvements
- **Automatic Cleanup**: Prevents database bloat

### Storage Management
- **Automatic Cleanup**: 2-day data retention
- **Log Cleanup**: 7-day log retention
- **Storage Monitoring**: Provides database statistics
- **Prevents Overflow**: Automatic deletion of old data

## Monitoring

### Check Evolution Logs
```bash
tail -f ~/win/logs/daily_evolution.log
```

### Check Model Versions
```bash
# Connect to database
psql $DATABASE_URL

# Query model versions
SELECT * FROM model_versions ORDER BY training_date DESC LIMIT 10;
```

### Check Training Metrics
```bash
# Query training metrics
SELECT * FROM training_metrics ORDER BY training_date DESC LIMIT 10;
```

### Check Storage Statistics
```bash
# Run cleanup script to see current stats
python backend/data_cleanup.py
```

## Customization

### Change Retention Period
Edit `backend/data_cleanup.py`:
```python
# Change from 2 days to desired period
deleted = delete_old_data(days_old=7)  # Keep 7 days instead of 2
```

### Change Training Frequency
Edit crontab:
```bash
# Run every 12 hours instead of daily
0 */12 * * * cd /home/ubuntu/win && /home/ubuntu/win/.venv/bin/python backend/daily_evolution.py >> /home/ubuntu/win/logs/daily_evolution.log 2>&1
```

### Change Training Data Window
Edit `backend/evolving_intelligence.py`:
```python
# Change from 7 days to 14 days
training_data = fetch_comprehensive_training_data(days=14)
```

## Troubleshooting

### Script Not Running
```bash
# Check cron service
sudo systemctl status cron

# Check crontab
crontab -l

# Check logs
tail -50 ~/win/logs/daily_evolution.log
```

### Database Connection Issues
```bash
# Test database connection
python -c "from backend.database import engine; print('Connected')"
```

### Memory Issues
```bash
# Monitor memory usage
free -h

# Check running processes
ps aux | grep python
```

## Benefits

### True Evolving Intelligence
- **Daily Learning**: Models learn from fresh data daily
- **Pattern Adaptation**: Adapts to changing patterns
- **Performance Tracking**: Monitors accuracy improvements
- **Version Control**: Tracks model evolution

### Automatic Maintenance
- **Storage Management**: Automatic cleanup of old data
- **Log Management**: Automatic log cleanup
- **Error Handling**: Comprehensive error handling and logging
- **Monitoring**: Detailed logs for troubleshooting

### Cost Efficiency
- **Storage Optimization**: Only keeps recent data
- **Resource Efficient**: Scheduled operations
- **No Manual Intervention**: Fully automated
- **Scalable**: Handles increasing data volumes

## Next Steps

1. **Test Scripts**: Run each script manually to verify functionality
2. **Set Up Cron**: Configure cron job for daily execution
3. **Monitor Logs**: Check logs for successful execution
4. **Track Performance**: Monitor model accuracy improvements
5. **Adjust Parameters**: Fine-tune retention periods and frequency