# Free Tier Optimized Setup Guide

## 🎯 **Architecture Overview**

This setup is optimized for **Render Free Tier** usage by moving all intensive processing to your local machine while using Render only for lightweight database operations.

### **Current Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                  YOUR LOCAL MACHINE                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  1. Local Scraper (run_local_scraper.py)              │  │
│  │     - Polls WinGo API every 1.5s                       │  │
│  │     - Stores outcomes to cloud database                │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  2. Local AI Engine (backend/local_ai_engine.py)       │  │
│  │     - Enhanced EVOSEQ prediction engine                │  │
│  │     - All ML processing runs locally                   │  │
│  │     - Stores predictions to cloud database            │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP (Database Operations)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              RENDER FREE TIER (Lightweight)                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Database Gateway (backend/server.py)                  │  │
│  │     - Lightweight API for database operations           │  │
│  │     - GET /api/state (read predictions)                │  │
│  │     - POST /api/outcomes (store outcomes)              │  │
│  │     - NO ML processing, NO scraping                    │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  PostgreSQL Database (Supabase)                       │  │
│  │     - Stores historical outcomes                      │  │
│  │     - Stores AI predictions                            │  │
│  │     - Shared data storage                             │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP (API Gateway)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   VERCEL FRONTEND                            │
│  React Dashboard                                           │
│  - Displays predictions from database                      │
│  - Shows live game history                                 │
│  - Manages betting interface                               │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 **Setup Instructions**

### **1. Environment Configuration**

Set up your environment variables in `backend/.env`:

```bash
# Database Connection (Required)
DATABASE_URL=postgresql://your-database-url

# Optional: Render-specific
RENDER_ENV=production
```

### **2. Local Components Setup**

#### **Option A: Run Both Components with One Command (Easiest)**

**Start Both Components:**
```bash
cd /Users/jaf/win
./run_all_local.sh
```

This script will:
- Automatically activate the virtual environment
- Install/update dependencies
- Start both the scraper and AI engine
- Provide process IDs for management

#### **Option B: Run Both Components Separately**

**Terminal 1 - Start Local Scraper:**
```bash
cd /Users/jaf/win
source .venv/bin/activate
python3 run_local_scraper.py
```

**Terminal 2 - Start Local AI Engine:**
```bash
cd /Users/jaf/win
source .venv/bin/activate
python3 backend/local_ai_engine.py
```

#### **Option C: Run Scraper in Test Mode**
```bash
cd /Users/jaf/win
source .venv/bin/activate
python3 run_local_scraper.py --once
```

### **3. Render Deployment**

Deploy the lightweight database gateway to Render:

```bash
# Install Render CLI (if not installed)
npm install -g @render/cli

# Deploy to Render
render deploy
```

Or use the `render.yaml` configuration for automatic deployment.

## 🔧 **Component Details**

### **Convenience Script (`run_all_local.sh`)**
**Purpose**: Start both local components with a single command  
**Features**:
- Automatically activates virtual environment
- Installs/updates dependencies
- Starts both scraper and AI engine
- Provides process IDs for management
- Easy Ctrl+C to stop both

**Usage**:
```bash
cd /Users/jaf/win
./run_all_local.sh
```

### **Local Scraper (`run_local_scraper.py`)**
- **Purpose**: Collect historical outcomes from WinGo API
- **Frequency**: Polls every 1.5 seconds
- **Storage**: Saves to cloud database (Supabase/PostgreSQL)
- **Resource Usage**: Minimal (HTTP requests only)
- **When to Run**: Keep running continuously to collect data

### **Local AI Engine (`backend/local_ai_engine.py`)**
- **Purpose**: Generate predictions using enhanced EVOSEQ
- **Processing**: All ML runs locally on your machine
- **Storage**: Saves predictions to cloud database
- **Resource Usage**: Higher (PyTorch, neural networks)
- **When to Run**: Keep running for continuous predictions

### **Render Database Gateway (`backend/server.py`)**
- **Purpose**: Lightweight API for database operations
- **Endpoints**:
  - `GET /api/state` - Read latest prediction
  - `POST /api/outcomes` - Store new outcomes
- **Resource Usage**: Minimal (database queries only)
- **Free Tier**: Optimized to stay within limits

## 📊 **Free Tier Optimization**

### **Render Free Tier Limits**
- **RAM**: 512 MB
- **CPU**: Shared
- **Build Time**: 15 minutes
- **Sleep**: After 15 minutes of inactivity

### **Optimizations Applied**
1. **No ML Processing**: All intensive computation moved to local machine
2. **Lightweight API**: Only database read/write operations
3. **No Background Workers**: Removed scraper daemon from Render
4. **Health Check**: `/healthz` endpoint to prevent sleep
5. **Auto Deploy Disabled**: Prevents unnecessary rebuilds

### **Expected Resource Usage**
- **Render**: <100 MB RAM, minimal CPU
- **Local Machine**: Higher usage (ML processing)
- **Database**: Standard query load

## 🎮 **Usage Workflow**

### **Daily Operation**

1. **Start Local Scraper** (Terminal 1):
   ```bash
   python3 run_local_scraper.py
   ```
   This will continuously collect outcomes and store them to the database.

2. **Start Local AI Engine** (Terminal 2):
   ```bash
   python3 backend/local_ai_engine.py
   ```
   This will continuously generate predictions and store them to the database.

3. **Access Frontend**:
   - Vercel frontend automatically fetches from Render database gateway
   - Displays predictions and live game data

### **Verification**

Check that all components are working:

1. **Scraper**: Look for "Outcome #XXX: Y | Total collected: Z" messages
2. **AI Engine**: Look for "✨ PREDICTION: Big/Small (X%) for Issue #XXX" messages
3. **Render**: Check health endpoint: `https://your-app.onrender.com/healthz`
4. **Frontend**: Visit your Vercel URL and check for live predictions

## 🔍 **Troubleshooting**

### **Scraper Not Collecting Data**
- Check internet connection
- Verify WinGo API is accessible
- Check database connection string

### **AI Engine Not Generating Predictions**
- Ensure scraper is collecting data first
- Check database has sufficient history (need 10+ outcomes)
- Verify Python dependencies are installed

### **Render Gateway Issues**
- Check Render logs for errors
- Verify DATABASE_URL is set correctly
- Ensure database is accessible

### **Frontend Not Updating**
- Check Render gateway is running
- Verify API rewrites in vercel.json
- Check browser console for errors

## 📈 **Performance Monitoring**

### **Key Metrics to Monitor**
- **Scraper**: Outcomes collected per hour
- **AI Engine**: Predictions generated per cycle
- **Render**: Response time, error rate
- **Database**: Query performance, storage growth

### **Expected Performance**
- **Scraper**: ~120 outcomes per hour (one every 30s)
- **AI Engine**: ~120 predictions per hour
- **Render**: <100ms response time
- **Database**: Minimal query load

## 💡 **Cost Optimization**

### **Current Setup Costs**
- **Render**: Free (within limits)
- **Vercel**: Free (within limits)
- **Database**: Free (Supabase free tier) or paid if exceeding limits
- **Local Machine**: Your existing hardware

### **When to Upgrade**
- **Render**: If you need more RAM/CPU for API operations
- **Database**: If storage exceeds free tier limits
- **Local Machine**: If ML processing is too slow

## 🔄 **Backup Strategy**

### **Data Backup**
- **Database**: Supabase handles automatic backups
- **Local Models**: Brain state saved to `backend/brain_*.pt` files
- **Configuration**: Environment variables in `.env` files

### **Recovery**
1. **Database**: Restore from Supabase dashboard
2. **Models**: Copy brain state files to new machine
3. **Configuration**: Copy `.env` files

## 🎯 **Next Steps**

1. **Deploy to Render**: Push changes and deploy using `render.yaml`
2. **Test Locally**: Run both local components and verify operation
3. **Monitor Performance**: Check all components are working correctly
4. **Optimize Further**: Adjust parameters based on performance metrics

This setup gives you maximum ML capability while staying within Render free tier limits by leveraging your local machine's processing power.