# EVOSEQ Deployment & Operational Guide

EVOSEQ is engineered for zero-downtime operations across **Vercel** and **Render** connected to **PostgreSQL**.

---

## 🚀 Deployment Topologies

1. **Frontend (Vercel)**:
   - Framework: React + Vite
   - Routing: `vercel.json` rewrites `/api/draws` to live feeds and `/api/(.*)` to serverless Python handlers.
2. **Backend Web Service & Worker (Render)**:
   - Runtime: Python 3.10+
   - Web Server: `uvicorn server:app --host 0.0.0.0 --port $PORT`
   - Background Daemons: `run_scraper_daemon` (polls every 10s), `keep_alive_self_pinger` (pings `/healthz` every 8 minutes to prevent free-tier idling).
3. **Database (PostgreSQL / Supabase)**:
   - Connection Pooling: SQLAlchemy `NullPool` with connection timeouts for resilient serverless operation.

---

## 🔒 Environment Variables

| Variable | Description | Required |
| :--- | :--- | :--- |
| `DATABASE_URL` | PostgreSQL connection string (`postgresql://...`) | Yes (production) |
| `PORT` | HTTP port on Render (default: `8080`) | Render-assigned |
| `RENDER_EXTERNAL_URL` | Public domain for auto-keepalive pinger | Optional |
