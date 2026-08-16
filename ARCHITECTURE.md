# EVOSEQ System Architecture

EVOSEQ is a continuous sequence intelligence and meta-learning research platform deployed on **Vercel (Frontend & Serverless API)** and **Render (24/7 Web Service & Background Workers)**, backed by **PostgreSQL (Supabase / Cloud Postgres)**.

---

## 🏛️ End-to-End Architecture

```
  ┌────────────────────────────────────────────────────────┐
  │                 CLIENT DASHBOARD (Vite)                │
  │                  Hosted on Vercel                      │
  │     (src/App.jsx, src/components/PredictionDisplay)    │
  └───────────────┬────────────────────────┬───────────────┘
                  │ /api/draws             │ /api/state (POST/GET)
                  ▼                        ▼
       ┌────────────────────┐   ┌─────────────────────────────┐
       │   WinGo API Proxy  │   │      Vercel Serverless      │
       │ (Ar-Lottery30S API)│   │       (api/index.py)        │
       └────────────────────┘   └──────────────┬──────────────┘
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │  PostgreSQL (Supabase / Render) │
                              │   50,000+ unbroken sequence rows│
                              └────────────────┬────────────────┘
                                               ▲
                                               │ Read / Sync / Verify
                                               ▼
                               ┌────────────────────────────────┐
                               │   Render 24/7 Web & Worker     │
                               │  (backend/server.py + scraper) │
                               │  FastAPI + Background Daemons  │
                               └────────────────┬───────────────┘
                                                ▲
                                                │ Plug-in Layer
                                                ▼
                               ┌────────────────────────────────┐
                               │      EVOSEQ ENGINE (v16.0)     │
                               │  Population + Dynamic Ensemble │
                               │   + Research Director Loop     │
                               └────────────────────────────────┘
```

---

## ⚡ Fast Path vs. Slow Path Decoupling

| Path | Components | Latency Target | Functions |
| :--- | :--- | :--- | :--- |
| **Fast Path** | `api/index.py`, `GET /api/state`, `GET /api/ensemble` | $< 25\text{ms}$ | Inference, multi-horizon softmax scaling, temperature calibration, uncertainty decomposition |
| **Slow Path** | `backend/scraper.py`, `backend/evoseq_loop.py`, `AutonomousResearchDirector` | Background ($10\text{s}$ poll, $00:00\text{ UTC}$ daily) | Ingestion deduplication, multi-dimensional drift detection, hypothesis generation, multi-seed temporal validation, surrogate null testing, promotion gating |

---

## 🔬 Multi-Horizon Predictions ($H_1, H_2, H_3$)

Every inference step produces calibrated probability distributions for:
- $H_1 = P(X_{t+1}=k \mid X_{1:t})$ (Next step)
- $H_2 = P(X_{t+2}=k \mid X_{1:t})$ (+2 steps)
- $H_3 = P(X_{t+3}=k \mid X_{1:t})$ (+3 steps)

Where $\sum_{k=0}^9 P(X=k) = 1.0$ and $P(X=k) \ge 0.0$.
