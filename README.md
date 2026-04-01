# F-Score Prospect Engine

**ML-powered prospect matching for financial advisor platforms.**

A full-stack prototype demonstrating how machine learning can match financial advisors with high-conversion individual prospects — inspired by the advisor-prospect matching problem in wealthtech.

Built by **Anastasis Moraitis** · Georgia Tech '27 · Computer Engineering · [Live Demo →](https://anastmor.github.io/fscore-prospect-engine/frontend/index.html)

---

## Why FINNY

FINNY is building the **growth OS for independent financial advisors** — the infrastructure layer that handles everything from prospect discovery to client onboarding.

The numbers reveal the problem: advisors spend an average of **~58 hours per new client converted**, with conversion rates below **1%** on cold outreach. The matching problem is hard: 270M+ U.S. adults, sparse behavioral signals, and the window to reach a prospect is narrow — most advisor-seeking behavior clusters around a handful of life events that decay within 90 days.

This prototype explores the ML architecture behind that matching problem:

- **F-Score concept** — a calibrated 0–100 compatibility score per advisor-prospect pair, built on a 6-feature weighted ensemble with sigmoid calibration
- **Life event timing with exponential decay** — a liquidity event 7 days ago is ~15× more signal-rich than the same event 6 months ago (λ = 90-day decay constant)
- **Advisor-specific pairwise matching** — features are computed per advisor–prospect pair, not globally, so the same prospect can score differently for different advisors
- **Candidate retrieval at scale with ANN** — in production, FAISS/ScaNN narrows 270M profiles to ~10K candidates before the ranking model runs

---

## Live Demo

- **Frontend:** [https://anastmor.github.io/fscore-prospect-engine/frontend/index.html](https://anastmor.github.io/fscore-prospect-engine/frontend/index.html)
- **API Docs:** [https://fscore-prospect-engine.onrender.com/docs](https://fscore-prospect-engine.onrender.com/docs)

---

## Demo

![F-Score Engine Screenshot](https://img.shields.io/badge/status-working-brightgreen)

**Quick start:**
```bash
# Terminal 1 — Backend
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
open index.html    # or: python -m http.server 3000
```

Then open `http://localhost:3000` (or just the HTML file) in your browser.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              React Frontend (SPA)                │
│  Advisor Profile → Score Results → ML Explain    │
│  index.html — zero build step                    │
└──────────────────┬──────────────────────────────┘
                   │ fetch() → JSON
                   ▼
┌──────────────────────────────────────────────────┐
│              FastAPI Backend (:8000)              │
│                                                  │
│  POST /api/score ──────→ scoring.py              │
│  POST /api/explain/{id} → scoring.py             │
│  POST /api/stats ──────→ scoring.py              │
│  GET  /api/prospects ──→ data.py                 │
│  GET  /api/methodology → scoring.py              │
│  GET  /api/health                                │
│                                                  │
│  ┌────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │ models.py  │  │  data.py    │  │scoring.py │ │
│  │ Pydantic   │  │  Synthetic  │  │ ML core   │ │
│  │ schemas    │  │  200 prosps │  │ 6 features│ │
│  └────────────┘  └─────────────┘  └───────────┘ │
└──────────────────────────────────────────────────┘
```

## ML Pipeline

### Feature Engineering (6 features per prospect-advisor pair)

| # | Feature | Method | Weight | Description |
|---|---------|--------|--------|-------------|
| 1 | `niche_alignment` | Sector similarity + adjacency graph | 0.28 | Cosine-like match; adjacent sectors get partial credit |
| 2 | `asset_fit` | Gaussian PDF N(μ, σ²) | 0.22 | Distance from advisor's ideal asset range |
| 3 | `life_event_signal` | Exponential decay (λ=90d) | 0.20 | Recent life events weighted by relevance |
| 4 | `geo_proximity` | State matching | 0.08 | Geographic distance (Haversine in production) |
| 5 | `conversion_likelihood` | Logistic regression proxy | 0.15 | No advisor + search intent + engagement |
| 6 | `wealth_coherence` | Anomaly detection | 0.07 | Flags inconsistent age-income-asset patterns |

### Scoring Function

```
raw = Σ(feature_i × weight_i) / Σ(weights)
F-Score = σ(8 × (raw - 0.45)) × 100

where σ(x) = 1 / (1 + e^(-x))
```

### Production Architecture (Theoretical)

| Stage | Method | Purpose |
|-------|--------|---------|
| Candidate Retrieval | ANN (FAISS/ScaNN) | Narrow 270M → 10K candidates |
| Ranking | XGBoost/LightGBM | Score with interaction features |
| Collaborative Filtering | ALS matrix factorization | "Advisors like you" signal |
| Calibration | Platt scaling | Map to true probabilities |
| Real-Time | Kafka + Flink streaming | Life event detection, score decay |

Full methodology is available at `GET /api/methodology`.

## API Reference

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/api/score?limit=N` | `AdvisorProfile` | Score & rank all prospects |
| `POST` | `/api/explain/{id}` | `AdvisorProfile` | Feature decomposition + NL explanation |
| `POST` | `/api/stats` | `AdvisorProfile` | Score distribution & aggregate stats |
| `GET` | `/api/prospects` | — | List prospects (filterable) |
| `GET` | `/api/methodology` | — | Full ML methodology as JSON |
| `GET` | `/api/health` | — | Health check |

**AdvisorProfile schema:**
```json
{
  "sectors": ["Tech", "Finance"],
  "event_categories": ["liquidity"],
  "states": ["CA", "NY"],
  "min_assets": 500000,
  "max_assets": 5000000
}
```

Interactive API docs: `http://localhost:8000/docs`

## Project Structure

```
fscore-engine/
├── README.md
├── render.yaml              # Render deploy config
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, routes, CORS
│   ├── models.py            # Pydantic schemas
│   ├── data.py              # Synthetic prospect generator
│   ├── scoring.py           # ML pipeline (features, scoring, explain)
│   └── requirements.txt
└── frontend/
    └── index.html           # React SPA (zero build step)
```

## Tech Stack

- **Backend:** Python 3.11+, FastAPI, Pydantic v2
- **Frontend:** React 18 (CDN, no build step), vanilla CSS
- **ML:** Custom feature engineering, weighted ensemble, sigmoid calibration
- **Data:** Seeded synthetic generation (reproducible via LCG PRNG)

## License

MIT
