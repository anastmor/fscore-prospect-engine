"""
F-Score Prospect Engine — FastAPI Backend

REST API for scoring financial advisor prospects using an ML-inspired pipeline.

Endpoints:
    POST /api/score         — Score all prospects against an advisor profile
    GET  /api/prospects      — List all prospects
    GET  /api/prospects/{id} — Get single prospect
    POST /api/explain/{id}   — Feature decomposition for a specific prospect
    GET  /api/stats          — Aggregate scoring statistics
    GET  /api/methodology    — ML methodology documentation
    GET  /api/health         — Health check

Run:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from statistics import median

try:
    from backend.models import (
        AdvisorProfile, ScoreResponse, StatsResponse,
        ExplainResponse, MethodologyResponse, Prospect,
    )
    from backend.data import generate_prospects
    from backend.scoring import (
        score_all_prospects, explain_score, compute_f_score,
        extract_features, get_status, MODEL_WEIGHTS, METHODOLOGY,
    )
except ImportError:
    from models import (
        AdvisorProfile, ScoreResponse, StatsResponse,
        ExplainResponse, MethodologyResponse, Prospect,
    )
    from data import generate_prospects
    from scoring import (
        score_all_prospects, explain_score, compute_f_score,
        extract_features, get_status, MODEL_WEIGHTS, METHODOLOGY,
    )


# ─── App Setup ───

app = FastAPI(
    title="F-Score Prospect Engine",
    description=(
        "ML-powered prospect matching for financial advisor platforms. "
        "Scores individual prospects against an advisor's ideal client profile "
        "using a 6-feature weighted ensemble with sigmoid calibration."
    ),
    version="1.0.0",
    contact={
        "name": "Anastasis Moraitis",
        "url": "https://github.com/amoraitis",
        "email": "amoraitis3@gatech.edu",
    },
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Data Store ───
# In production: database connection, not in-memory generation
PROSPECTS = generate_prospects(n=200, seed=42)
PROSPECT_MAP = {p.id: p for p in PROSPECTS}


# ─── Routes ───

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "prospects_loaded": len(PROSPECTS),
        "model_version": "1.0.0",
    }


@app.get("/api/prospects", response_model=list[Prospect])
def list_prospects(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    sector: str | None = Query(default=None),
    state: str | None = Query(default=None),
    min_assets: int | None = Query(default=None),
):
    """
    List all prospects with optional filtering.
    No scoring applied — returns raw prospect data.
    """
    filtered = PROSPECTS

    if sector:
        filtered = [p for p in filtered if p.sector.value.lower() == sector.lower()]
    if state:
        filtered = [p for p in filtered if p.state == state.upper()]
    if min_assets is not None:
        filtered = [p for p in filtered if p.assets >= min_assets]

    return filtered[offset : offset + limit]


@app.get("/api/prospects/{prospect_id}", response_model=Prospect)
def get_prospect(prospect_id: int):
    """Get a single prospect by ID."""
    if prospect_id not in PROSPECT_MAP:
        raise HTTPException(status_code=404, detail=f"Prospect {prospect_id} not found")
    return PROSPECT_MAP[prospect_id]


@app.post("/api/score", response_model=ScoreResponse)
def score_prospects(
    profile: AdvisorProfile,
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    Score all prospects against the provided advisor profile.

    This is the main endpoint. It runs the full ML pipeline:
    1. Feature extraction for each prospect-advisor pair
    2. Weighted ensemble scoring
    3. Sigmoid calibration
    4. Ranking and status assignment

    Returns ranked results with feature decomposition.
    """
    scored = score_all_prospects(PROSPECTS, profile)

    high_intent = sum(1 for s in scored if s.f_score >= 80)
    warm = sum(1 for s in scored if 60 <= s.f_score < 80)

    return ScoreResponse(
        total_prospects=len(scored),
        high_intent_count=high_intent,
        warm_count=warm,
        results=scored[:limit],
        model_weights=MODEL_WEIGHTS,
    )


@app.post("/api/explain/{prospect_id}", response_model=ExplainResponse)
def explain_prospect_score(prospect_id: int, profile: AdvisorProfile):
    """
    Get detailed feature decomposition and natural language explanation
    for a specific prospect's F-Score.

    Useful for:
    - Understanding why a prospect ranked high/low
    - Building advisor trust in the scoring algorithm
    - Debugging feature contributions
    """
    if prospect_id not in PROSPECT_MAP:
        raise HTTPException(status_code=404, detail=f"Prospect {prospect_id} not found")

    return explain_score(PROSPECT_MAP[prospect_id], profile)


@app.post("/api/stats", response_model=StatsResponse)
def get_stats(profile: AdvisorProfile):
    """
    Aggregate statistics for a scoring run.
    Includes score distribution, top sectors, and top life events.
    """
    scored = score_all_prospects(PROSPECTS, profile)
    scores = [s.f_score for s in scored]

    # Score distribution buckets
    buckets = {"0-20": 0, "20-40": 0, "40-60": 0, "60-80": 0, "80-100": 0}
    for s in scores:
        if s < 20:
            buckets["0-20"] += 1
        elif s < 40:
            buckets["20-40"] += 1
        elif s < 60:
            buckets["40-60"] += 1
        elif s < 80:
            buckets["60-80"] += 1
        else:
            buckets["80-100"] += 1

    # Top sectors among high-scoring prospects
    high_scorers = [s for s in scored if s.f_score >= 60]
    sector_counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}

    for s in high_scorers:
        sec = s.prospect.sector.value
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        for ev in s.prospect.events:
            event_counts[ev.event] = event_counts.get(ev.event, 0) + 1

    top_sectors = sorted(
        [{"sector": k, "count": v} for k, v in sector_counts.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:5]

    top_events = sorted(
        [{"event": k, "count": v} for k, v in event_counts.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:5]

    return StatsResponse(
        total_prospects=len(scored),
        high_intent=sum(1 for s in scores if s >= 80),
        warm=sum(1 for s in scores if 60 <= s < 80),
        nurturing=sum(1 for s in scores if 40 <= s < 60),
        low=sum(1 for s in scores if s < 40),
        avg_f_score=round(sum(scores) / len(scores), 1),
        median_f_score=float(median(scores)),
        score_distribution=buckets,
        top_sectors=top_sectors,
        top_life_events=top_events,
    )


@app.get("/api/methodology", response_model=MethodologyResponse)
def get_methodology():
    """
    Return the full ML methodology documentation.

    Covers:
    - Pipeline stages
    - Feature engineering details
    - Model weights
    - Scoring function
    - Production architecture (theoretical)
    """
    return MethodologyResponse(**METHODOLOGY)


# ─── Startup ───

@app.on_event("startup")
def startup():
    print(f"F-Score Engine loaded with {len(PROSPECTS)} prospects")
    print(f"Model weights: {MODEL_WEIGHTS}")
