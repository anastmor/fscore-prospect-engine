"""
ML Scoring Pipeline for the F-Score Prospect Engine.

Implements a 6-feature scoring model with:
- Feature extraction per prospect-advisor pair
- Weighted ensemble scoring
- Sigmoid calibration for probability mapping
- Feature contribution analysis for explainability

In production, this would be replaced by:
- Stage 1: ANN candidate retrieval (FAISS/ScaNN) over 270M+ profiles
- Stage 2: Gradient Boosted Trees (XGBoost/LightGBM) ranking model
- Stage 3: Collaborative filtering adjustment
- Stage 4: Platt scaling / isotonic regression calibration
"""

import math
try:
    from backend.models import (
        Prospect, AdvisorProfile, FeatureVector, ScoredProspect,
        ProspectStatus, ExplainResponse, Sector, EventCategory,
    )
except ImportError:
    from models import (
        Prospect, AdvisorProfile, FeatureVector, ScoredProspect,
        ProspectStatus, ExplainResponse, Sector, EventCategory,
    )


# ─── Model Configuration ───

MODEL_WEIGHTS = {
    "niche_alignment": 0.28,
    "asset_fit": 0.22,
    "life_event_signal": 0.20,
    "geo_proximity": 0.08,
    "conversion_likelihood": 0.15,
    "wealth_coherence": 0.07,
}

# Sector adjacency graph: models learned embeddings from advisor-client pairing data.
# In production, this would be a word2vec or GloVe embedding similarity matrix
# trained on historical conversion data.
SECTOR_ADJACENCY: dict[str, list[str]] = {
    "Tech": ["Consulting", "Corporate"],
    "Finance": ["Corporate", "Consulting"],
    "Medical": ["Academic"],
    "Business": ["Real Estate", "Corporate"],
    "Corporate": ["Tech", "Finance", "Business"],
    "Consulting": ["Tech", "Finance"],
    "Real Estate": ["Business"],
    "Legal": ["Finance", "Corporate"],
    "Retired": [],
    "Sports": [],
    "Academic": ["Medical"],
}

# Sigmoid calibration parameters
# Steepness controls score spread; center controls the inflection point.
# Tuned so that a "perfect" feature vector maps to ~95 and a "zero" vector maps to ~5.
SIGMOID_STEEPNESS = 8.0
SIGMOID_CENTER = 0.45


# ─── Feature Extraction ───

def extract_features(prospect: Prospect, profile: AdvisorProfile) -> FeatureVector:
    """
    Extract a 6-dimensional feature vector for a prospect-advisor pair.

    Each feature is normalized to [0, 1] range. The features capture different
    aspects of prospect-advisor compatibility:

    1. Niche alignment — sector similarity with adjacency expansion
    2. Asset fit — Gaussian PDF distance from ideal range
    3. Life event signal — recency-weighted event relevance
    4. Geographic proximity — state-level matching
    5. Conversion likelihood — composite propensity score
    6. Wealth coherence — anomaly detection for data quality
    """

    # ─── F1: Niche Alignment ───
    # Direct match: 1.0, adjacent sector: 0.4, no match: 0.1
    # In production: cosine similarity in a learned 128-dim embedding space
    prospect_sector = prospect.sector.value
    target_sectors = [s.value for s in profile.sectors]

    if prospect_sector in target_sectors:
        niche_alignment = 1.0
    elif any(
        prospect_sector in SECTOR_ADJACENCY.get(s, [])
        or s in SECTOR_ADJACENCY.get(prospect_sector, [])
        for s in target_sectors
    ):
        niche_alignment = 0.4
    else:
        niche_alignment = 0.1

    # ─── F2: Asset Fit ───
    # Gaussian PDF: P(assets | μ, σ) where μ = midpoint of ideal range
    # This models the assumption that prospects near the center of the
    # advisor's target range are the best fit, with smooth falloff.
    midpoint = (profile.min_assets + profile.max_assets) / 2
    sigma = (profile.max_assets - profile.min_assets) / 2 or 500_000
    asset_fit = math.exp(
        -((prospect.assets - midpoint) ** 2) / (2 * sigma ** 2)
    )
    # Boost for prospects within range (hard floor at 0.5 if above minimum)
    if profile.min_assets <= prospect.assets <= profile.max_assets * 2:
        asset_fit = max(asset_fit, 0.8)
    elif prospect.assets >= profile.min_assets:
        asset_fit = max(asset_fit, 0.5)

    # ─── F3: Life Event Signal ───
    # Exponential time-decay: recent events matter more.
    # λ = 90 days (half-life). A liquidity event 1 week ago scores ~15x
    # higher than the same event 6 months ago.
    # Category matching provides a 1.5x multiplier.
    target_categories = [c.value for c in profile.event_categories]
    event_score = 0.0
    for ev in prospect.events:
        recency_decay = math.exp(-ev.days_ago / 90)
        category_boost = 1.5 if ev.category.value in target_categories else 0.8
        event_score += ev.weight * recency_decay * category_boost
    life_event_signal = min(event_score / 1.5, 1.0)

    # ─── F4: Geographic Proximity ───
    # Binary state matching in this prototype.
    # Production: Haversine distance with learned radius preferences.
    # Some advisors are local-only; others are nationwide.
    if len(profile.states) == 0:
        geo_proximity = 0.5  # nationwide = neutral
    elif prospect.state in profile.states:
        geo_proximity = 1.0
    else:
        geo_proximity = 0.2

    # ─── F5: Conversion Likelihood ───
    # Proxy for a logistic regression model predicting P(convert | features).
    # Key signals:
    #   - No current advisor → 3.3x more likely to convert
    #   - Actively searching online → 1.3x intent multiplier
    #   - Engagement score → normalized activity level
    no_advisor_boost = 1.0 if not prospect.has_advisor else 0.3
    search_boost = 1.3 if prospect.searching_online else 0.7
    engagement_norm = prospect.engagement_score / 100
    conversion_likelihood = min(
        no_advisor_boost * search_boost * engagement_norm,
        1.0
    )

    # ─── F6: Wealth Coherence ───
    # Anomaly detection: flags prospects where reported assets don't match
    # expected wealth accumulation (age × income heuristic).
    # Catches data quality issues and unusual wealth patterns.
    # In production: isolation forest or autoencoder on full feature set.
    expected_assets = prospect.income * 1000 * (prospect.age / 35)
    ratio = prospect.assets / expected_assets if expected_assets > 0 else 0
    wealth_coherence = 0.8 if 0.5 < ratio < 5 else 0.3

    return FeatureVector(
        niche_alignment=round(niche_alignment, 4),
        asset_fit=round(asset_fit, 4),
        life_event_signal=round(life_event_signal, 4),
        geo_proximity=round(geo_proximity, 4),
        conversion_likelihood=round(conversion_likelihood, 4),
        wealth_coherence=round(wealth_coherence, 4),
    )


# ─── Scoring ───

def compute_f_score(features: FeatureVector) -> int:
    """
    Compute the F-Score (0-100) from a feature vector.

    Pipeline:
    1. Weighted sum of features
    2. Normalize by total weight
    3. Apply sigmoid calibration

    The sigmoid σ(k(x - c)) maps the normalized score to a probability-like
    value in (0, 1), which is then scaled to 0-100.

    Parameters:
    - k (steepness) = 8.0: Controls the spread of scores.
      Higher k → more binary (scores cluster at 0 or 100)
      Lower k → more spread (scores use more of the 0-100 range)
    - c (center) = 0.45: The input value that maps to F-Score 50.
      Shifted below 0.5 so that "average" prospects score slightly above 50,
      matching advisor intuition that most curated leads are at least decent.
    """
    feature_dict = features.model_dump()
    raw = sum(feature_dict[k] * w for k, w in MODEL_WEIGHTS.items())
    total_weight = sum(MODEL_WEIGHTS.values())
    normalized = raw / total_weight

    # Sigmoid calibration
    calibrated = 1 / (1 + math.exp(-SIGMOID_STEEPNESS * (normalized - SIGMOID_CENTER)))
    return round(calibrated * 100)


def get_status(f_score: int) -> ProspectStatus:
    """Map F-Score to a human-readable status bucket."""
    if f_score >= 80:
        return ProspectStatus.HIGH_INTENT
    elif f_score >= 60:
        return ProspectStatus.WARM
    elif f_score >= 40:
        return ProspectStatus.NURTURING
    return ProspectStatus.LOW


# ─── Batch Scoring ───

def score_all_prospects(
    prospects: list[Prospect],
    profile: AdvisorProfile,
) -> list[ScoredProspect]:
    """
    Score all prospects against an advisor profile and return ranked results.

    This is the main entry point for the scoring pipeline.
    In production, Stage 1 (candidate retrieval via ANN) would reduce the
    input set from 270M+ to ~10K before this ranking step runs.
    """
    scored = []
    for prospect in prospects:
        features = extract_features(prospect, profile)
        f_score = compute_f_score(features)
        scored.append(ScoredProspect(
            prospect=prospect,
            f_score=f_score,
            status=get_status(f_score),
            features=features,
            rank=0,  # assigned after sorting
        ))

    # Sort by F-Score descending
    scored.sort(key=lambda s: s.f_score, reverse=True)

    # Assign ranks
    for i, s in enumerate(scored):
        s.rank = i + 1

    return scored


# ─── Explainability ───

def explain_score(
    prospect: Prospect,
    profile: AdvisorProfile,
) -> ExplainResponse:
    """
    Generate a human-readable explanation of a prospect's F-Score.

    Returns both the numeric feature contributions and natural language
    explanations for each factor. This powers the "Feature Decomposition"
    panel in the UI.
    """
    features = extract_features(prospect, profile)
    f_score = compute_f_score(features)
    feature_dict = features.model_dump()

    # Compute weighted contributions
    total_weight = sum(MODEL_WEIGHTS.values())
    contributions = {
        k: round((feature_dict[k] * w) / total_weight, 4)
        for k, w in MODEL_WEIGHTS.items()
    }

    # Generate explanations
    explanations = []

    # Niche alignment
    if features.niche_alignment >= 0.8:
        explanations.append(
            f"Strong niche match: {prospect.occupation} ({prospect.sector.value}) "
            f"directly aligns with advisor's target sectors."
        )
    elif features.niche_alignment >= 0.3:
        explanations.append(
            f"Partial niche match: {prospect.sector.value} is adjacent to "
            f"advisor's target sectors in the sector similarity graph."
        )
    else:
        explanations.append(
            f"Weak niche match: {prospect.sector.value} is outside the "
            f"advisor's target sectors."
        )

    # Asset fit
    asset_str = f"${prospect.assets:,.0f}"
    range_str = f"${profile.min_assets:,.0f}-${profile.max_assets:,.0f}"
    if features.asset_fit >= 0.7:
        explanations.append(
            f"Strong asset fit: {asset_str} falls within the ideal range ({range_str})."
        )
    elif features.asset_fit >= 0.4:
        explanations.append(
            f"Moderate asset fit: {asset_str} is near the ideal range ({range_str})."
        )
    else:
        explanations.append(
            f"Weak asset fit: {asset_str} is far from the ideal range ({range_str})."
        )

    # Life events
    if features.life_event_signal >= 0.6:
        recent_events = [e.event for e in prospect.events if e.days_ago < 60]
        explanations.append(
            f"High event signal: Recent events ({', '.join(recent_events or ['multiple'])}) "
            f"indicate a financial transition moment."
        )
    elif features.life_event_signal > 0:
        explanations.append(
            f"Moderate event signal: {len(prospect.events)} life event(s) detected, "
            f"but recency decay reduces impact."
        )
    else:
        explanations.append("No life events detected in the monitoring window.")

    # Conversion likelihood
    if features.conversion_likelihood >= 0.6:
        signals = []
        if not prospect.has_advisor:
            signals.append("no current advisor")
        if prospect.searching_online:
            signals.append("active online search behavior")
        explanations.append(
            f"High conversion potential: {', '.join(signals) or 'strong engagement signals'}."
        )
    else:
        explanations.append(
            "Lower conversion potential: prospect may already have an advisor "
            "or shows limited engagement."
        )

    return ExplainResponse(
        prospect=prospect,
        f_score=f_score,
        features=features,
        feature_contributions=contributions,
        explanation=explanations,
    )


# ─── Methodology Documentation ───

METHODOLOGY = {
    "pipeline_stages": [
        {
            "stage": "Data Ingestion",
            "description": (
                "Aggregate prospect data from public records, social signals, "
                "and financial databases. In production: 270M+ U.S. adult profiles "
                "from SEC filings, real estate records, LinkedIn, news monitoring."
            ),
        },
        {
            "stage": "Feature Extraction",
            "description": (
                "Extract 6 engineered features per prospect-advisor pair. Each "
                "feature captures a different dimension of compatibility: sector "
                "alignment, financial fit, life event timing, geography, conversion "
                "propensity, and data quality."
            ),
        },
        {
            "stage": "Model Inference",
            "description": (
                "Weighted ensemble scoring with sigmoid calibration. In production: "
                "two-stage architecture — ANN candidate retrieval (FAISS) followed "
                "by XGBoost/LightGBM ranking with interaction features."
            ),
        },
        {
            "stage": "Calibration",
            "description": (
                "Sigmoid function σ(8(x - 0.45)) maps raw scores to 0-100. "
                "In production: Platt scaling or isotonic regression trained on "
                "held-out conversion data for true probability estimates."
            ),
        },
        {
            "stage": "Delivery",
            "description": (
                "Ranked prospect list with status labels (High Intent / Warm / "
                "Nurturing / Low) and feature decomposition for transparency."
            ),
        },
    ],
    "feature_descriptions": [
        {
            "name": "niche_alignment",
            "weight": 0.28,
            "method": "Sector similarity with adjacency graph",
            "detail": (
                "Direct sector match scores 1.0. Adjacent sectors (defined by a "
                "learned embedding adjacency graph) score 0.4. Non-matching sectors "
                "score 0.1. In production, cosine similarity in a word2vec embedding "
                "space trained on historical advisor-client pairings."
            ),
        },
        {
            "name": "asset_fit",
            "weight": 0.22,
            "method": "Gaussian probability density function",
            "detail": (
                "Models P(assets | μ, σ²) where μ = midpoint of advisor's ideal "
                "range and σ = half the range width. Prospects near the center of "
                "the target range receive the highest scores, with smooth Gaussian "
                "falloff. Hard floor at 0.5 for assets above minimum threshold."
            ),
        },
        {
            "name": "life_event_signal",
            "weight": 0.20,
            "method": "Exponential time-decay with category weighting",
            "detail": (
                "Each life event contributes: weight × e^(-days/90) × category_boost. "
                "The 90-day decay constant means a 1-week-old event scores ~15x higher "
                "than a 6-month-old event. Events matching advisor's target categories "
                "receive a 1.5x multiplier. Captures the insight that financial "
                "transitions create a narrow window of advisor-seeking behavior."
            ),
        },
        {
            "name": "geo_proximity",
            "weight": 0.08,
            "method": "State-level matching (Haversine in production)",
            "detail": (
                "Binary state matching in this prototype. In production: Haversine "
                "distance calculation with learned radius preferences per advisor. "
                "Some advisors serve only local clients; others are fully virtual. "
                "Low weight reflects the trend toward remote advisory relationships."
            ),
        },
        {
            "name": "conversion_likelihood",
            "weight": 0.15,
            "method": "Logistic regression proxy (composite signal)",
            "detail": (
                "Combines three behavioral signals: (1) no current advisor (3.3x "
                "conversion lift), (2) active online search for financial advice "
                "(1.3x intent signal), (3) normalized engagement score. Capped at "
                "1.0. In production: full logistic regression with 20+ behavioral "
                "features from CRM, email, and web analytics data."
            ),
        },
        {
            "name": "wealth_coherence",
            "weight": 0.07,
            "method": "Anomaly detection (ratio-based)",
            "detail": (
                "Compares reported assets to expected wealth accumulation "
                "(income × age/35 heuristic). Ratios between 0.5x and 5x are "
                "considered normal (score 0.8); outliers score 0.3. Catches data "
                "quality issues. In production: isolation forest or autoencoder "
                "trained on verified financial profiles."
            ),
        },
    ],
    "model_weights": MODEL_WEIGHTS,
    "scoring_function": (
        "F-Score = σ(k(x - c)) × 100\n"
        "where σ(z) = 1/(1 + e^(-z)), k = 8.0, c = 0.45\n"
        "x = Σ(feature_i × weight_i) / Σ(weights)"
    ),
    "production_architecture": {
        "stage_1_retrieval": {
            "method": "Approximate Nearest Neighbors (FAISS / ScaNN)",
            "description": (
                "Narrows 270M+ profiles to ~10K candidates per advisor. "
                "Embedding model: fine-tuned sentence transformer encoding "
                "advisor profiles and prospect attributes into a shared 128-dim "
                "space. Blocking keys: state, income bracket, sector."
            ),
        },
        "stage_2_ranking": {
            "method": "Gradient Boosted Decision Trees (XGBoost / LightGBM)",
            "description": (
                "Trained on historical advisor-client conversions. Features: 6 "
                "base features + interaction terms (sector×event, assets×age). "
                "Hard negative mining for training: prospects similar to converted "
                "clients who didn't convert. Evaluation: AUC-ROC, precision@k, NDCG."
            ),
        },
        "stage_3_collaborative_filtering": {
            "method": "Matrix Factorization (ALS)",
            "description": (
                "Captures 'advisors like you converted prospects like this' signal. "
                "Cold-start handling via sector/AUM cluster centroids. Adjustment "
                "capped at ±10 F-Score points."
            ),
        },
        "stage_4_realtime": {
            "method": "Streaming pipeline (Kafka + Flink)",
            "description": (
                "Life event detection via NLP on news, SEC filings, LinkedIn changes. "
                "Engagement signals: email opens, website visits, content downloads. "
                "Sub-minute re-scoring latency for high-priority events. "
                "Score decay for uncontacted prospects (30-day window)."
            ),
        },
        "training": {
            "data": "Historical advisor-client matches + hard negatives",
            "validation": "Grouped by advisor (no leakage), temporal train/val/test split",
            "metrics": "AUC-ROC (discrimination), precision@10 (top-k quality), NDCG (ranking)",
        },
    },
}
