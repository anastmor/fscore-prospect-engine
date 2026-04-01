"""
Pydantic data models for the F-Score Prospect Engine.
Defines schemas for prospects, advisor profiles, feature vectors, and API responses.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ─── Enums ───

class Sector(str, Enum):
    TECH = "Tech"
    MEDICAL = "Medical"
    LEGAL = "Legal"
    FINANCE = "Finance"
    REAL_ESTATE = "Real Estate"
    BUSINESS = "Business"
    CORPORATE = "Corporate"
    RETIRED = "Retired"
    CONSULTING = "Consulting"
    SPORTS = "Sports"
    ACADEMIC = "Academic"


class EventCategory(str, Enum):
    LIQUIDITY = "liquidity"
    INHERITANCE = "inheritance"
    RETIREMENT = "retirement"
    DIVORCE = "divorce"
    REAL_ESTATE = "real_estate"
    CAREER = "career"
    FAMILY = "family"


class ProspectStatus(str, Enum):
    HIGH_INTENT = "High Intent"
    WARM = "Warm"
    NURTURING = "Nurturing"
    LOW = "Low"


# ─── Core Models ───

class LifeEvent(BaseModel):
    event: str
    category: EventCategory
    weight: float = Field(ge=0, le=1)
    days_ago: int = Field(ge=0)


class Prospect(BaseModel):
    id: int
    name: str
    age: int
    city: str
    state: str
    occupation: str
    sector: Sector
    income: int                 # in thousands (e.g. 250 = $250K/yr)
    assets: int                 # in dollars
    events: list[LifeEvent]
    has_advisor: bool
    searching_online: bool
    engagement_score: int = Field(ge=0, le=100)


class AdvisorProfile(BaseModel):
    """Input from the financial advisor defining their ideal client."""
    sectors: list[Sector] = Field(default_factory=lambda: [Sector.TECH, Sector.FINANCE])
    event_categories: list[EventCategory] = Field(default_factory=lambda: [EventCategory.LIQUIDITY])
    states: list[str] = Field(default_factory=list, description="Target states (empty = nationwide)")
    min_assets: int = Field(default=500_000, description="Minimum investable assets ($)")
    max_assets: int = Field(default=5_000_000, description="Maximum investable assets ($)")


# ─── Feature & Scoring Models ───

class FeatureVector(BaseModel):
    """6-dimensional feature vector extracted per prospect-advisor pair."""
    niche_alignment: float = Field(ge=0, le=1, description="Sector similarity score")
    asset_fit: float = Field(ge=0, le=1, description="Gaussian distance from ideal range")
    life_event_signal: float = Field(ge=0, le=1, description="Recency-weighted event relevance")
    geo_proximity: float = Field(ge=0, le=1, description="Geographic match score")
    conversion_likelihood: float = Field(ge=0, le=1, description="Propensity-to-convert estimate")
    wealth_coherence: float = Field(ge=0, le=1, description="Age-income-asset consistency")


class ScoredProspect(BaseModel):
    """A prospect with computed F-Score and feature breakdown."""
    prospect: Prospect
    f_score: int = Field(ge=0, le=100)
    status: ProspectStatus
    features: FeatureVector
    rank: int


# ─── API Response Models ───

class ScoreResponse(BaseModel):
    """Response from the /score endpoint."""
    total_prospects: int
    high_intent_count: int
    warm_count: int
    results: list[ScoredProspect]
    model_weights: dict[str, float]


class StatsResponse(BaseModel):
    """Aggregate statistics for a scoring run."""
    total_prospects: int
    high_intent: int
    warm: int
    nurturing: int
    low: int
    avg_f_score: float
    median_f_score: float
    score_distribution: dict[str, int]  # buckets: "0-20", "20-40", etc.
    top_sectors: list[dict]
    top_life_events: list[dict]


class ExplainResponse(BaseModel):
    """Detailed explanation of a prospect's F-Score."""
    prospect: Prospect
    f_score: int
    features: FeatureVector
    feature_contributions: dict[str, float]  # weighted contribution of each feature
    explanation: list[str]  # human-readable explanations


class MethodologyResponse(BaseModel):
    """ML methodology documentation."""
    pipeline_stages: list[dict]
    feature_descriptions: list[dict]
    model_weights: dict[str, float]
    scoring_function: str
    production_architecture: dict
