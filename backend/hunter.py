"""
Hunter — Autonomous AI Growth Agent for financial advisors.

Powered by Claude. Transforms the F-Score prospect engine into a full growth OS:
strategy, campaigns, personalized outreach, and weekly intelligence briefings.

Endpoints:
    POST /api/hunter/strategy         — Personalized growth strategy
    POST /api/hunter/campaign         — 4-week campaign plan + sample content
    POST /api/hunter/outreach/{id}    — Personalized outreach (email, LinkedIn, voicemail)
    POST /api/hunter/analyze          — Weekly intelligence briefing
"""

import os
import json
import re
from dotenv import load_dotenv
import anthropic
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

load_dotenv()

try:
    from backend.models import AdvisorProfile
    from backend.scoring import score_all_prospects
except ImportError:
    from models import AdvisorProfile
    from scoring import score_all_prospects


# ─── Module State (initialized by main.py at startup) ───

PROSPECTS: list = []
PROSPECT_MAP: dict = {}


def init_prospects(prospects: list) -> None:
    global PROSPECTS, PROSPECT_MAP
    PROSPECTS = prospects
    PROSPECT_MAP = {p.id: p for p in prospects}


# ─── Router ───

router = APIRouter(prefix="/api/hunter", tags=["hunter"])

HUNTER_MODEL = "claude-sonnet-4-20250514"

CGO_SYSTEM_PROMPT = """You are Hunter, an AI Chief Growth Officer built specifically for independent financial advisors.

Your job is to help advisors grow their practice through data-driven prospect strategy, personalized outreach, and intelligent campaign design.

You have deep expertise in:
- Financial advisor marketing and business development
- Wealth management client acquisition (HNW, UHNW, mass affluent segments)
- LinkedIn thought leadership, cold email, referral networks, and event-based outreach
- Life event-driven prospecting (liquidity events, inheritances, retirement transitions, divorce)
- Compliance-aware communication for financial services

Rules:
- Be specific and actionable. Reference the advisor's actual niche and the data provided.
- Write in a professional but approachable tone — a trusted CMO, not a generic consultant.
- Always return valid JSON when asked. No markdown fences. No commentary outside the JSON.
- All outreach must be compliant: no promises of returns, no guarantees, no hard sells."""


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY is not configured on the server.",
        )
    return anthropic.Anthropic(api_key=api_key)


def _profile_summary(profile: AdvisorProfile) -> str:
    sectors = ", ".join(s.value for s in profile.sectors) if profile.sectors else "any sector"
    events = ", ".join(e.value for e in profile.event_categories) if profile.event_categories else "any life event"
    states = ", ".join(profile.states) if profile.states else "nationwide"
    return (
        f"Target sectors: {sectors}\n"
        f"Target life events: {events}\n"
        f"Geography: {states}\n"
        f"Asset range: ${profile.min_assets:,} – ${profile.max_assets:,}"
    )


def _parse_json(text: str, endpoint: str) -> dict:
    """Parse JSON from a Claude response, with fallback regex extraction."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    raise HTTPException(
        status_code=500,
        detail=f"Hunter ({endpoint}): failed to parse AI response as valid JSON.",
    )


# ─── Request / Response Models ───

class StrategyRequest(BaseModel):
    profile: AdvisorProfile
    advisor_description: str


class StrategyResponse(BaseModel):
    campaign_types: list[str]
    target_segments: list[str]
    content_themes: list[str]
    channel_mix: dict[str, str]
    full_strategy: str


class CampaignRequest(BaseModel):
    profile: AdvisorProfile
    campaign_type: str


class CampaignResponse(BaseModel):
    campaign_type: str
    four_week_plan: str
    sample_content: list[str]
    targeting_criteria: str
    kpi_targets: dict[str, str]


class OutreachResponse(BaseModel):
    prospect_name: str
    f_score: int
    email_variant: str
    linkedin_variant: str
    voicemail_script: str


class BriefingResponse(BaseModel):
    top_opportunities: str
    reasoning: str
    recommendations: str
    full_briefing: str


# ─── Endpoints ───

@router.post("/strategy", response_model=StrategyResponse)
def generate_strategy(req: StrategyRequest):
    """
    Generate a personalized growth strategy for the advisor.
    Returns campaign types, target segments, content themes, and channel mix.
    """
    client = _get_client()
    profile_text = _profile_summary(req.profile)

    message = client.messages.create(
        model=HUNTER_MODEL,
        max_tokens=2048,
        system=CGO_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                "Generate a personalized growth strategy for this financial advisor.\n\n"
                f"ADVISOR DESCRIPTION:\n{req.advisor_description}\n\n"
                f"IDEAL CLIENT PROFILE (from F-Score engine):\n{profile_text}\n\n"
                "Return a JSON object with these exact keys:\n"
                '- "campaign_types": array of 4-5 recommended campaign type strings\n'
                '- "target_segments": array of 3-4 specific prospect segment descriptions\n'
                '- "content_themes": array of 5-6 content theme strings\n'
                '- "channel_mix": object mapping channel names to brief strategy notes\n'
                '- "full_strategy": a 300-400 word narrative growth strategy\n\n'
                "Return only valid JSON. No markdown fences."
            ),
        }],
    )

    data = _parse_json(message.content[0].text, "strategy")
    return StrategyResponse(**data)


@router.post("/campaign", response_model=CampaignResponse)
def generate_campaign(req: CampaignRequest):
    """
    Generate a detailed 4-week campaign plan.
    Includes sample content in the advisor's voice, targeting criteria, and KPI targets.
    """
    client = _get_client()
    profile_text = _profile_summary(req.profile)

    message = client.messages.create(
        model=HUNTER_MODEL,
        max_tokens=2500,
        system=CGO_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                "Create a detailed 4-week campaign plan.\n\n"
                f"CAMPAIGN TYPE: {req.campaign_type}\n\n"
                f"IDEAL CLIENT PROFILE:\n{profile_text}\n\n"
                "Return a JSON object with these exact keys:\n"
                '- "campaign_type": the campaign type string (same as input)\n'
                '- "four_week_plan": a week-by-week plan (Week 1–4), as a single multi-paragraph string\n'
                '- "sample_content": array of exactly 3 sample content pieces in the advisor\'s voice\n'
                '- "targeting_criteria": a paragraph on who to target and how to find them\n'
                '- "kpi_targets": object with 4-5 KPI names as keys and benchmark values as string values\n\n'
                "Return only valid JSON. No markdown fences."
            ),
        }],
    )

    data = _parse_json(message.content[0].text, "campaign")
    return CampaignResponse(**data)


@router.post("/outreach/{prospect_id}", response_model=OutreachResponse)
def generate_outreach(prospect_id: int, profile: AdvisorProfile):
    """
    Generate personalized outreach for a specific prospect.
    Runs the F-Score pipeline first, then returns email, LinkedIn, and voicemail variants.
    """
    if prospect_id not in PROSPECT_MAP:
        raise HTTPException(status_code=404, detail=f"Prospect {prospect_id} not found")

    prospect = PROSPECT_MAP[prospect_id]
    client = _get_client()

    scored = score_all_prospects([prospect], profile)
    result = scored[0]

    events_text = (
        ", ".join(f"{e.event} ({e.days_ago}d ago)" for e in prospect.events)
        if prospect.events else "no recent life events"
    )
    profile_text = _profile_summary(profile)

    message = client.messages.create(
        model=HUNTER_MODEL,
        max_tokens=2000,
        system=CGO_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                "Generate 3 personalized outreach message variants for this prospect.\n\n"
                "PROSPECT:\n"
                f"- Name: {prospect.name}\n"
                f"- Age: {prospect.age}, Occupation: {prospect.occupation}\n"
                f"- Location: {prospect.city}, {prospect.state}\n"
                f"- Assets: ${prospect.assets:,}\n"
                f"- Life events: {events_text}\n"
                f"- Currently has advisor: {'Yes' if prospect.has_advisor else 'No'}\n"
                f"- F-Score: {result.f_score}/100 ({result.status.value})\n\n"
                f"ADVISOR PROFILE:\n{profile_text}\n\n"
                "Return a JSON object with these exact keys:\n"
                '- "email_variant": a personalized cold email with subject line + body (~150 words)\n'
                '- "linkedin_variant": a LinkedIn connection request message (~300 chars max)\n'
                '- "voicemail_script": a 30-second voicemail script\n\n'
                "Reference life events and occupation naturally. Be compliant — no promises of returns.\n"
                "Return only valid JSON. No markdown fences."
            ),
        }],
    )

    data = _parse_json(message.content[0].text, "outreach")
    return OutreachResponse(
        prospect_name=prospect.name,
        f_score=result.f_score,
        email_variant=data["email_variant"],
        linkedin_variant=data["linkedin_variant"],
        voicemail_script=data["voicemail_script"],
    )


@router.post("/analyze", response_model=BriefingResponse)
def generate_briefing(profile: AdvisorProfile):
    """
    Generate a weekly intelligence briefing.
    Scores all 200 prospects and synthesizes them into a CMO-style briefing.
    """
    client = _get_client()

    scored = score_all_prospects(PROSPECTS, profile)
    top_10 = scored[:10]

    top_prospects_text = "\n".join(
        f"{i+1}. {s.prospect.name} | {s.prospect.occupation} | "
        f"{s.prospect.city}, {s.prospect.state} | "
        f"Assets: ${s.prospect.assets:,} | F-Score: {s.f_score} | "
        f"Events: {', '.join(e.event for e in s.prospect.events) or 'none'}"
        for i, s in enumerate(top_10)
    )

    high_intent = sum(1 for s in scored if s.f_score >= 80)
    warm = sum(1 for s in scored if 60 <= s.f_score < 80)
    profile_text = _profile_summary(profile)

    message = client.messages.create(
        model=HUNTER_MODEL,
        max_tokens=2000,
        system=CGO_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                "Generate a weekly prospect intelligence briefing.\n\n"
                "SCORING SUMMARY:\n"
                f"- Total prospects scored: {len(scored)}\n"
                f"- High Intent (80+): {high_intent}\n"
                f"- Warm (60-79): {warm}\n\n"
                f"IDEAL CLIENT PROFILE:\n{profile_text}\n\n"
                f"TOP 10 PROSPECTS THIS WEEK:\n{top_prospects_text}\n\n"
                "Return a JSON object with these exact keys:\n"
                '- "top_opportunities": a paragraph on the 3-5 best prospects to contact this week and why\n'
                '- "reasoning": a paragraph on patterns in the data (life events, sectors, timing)\n'
                '- "recommendations": a numbered list of 4-5 specific actions this week (as a single string)\n'
                '- "full_briefing": a 400-500 word weekly briefing memo from a virtual CMO\n\n'
                "Return only valid JSON. No markdown fences."
            ),
        }],
    )

    data = _parse_json(message.content[0].text, "analyze")
    return BriefingResponse(**data)
