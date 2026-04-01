"""
Synthetic prospect data generator.

Generates realistic individual profiles modeled on U.S. demographics.
Uses seeded PRNG for reproducible results across runs.

In production, this module would be replaced by:
- SEC EDGAR / FINRA public filings
- Data aggregators (ZoomInfo, PitchBook, LinkedIn)
- Public records (real estate, court filings)
- News/event monitoring pipelines
"""

import math
try:
    from backend.models import Prospect, LifeEvent, EventCategory, Sector
except ImportError:
    from models import Prospect, LifeEvent, EventCategory, Sector


# ─── Reference Data ───

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Lisa", "Daniel", "Nancy",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
    "Kenneth", "Carol", "Kevin", "Amanda", "Brian", "Dorothy", "George", "Melissa",
    "Timothy", "Deborah", "Ronald", "Stephanie", "Edward", "Rebecca", "Jason", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
    "Nicholas", "Angela", "Eric", "Shirley", "Jonathan", "Anna", "Stephen", "Brenda",
    "Larry", "Pamela", "Justin", "Emma", "Scott", "Nicole", "Brandon", "Helen",
    "Benjamin", "Samantha", "Samuel", "Katherine", "Raymond", "Christine", "Gregory", "Debra",
    "Frank", "Rachel", "Alexander", "Carolyn", "Patrick", "Janet", "Jack", "Catherine",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
    "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
    "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey",
    "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson", "Watson",
    "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza", "Ruiz", "Hughes",
    "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers", "Long", "Ross",
]

CITIES = [
    {"city": "New York", "state": "NY"},
    {"city": "Los Angeles", "state": "CA"},
    {"city": "Chicago", "state": "IL"},
    {"city": "Houston", "state": "TX"},
    {"city": "Phoenix", "state": "AZ"},
    {"city": "San Francisco", "state": "CA"},
    {"city": "Seattle", "state": "WA"},
    {"city": "Austin", "state": "TX"},
    {"city": "Denver", "state": "CO"},
    {"city": "Boston", "state": "MA"},
    {"city": "Miami", "state": "FL"},
    {"city": "Atlanta", "state": "GA"},
    {"city": "Dallas", "state": "TX"},
    {"city": "Nashville", "state": "TN"},
    {"city": "Portland", "state": "OR"},
    {"city": "San Diego", "state": "CA"},
    {"city": "Scottsdale", "state": "AZ"},
    {"city": "Charlotte", "state": "NC"},
    {"city": "Raleigh", "state": "NC"},
    {"city": "Minneapolis", "state": "MN"},
    {"city": "Tampa", "state": "FL"},
    {"city": "Salt Lake City", "state": "UT"},
    {"city": "Greenwich", "state": "CT"},
    {"city": "Palo Alto", "state": "CA"},
    {"city": "Bethesda", "state": "MD"},
    {"city": "Naples", "state": "FL"},
    {"city": "Boca Raton", "state": "FL"},
    {"city": "Stamford", "state": "CT"},
    {"city": "Philadelphia", "state": "PA"},
    {"city": "San Jose", "state": "CA"},
]

OCCUPATIONS = [
    {"title": "Software Engineer", "sector": "Tech", "income_range": (120, 350)},
    {"title": "VP of Engineering", "sector": "Tech", "income_range": (250, 600)},
    {"title": "Product Manager", "sector": "Tech", "income_range": (150, 400)},
    {"title": "Startup Founder", "sector": "Tech", "income_range": (80, 2000)},
    {"title": "Physician", "sector": "Medical", "income_range": (250, 700)},
    {"title": "Surgeon", "sector": "Medical", "income_range": (400, 1200)},
    {"title": "Dentist", "sector": "Medical", "income_range": (180, 450)},
    {"title": "Attorney", "sector": "Legal", "income_range": (150, 800)},
    {"title": "Partner at Law Firm", "sector": "Legal", "income_range": (300, 2000)},
    {"title": "Investment Banker", "sector": "Finance", "income_range": (200, 1500)},
    {"title": "Hedge Fund Analyst", "sector": "Finance", "income_range": (180, 900)},
    {"title": "Real Estate Developer", "sector": "Real Estate", "income_range": (150, 3000)},
    {"title": "Commercial Broker", "sector": "Real Estate", "income_range": (100, 800)},
    {"title": "Small Business Owner", "sector": "Business", "income_range": (80, 500)},
    {"title": "Franchise Owner", "sector": "Business", "income_range": (100, 600)},
    {"title": "CEO", "sector": "Corporate", "income_range": (200, 5000)},
    {"title": "CFO", "sector": "Corporate", "income_range": (200, 2000)},
    {"title": "VP of Sales", "sector": "Corporate", "income_range": (180, 500)},
    {"title": "Retired Executive", "sector": "Retired", "income_range": (0, 200)},
    {"title": "Retired Physician", "sector": "Retired", "income_range": (0, 150)},
    {"title": "Professor", "sector": "Academic", "income_range": (90, 250)},
    {"title": "Professional Athlete", "sector": "Sports", "income_range": (200, 10000)},
    {"title": "Consultant", "sector": "Consulting", "income_range": (120, 600)},
    {"title": "Management Consultant", "sector": "Consulting", "income_range": (150, 500)},
]

LIFE_EVENTS = [
    {"event": "Sold company", "category": "liquidity", "weight": 0.95},
    {"event": "IPO vesting", "category": "liquidity", "weight": 0.92},
    {"event": "RSU cliff vesting", "category": "liquidity", "weight": 0.78},
    {"event": "Stock options exercise", "category": "liquidity", "weight": 0.80},
    {"event": "Inheritance received", "category": "inheritance", "weight": 0.88},
    {"event": "Trust distribution", "category": "inheritance", "weight": 0.75},
    {"event": "Approaching retirement", "category": "retirement", "weight": 0.85},
    {"event": "Early retirement", "category": "retirement", "weight": 0.90},
    {"event": "Pension rollover", "category": "retirement", "weight": 0.72},
    {"event": "Divorce settlement", "category": "divorce", "weight": 0.82},
    {"event": "Home sale (>$1M)", "category": "real_estate", "weight": 0.68},
    {"event": "Job change to C-suite", "category": "career", "weight": 0.70},
    {"event": "New baby", "category": "family", "weight": 0.45},
    {"event": "Child entering college", "category": "family", "weight": 0.55},
    {"event": "Business acquisition", "category": "liquidity", "weight": 0.88},
    {"event": "Relocation", "category": "career", "weight": 0.40},
    {"event": "Widow/widower", "category": "inheritance", "weight": 0.78},
    {"event": "401k rollover", "category": "retirement", "weight": 0.60},
]


# ─── Seeded PRNG ───

class SeededRandom:
    """
    Linear congruential generator for reproducible random sequences.
    Same algorithm used in the frontend for parity.
    """

    def __init__(self, seed: int = 42):
        self.state = seed

    def next(self) -> float:
        self.state = (self.state * 16807) % 2147483647
        return self.state / 2147483647

    def randint(self, low: int, high: int) -> int:
        return low + int(self.next() * (high - low))

    def choice(self, lst: list):
        return lst[int(self.next() * len(lst))]


# ─── Generator ───

def generate_prospects(n: int = 200, seed: int = 42) -> list[Prospect]:
    """
    Generate n synthetic prospect profiles.

    Correlations modeled:
    - Assets scale with age and income (wealth accumulation curve)
    - Life event count increases with age (more life transitions)
    - Occupation determines income range (sector-specific distributions)
    - Geographic distribution weighted toward financial hubs
    """
    rng = SeededRandom(seed)
    prospects = []

    for i in range(n):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        loc = rng.choice(CITIES)
        occ = rng.choice(OCCUPATIONS)

        age = rng.randint(25, 70)
        income_lo, income_hi = occ["income_range"]
        income = rng.randint(income_lo, income_hi)

        # Assets correlated with age and income
        asset_multiplier = (age / 40) * (1 + rng.next() * 2)
        assets = round(income * asset_multiplier * 1000)

        # Life events: older people have more
        max_events = int(rng.next() * 3 * (1.3 if age > 45 else 0.8))
        events = []
        for _ in range(max_events):
            ev_data = rng.choice(LIFE_EVENTS)
            if not any(e.event == ev_data["event"] for e in events):
                days_ago = rng.randint(0, 180)
                events.append(LifeEvent(
                    event=ev_data["event"],
                    category=EventCategory(ev_data["category"]),
                    weight=ev_data["weight"],
                    days_ago=days_ago,
                ))

        # Behavioral signals
        has_advisor = rng.next() < 0.35
        searching = rng.next() < 0.45
        engagement = round(rng.next() * 100)

        prospects.append(Prospect(
            id=i,
            name=f"{first} {last}",
            age=age,
            city=loc["city"],
            state=loc["state"],
            occupation=occ["title"],
            sector=Sector(occ["sector"]),
            income=income,
            assets=assets,
            events=events,
            has_advisor=has_advisor,
            searching_online=searching,
            engagement_score=engagement,
        ))

    return prospects
