"""
scripts/synthetic_data.py — Insurance synthetic database generator (ins_prod_v3 mirror)
Data Analyst Copilot · Phase 3 · Python 3.11+

Generates a 5-table SQLite insurance database that is a structural mirror of the
ins_prod_v3 PostgreSQL schema (generate_insurance_data.py).  Column names, data
types, FK relationships, null rates, claim_status values, and payment-generation
logic are intentionally aligned so that eval pairs authored against ins_prod_v3
execute correctly against this SQLite database.

Schema (mirrors ins_prod_v3 exactly):
  customers  (n_customers rows)  ← base table
  agents     (n_agents rows)     ← reference table (default: 100)
  policies   (1.5× rows)        → FK: customer_id → customers
                                    FK: agent_id   → agents
  claims     (2× rows)          → FK: policy_id   → policies
  payments   (~46% of claims)   → FK: claim_id    → claims
                                   (approved + paid claims with non-null amounts only)

Null rates (matched exactly to ins_prod_v3.meta):
  policies.end_date   : ~85.3%   active policies have no end_date
  claims.claim_amount :  ~7.9%   open/pending claims
  claims.adjuster_id  : ~12.3%
  claims.resolved_at  : ~24.9%   open claims are unresolved
  claims.notes        : ~69.7%

Column name alignment (previous SQLite → ins_prod_v3):
  customers.birth_date        → date_of_birth
  customers.first_name/last_name → full_name
  policies.premium            → premium_amt
  policies.status             → is_active (BOOLEAN)
  claims.settled_date         → resolved_at
  claims.description          → notes
  payments.amount             → paid_amount
  payments.method             → payment_method
  payments.payment_date       → paid_at

claim_status values (previous SQLite → ins_prod_v3):
  open/closed/pending/denied/appealed → open/approved/rejected/paid

Usage:
    python scripts/synthetic_data.py --output data/insurance.db --rows 10000
    python scripts/synthetic_data.py --output data/insurance.db --rows 100000 --seed 42
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import string
import time
import uuid
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — aligned to ins_prod_v3 / generate_insurance_data.py
# ---------------------------------------------------------------------------

# M9 FIX: "commercial" removed — it is absent from the PostgreSQL schema
# (generate_insurance_data.py) and explicitly documented as a hallucinated
# value in reingest_with_overrides.py.  Including it in the SQLite eval DB
# caused the LLM's use of policy_type='commercial' to be scored as correct
# on eval pairs, inflating correctness_rate against the production schema.
POLICY_TYPES = ["auto", "home", "life", "health"]

# Aligned to generate_insurance_data.py CLAIM_STATUSES.
# Previous set (open/closed/pending/denied/appealed) shared only "open"
# with production, causing all status-filter eval pairs to return zero rows
# against this SQLite database.
CLAIM_STATUSES = ["open", "approved", "rejected", "paid"]

# Aligned to generate_insurance_data.py REGIONS (title-case, not lowercase)
REGIONS = ["North", "South", "East", "West", "Central"]

# Aligned to generate_insurance_data.py PAYMENT_METHODS
PAYMENT_METHODS = ["bank_transfer", "cheque", "direct_debit", "card"]

# ---------------------------------------------------------------------------
# Actuarial claim amount distributions — Issue #3 fix
# ---------------------------------------------------------------------------
# Previously all policy types used random.expovariate(1/8_000) → mean ~$8K
# for every type, producing ~$379 spread across GROUP BY policy_type queries.
#
# Parameters: lognormal(mu, sigma) so E[amount] = exp(mu + σ²/2):
#   auto   → mean ≈  $8 K  (collision/theft; long tail for total-loss)
#   home   → mean ≈ $18 K  (structural repair; tail for fire/flood)
#   health → mean ≈ $30 K  (outpatient-heavy; tail for surgical/ICU)
#   life   → mean ≈ $120 K (face-value payouts; tight σ = standardised)
_CLAIM_LOGNORM: dict[str, tuple[float, float]] = {
    "auto": (8.667, 0.80),
    "home": (9.393, 0.90),
    "health": (9.809, 1.00),
    "life": (11.450, 0.70),
}
_CLAIM_FLOOR: float = 100.0
_CLAIM_CAP: dict[str, float] = {
    "auto": 200_000.0,
    "home": 750_000.0,
    "health": 1_000_000.0,
    "life": 3_000_000.0,
}

# Actuarial premium distributions — Issue #4 fix (mirrors generate_insurance_data.py)
# uniform(lo, hi) per policy_type; ranges calibrated for loss_ratio ≈ 0.85
# given synthetic claim frequency of ~3.68 non-null claims per policy.
_PREMIUM_RANGE: dict[str, tuple[float, float]] = {
    "auto": (17_000.0, 52_000.0),
    "home": (39_000.0, 117_000.0),
    "health": (65_000.0, 195_000.0),
    "life": (260_000.0, 779_000.0),
}


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate(
    output_path: str = "data/insurance.db",
    n_customers: int = 10_000,
    n_agents: int = 100,
    seed: int = 42,
    echo: bool = True,
) -> None:
    random.seed(seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if echo:
        print(f"Generating insurance DB → {output_path}")

    conn = sqlite3.connect(output_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    _create_tables(conn)

    t0 = time.monotonic()

    agents = _gen_agents(n_agents)
    _insert(conn, "agents", agents)
    if echo:
        print(f"  agents:    {len(agents):>8,}")

    customers = _gen_customers(n_customers)
    _insert(conn, "customers", customers)
    if echo:
        print(f"  customers: {len(customers):>8,}")

    customer_ids = [c["customer_id"] for c in customers]
    agent_ids = [a["agent_id"] for a in agents]

    n_policies = int(n_customers * 1.5)
    policies = _gen_policies(n_policies, customer_ids, agent_ids)
    _insert(conn, "policies", policies)
    if echo:
        print(f"  policies:  {len(policies):>8,}")

    n_claims = int(n_customers * 2)
    claims = _gen_claims(n_claims, policies)
    _insert(conn, "claims", claims)
    if echo:
        print(f"  claims:    {len(claims):>8,}")

    payments = _gen_payments(claims)
    _insert(conn, "payments", payments)
    if echo:
        print(f"  payments:  {len(payments):>8,}")

    conn.commit()

    # Null-rate validation against ins_prod_v3.meta targets
    if echo:
        rows = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        null_amt = conn.execute(
            "SELECT COUNT(*) FROM claims WHERE claim_amount IS NULL"
        ).fetchone()[0]
        null_adj = conn.execute("SELECT COUNT(*) FROM claims WHERE adjuster_id IS NULL").fetchone()[
            0
        ]
        null_res = conn.execute("SELECT COUNT(*) FROM claims WHERE resolved_at IS NULL").fetchone()[
            0
        ]
        null_notes = conn.execute("SELECT COUNT(*) FROM claims WHERE notes IS NULL").fetchone()[0]
        null_end = conn.execute("SELECT COUNT(*) FROM policies WHERE end_date IS NULL").fetchone()[
            0
        ]
        n_pol = conn.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
        print("\n[validate] null rates vs ins_prod_v3.meta targets:")
        print(f"  policies.end_date   : {null_end/n_pol:.1%}  (target ~85.3%)")
        print(f"  claims.claim_amount : {null_amt/rows:.1%}  (target  ~7.9%)")
        print(f"  claims.adjuster_id  : {null_adj/rows:.1%}  (target ~12.3%)")
        print(f"  claims.resolved_at  : {null_res/rows:.1%}  (target ~24.9%)")
        print(f"  claims.notes        : {null_notes/rows:.1%}  (target ~69.7%)")

    conn.close()
    elapsed = time.monotonic() - t0
    if echo:
        print(f"\nDone in {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Schema — mirrors ins_prod_v3 exactly
# ---------------------------------------------------------------------------


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    DROP TABLE IF EXISTS payments;
    DROP TABLE IF EXISTS claims;
    DROP TABLE IF EXISTS policies;
    DROP TABLE IF EXISTS customers;
    DROP TABLE IF EXISTS agents;

    CREATE TABLE agents (
        agent_id    TEXT PRIMARY KEY,
        agent_code  TEXT NOT NULL UNIQUE,
        region      TEXT,
        hired_at    DATE
    );

    CREATE TABLE customers (
        customer_id   TEXT PRIMARY KEY,
        full_name     TEXT NOT NULL,
        date_of_birth DATE,
        postcode      TEXT,
        email         TEXT,
        created_at    DATETIME NOT NULL
    );

    CREATE TABLE policies (
        policy_id   TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL REFERENCES customers(customer_id),
        agent_id    TEXT REFERENCES agents(agent_id),
        policy_type TEXT NOT NULL,
        premium_amt REAL,
        start_date  DATE NOT NULL,
        end_date    DATE,
        is_active   INTEGER NOT NULL,
        created_at  DATETIME NOT NULL
    );

    CREATE TABLE claims (
        claim_id     TEXT PRIMARY KEY,
        policy_id    TEXT NOT NULL REFERENCES policies(policy_id),
        claim_date   DATE NOT NULL,
        claim_amount REAL,
        claim_status TEXT NOT NULL,
        adjuster_id  TEXT,
        resolved_at  DATETIME,
        notes        TEXT
    );

    CREATE TABLE payments (
        payment_id     TEXT PRIMARY KEY,
        claim_id       TEXT NOT NULL REFERENCES claims(claim_id),
        paid_amount    REAL NOT NULL,
        paid_at        DATETIME NOT NULL,
        payment_method TEXT
    );

    CREATE INDEX idx_policies_customer ON policies(customer_id);
    CREATE INDEX idx_policies_type     ON policies(policy_type);
    CREATE INDEX idx_policies_active   ON policies(is_active);
    CREATE INDEX idx_claims_policy     ON claims(policy_id);
    CREATE INDEX idx_claims_date       ON claims(claim_date);
    CREATE INDEX idx_claims_status     ON claims(claim_status);
    CREATE INDEX idx_payments_claim    ON payments(claim_id);
    CREATE INDEX idx_customers_dob     ON customers(date_of_birth);
    """)


# ---------------------------------------------------------------------------
# Row generators
# ---------------------------------------------------------------------------


def _gen_agents(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "agent_id": _uid(),
                "agent_code": f"AGT{i:05d}",
                "region": random.choice(REGIONS),
                "hired_at": str(_rand_date(date(2010, 1, 1), date(2023, 12, 31))),
            }
        )
    return rows


def _gen_customers(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        first = random.choice(_FIRST_NAMES)
        last = random.choice(_LAST_NAMES)
        rows.append(
            {
                "customer_id": _uid(),
                "full_name": f"{first} {last}",
                "date_of_birth": str(_rand_date(date(1944, 1, 1), date(2006, 12, 31))),
                "postcode": _rand_postcode(),
                "email": _rand_email(first, last),
                "created_at": str(_rand_date(date(2018, 1, 1), date(2024, 6, 30))),
            }
        )
    return rows


def _gen_policies(
    n: int,
    customer_ids: list[str],
    agent_ids: list[str],
) -> list[dict]:
    rows = []
    for _ in range(n):
        start = _rand_date(date(2018, 1, 1), date(2024, 1, 1))
        # Match ins_prod_v3.meta: end_date null_rate=85.3%
        # generate_insurance_data.py: is_active = rng.random() > 0.15  (85% active)
        # Active policies have no end_date → 85% null
        is_active = random.random() > 0.15
        end = None if is_active else str(start + timedelta(days=random.choice([365, 730, 1095])))
        policy_type = random.choice(POLICY_TYPES)
        prem_lo, prem_hi = _PREMIUM_RANGE[policy_type]
        rows.append(
            {
                "policy_id": _uid(),
                "customer_id": random.choice(customer_ids),
                "agent_id": random.choice(agent_ids),  # null_rate=0.0% per meta
                "policy_type": policy_type,
                "premium_amt": round(random.uniform(prem_lo, prem_hi), 2),
                "start_date": str(start),
                "end_date": end,
                "is_active": 1 if is_active else 0,
                "created_at": str(start),
            }
        )
    return rows


def _gen_claims(n: int, policies: list[dict]) -> list[dict]:
    adjuster_pool = [_uid() for _ in range(50)]
    rows = []
    for _ in range(n):
        policy = random.choice(policies)
        claim_date = _rand_date(date(2019, 1, 1), date(2024, 6, 30))
        status = random.choice(CLAIM_STATUSES)  # equal weights — matches PG generator

        # resolved_at: null only for open claims (~25% null — matches meta 24.9%)
        resolved_at = None
        if status in ("approved", "rejected", "paid"):
            resolved_at = str(claim_date + timedelta(days=random.randint(1, 90)))

        rows.append(
            {
                "claim_id": _uid(),
                "policy_id": policy["policy_id"],
                "claim_date": str(claim_date),
                # null_rate=7.9% per meta (was 18% in old SQLite generator)
                "claim_amount": _maybe_null(_sample_claim_amount(policy["policy_type"]), 0.08),
                "claim_status": status,
                # null_rate=12.3% per meta
                "adjuster_id": _maybe_null(random.choice(adjuster_pool), 0.12),
                "resolved_at": resolved_at,
                # null_rate=69.7% per meta
                "notes": _maybe_null(_rand_note(), 0.70),
            }
        )
    return rows


def _gen_payments(claim_rows: list[dict]) -> list[dict]:
    """One payment per approved or paid claim with a non-null claim_amount.

    Mirrors generate_insurance_data.py generate_payments() logic:
    approved + paid claims ≈ 50% of all claims; ~8% of those have null
    claim_amount → expected payments ≈ 46% of claims (matches meta: 9220/20000).
    """
    rows = []
    for claim in claim_rows:
        if claim["claim_status"] not in ("approved", "paid"):
            continue
        if claim["claim_amount"] is None:
            continue
        base = date.fromisoformat((claim["resolved_at"] or str(date.today()))[:10])
        paid_at = str(base + timedelta(days=random.randint(1, 14)))
        rows.append(
            {
                "payment_id": _uid(),
                "claim_id": claim["claim_id"],
                "paid_amount": round(float(claim["claim_amount"]) * random.uniform(0.5, 1.0), 2),
                "paid_at": paid_at,
                "payment_method": random.choice(PAYMENT_METHODS),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uid() -> str:
    # H6 FIX: str(uuid.uuid4())[:8] gives 8 hex chars = 4.3B unique values.
    # Birthday-paradox collision probability at 300k rows is ~100%; SQLite's
    # PRIMARY KEY constraint raises IntegrityError mid-generate(), corrupting
    # the output database.  uuid4().hex returns the full 32-char hex string
    # with effectively zero collision probability.
    return uuid.uuid4().hex


def _rand_date(start: date, end: date) -> date:
    return start + timedelta(days=random.randint(0, (end - start).days))


def _maybe_null(value: object, null_rate: float) -> object:
    return None if random.random() < null_rate else value


def _sample_claim_amount(policy_type: str) -> float:
    """Sample from the policy-type-specific lognormal distribution.

    Falls back to the health distribution for any unrecognised policy_type.
    """
    mu, sigma = _CLAIM_LOGNORM.get(policy_type, _CLAIM_LOGNORM["health"])
    raw = random.lognormvariate(mu, sigma)
    cap = _CLAIM_CAP.get(policy_type, 1_000_000.0)
    return round(max(_CLAIM_FLOOR, min(raw, cap)), 2)


def _rand_postcode() -> str:
    """Generate a plausible UK postcode (matches Faker en_GB output in generate_insurance_data.py)."""
    area = random.choice(
        [
            "M",
            "EC",
            "SW",
            "N",
            "E",
            "W",
            "SE",
            "NW",
            "EH",
            "G",
            "B",
            "LS",
            "CF",
            "BT",
            "AB",
        ]
    )
    dist = random.randint(1, 99)
    sector = random.randint(1, 9)
    unit = "".join(random.choices(string.ascii_uppercase, k=2))
    return f"{area}{dist} {sector}{unit}"[:10]


def _rand_email(first: str, last: str) -> str:
    suffix = random.randint(10, 99)
    domain = random.choice(["example.net", "example.com", "example.co.uk"])
    return f"{first.lower()}{suffix}@{domain}"


def _rand_note() -> str:
    return random.choice(_NOTES)


# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "James",
    "Mary",
    "Robert",
    "Patricia",
    "John",
    "Jennifer",
    "Michael",
    "Linda",
    "William",
    "Barbara",
    "David",
    "Susan",
    "Richard",
    "Jessica",
    "Joseph",
    "Sarah",
    "Thomas",
    "Karen",
    "Charles",
    "Lisa",
    "Christopher",
    "Nancy",
    "Daniel",
    "Betty",
    "Matthew",
    "Sandra",
    "Anthony",
    "Dorothy",
    "Mark",
    "Ashley",
    "Donald",
    "Kimberly",
]
_LAST_NAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Gonzalez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
    "Lee",
    "Perez",
    "Thompson",
    "White",
]
_NOTES = [
    "Claim verified by field inspector.",
    "Awaiting third-party documentation.",
    "Customer contacted; additional evidence requested.",
    "Referred to specialist assessor.",
    "Fast-tracked under policy clause 7.",
    "Disputed by insured.",
    "Settlement offer issued.",
    "Fraud screening triggered.",
    "Medical report pending.",
    "Structural survey commissioned.",
]


def _insert(conn: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ",".join(["?"] * len(cols))
    col_str = ",".join(cols)
    values = [tuple(r[c] for c in cols) for r in rows]
    conn.executemany(f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})", values)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic insurance SQLite database (ins_prod_v3 mirror)"
    )
    parser.add_argument("--output", default="data/insurance.db")
    parser.add_argument(
        "--rows",
        type=int,
        default=10_000,
        help="Number of customer rows (other tables scale proportionally)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(
        output_path=args.output,
        n_customers=args.rows,
        seed=args.seed,
    )
