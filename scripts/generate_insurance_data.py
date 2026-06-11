"""
generate_insurance_data.py — Synthetic Insurance Dataset Generator
Data Analyst Copilot · Section 12.3

Generates the synthetic insurance schema defined in Section 12.3 and
populates it with realistic data at the specified scale:

  customers   :  10,000 rows  (PII-flagged: full_name, dob, postcode, email)
  agents      :   1,000 rows
  policies    :  50,000 rows
  claims      : 200,000 rows  (~8% null claim_amount, ~12% null adjuster_id)
  payments    : ~140,000 rows (one per approved/paid claim)

Intentionally absent columns (UNRESOLVED_REFERENCE test targets):
  churn_rate, loss_ratio, combined_ratio, profit_margin, risk_score

Usage
-----
  # Install deps first:
  pip install faker sqlalchemy psycopg2-binary tqdm --break-system-packages

  # Populate a running PostgreSQL instance:
  python generate_insurance_data.py \
      --db-url postgresql://user:pass@localhost:5432/analyst_copilot \
      --scale full

  # Dry-run (schema creation only, no data):
  python generate_insurance_data.py --db-url ... --scale schema-only

  # Mini scale for unit tests (100 customers, 500 policies, 2000 claims):
  python generate_insurance_data.py --db-url ... --scale mini
"""

from __future__ import annotations

import argparse
import hashlib
import random
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from faker import Faker
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.engine import Engine
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Scale presets
# ---------------------------------------------------------------------------


@dataclass
class ScaleConfig:
    n_customers: int
    n_agents: int
    n_policies: int
    n_claims: int


SCALES: dict[str, ScaleConfig] = {
    "mini": ScaleConfig(100, 20, 500, 2_000),
    "small": ScaleConfig(1_000, 100, 5_000, 20_000),
    "full": ScaleConfig(10_000, 1_000, 50_000, 200_000),
    "schema-only": ScaleConfig(0, 0, 0, 0),
}

POLICY_TYPES = ["auto", "home", "life", "health"]
CLAIM_STATUSES = ["open", "approved", "rejected", "paid"]
REGIONS = ["North", "South", "East", "West", "Central"]
PAYMENT_METHODS = ["bank_transfer", "cheque", "direct_debit", "card"]

# Null rates matching Section 12.3 spec
NULL_CLAIM_AMOUNT_RATE = 0.08
NULL_ADJUSTER_ID_RATE = 0.12
NULL_END_DATE_RATE = 0.15  # active policies have no end_date

# ---------------------------------------------------------------------------
# Actuarial claim amount distributions — Issue #3 fix
# ---------------------------------------------------------------------------
# Previously all policy types used rng.uniform(100, 50_000) → E[amount] ≈ $25K
# for every type, producing ~$309 spread across all GROUP BY policy_type
# queries (i.e. statistically indistinguishable).
#
# Real insurance: life payouts >> health >> home >> auto.  We model each as
# lognormal(mu, sigma) where E[amount] = exp(mu + σ²/2):
#
#   policy_type | target mean | mu     | sigma | rationale
#   ------------|-------------|--------|-------|-------------------------------
#   auto        |  ~$8 K      |  8.667 | 0.80  | fender-benders dominate;
#               |             |        |       | tail for total-loss events
#   home        | ~$18 K      |  9.393 | 0.90  | structural repairs; tail for
#               |             |        |       | fire/flood total-loss
#   health      | ~$30 K      |  9.809 | 1.00  | outpatient-heavy; long tail
#               |             |        |       | for surgical/ICU stays
#   life        | ~$120 K     | 11.450 | 0.70  | face-value payouts; tight σ
#               |             |        |       | reflects standardised benefits
#
# Lognormal is appropriate: right-skewed, always positive, matches empirical
# claim severity distributions in actuarial literature.
# ---------------------------------------------------------------------------
_CLAIM_LOGNORM: dict[str, tuple[float, float]] = {
    "auto": (8.667, 0.80),
    "home": (9.393, 0.90),
    "health": (9.809, 1.00),
    "life": (11.450, 0.70),
}
_CLAIM_FLOOR: float = 100.0  # minimum payable claim (deductible floor)
_CLAIM_CAP: dict[str, float] = {  # hard cap to prevent astronomic outliers
    "auto": 200_000.0,
    "home": 750_000.0,
    "health": 1_000_000.0,
    "life": 3_000_000.0,
}

# ---------------------------------------------------------------------------
# Actuarial premium distributions — Issue #4 fix
# ---------------------------------------------------------------------------
# Previously all policy types used rng.uniform(200, 5_000) → E[premium] ≈ $2,600
# regardless of policy type.  Combined with type-specific claim distributions
# (_CLAIM_LOGNORM above) this produced loss_ratios of 11–170× — actuarially
# implausible and outside the [0, 2.0] validation range.
#
# Root cause: synthetic claim frequency (~3.68 non-null claims per policy) is
# 8–10× higher than actuarial reality.  Premiums must be set to match this
# synthetic frequency rather than real-world annual rates.  Ranges are chosen
# so that E[premium] ≈ (claims/policy × E[claim]) / 0.85 — targeting a loss
# ratio of ~0.85 (plausible, in-range, with headroom below the 2.0 ceiling).
#
#   policy_type | E[claim]  | claims/policy | E[premium] | range
#   ------------|-----------|---------------|------------|------------------
#   auto        |  ~$8 K    |     3.68      |  ~$34.6 K  | uniform(17K, 52K)
#   home        | ~$18 K    |     3.68      |  ~$77.9 K  | uniform(39K, 117K)
#   health      | ~$30 K    |     3.68      | ~$129.9 K  | uniform(65K, 195K)
#   life        | ~$120 K   |     3.68      | ~$519.4 K  | uniform(260K, 779K)
#
# If claim frequency is ever reduced to actuarially accurate rates these
# ranges should be re-calibrated accordingly.
# ---------------------------------------------------------------------------
_PREMIUM_RANGE: dict[str, tuple[float, float]] = {
    "auto": (17_000.0, 52_000.0),
    "home": (39_000.0, 117_000.0),
    "health": (65_000.0, 195_000.0),
    "life": (260_000.0, 779_000.0),
}

# ---------------------------------------------------------------------------
# Schema DDL (Section 12.3)
# ---------------------------------------------------------------------------


def build_schema(metadata: MetaData) -> dict[str, Table]:
    customers = Table(
        "customers",
        metadata,
        Column("customer_id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        Column("full_name", String(100), nullable=False),  # pii_flagged
        Column("date_of_birth", Date),  # pii_flagged
        Column("postcode", String(10)),  # pii_flagged
        Column("email", String(200)),  # pii_flagged
        Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
    )
    agents = Table(
        "agents",
        metadata,
        Column("agent_id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        Column("agent_code", String(10), nullable=False, unique=True),
        Column("region", String(50)),
        Column("hired_at", Date),
    )
    policies = Table(
        "policies",
        metadata,
        Column("policy_id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        Column(
            "customer_id",
            UUID(as_uuid=True),
            ForeignKey("customers.customer_id"),
            nullable=False,
        ),
        Column("agent_id", UUID(as_uuid=True), ForeignKey("agents.agent_id")),
        Column("policy_type", String(50), nullable=False),
        Column("premium_amt", Numeric(10, 2)),
        Column("start_date", Date, nullable=False),
        Column("end_date", Date),
        Column("is_active", Boolean, nullable=False, default=True),
        Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
    )
    claims = Table(
        "claims",
        metadata,
        Column("claim_id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        Column(
            "policy_id",
            UUID(as_uuid=True),
            ForeignKey("policies.policy_id"),
            nullable=False,
        ),
        Column("claim_date", Date, nullable=False),
        Column("claim_amount", Numeric(12, 2)),  # nullable ~8%
        Column("claim_status", String(20), nullable=False),
        Column("adjuster_id", UUID(as_uuid=True)),  # nullable ~12%
        Column("resolved_at", DateTime),
        Column("notes", Text),  # pii_flagged (free-text)
    )
    payments = Table(
        "payments",
        metadata,
        Column("payment_id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        Column(
            "claim_id",
            UUID(as_uuid=True),
            ForeignKey("claims.claim_id"),
            nullable=False,
        ),
        Column("paid_amount", Numeric(12, 2), nullable=False),
        Column("paid_at", DateTime, nullable=False),
        Column("payment_method", String(30)),
    )
    return {
        "customers": customers,
        "agents": agents,
        "policies": policies,
        "claims": claims,
        "payments": payments,
    }


# ---------------------------------------------------------------------------
# Row generators
# ---------------------------------------------------------------------------


def _hash_pii(value: str) -> str:
    """SHA-256 hash used for embedding-safe PII masking (Section 2A)."""
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def generate_customers(n: int, fake: Faker, rng: random.Random) -> list[dict]:
    rows = []
    base_date = datetime(2018, 1, 1)
    for _ in range(n):
        rows.append(
            {
                "customer_id": uuid.uuid4(),
                "full_name": fake.name(),
                "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=80),
                "postcode": fake.postcode()[:10],
                "email": fake.email(),
                "created_at": base_date
                + timedelta(days=rng.randint(0, (datetime.now() - base_date).days)),
            }
        )
    return rows


def generate_agents(n: int, fake: Faker, rng: random.Random) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "agent_id": uuid.uuid4(),
                "agent_code": f"AGT{i:05d}",
                "region": rng.choice(REGIONS),
                "hired_at": fake.date_between(start_date="-10y", end_date="today"),
            }
        )
    return rows


def generate_policies(
    n: int,
    customer_ids: list[uuid.UUID],
    agent_ids: list[uuid.UUID],
    fake: Faker,
    rng: random.Random,
) -> list[dict]:
    rows = []
    for _ in range(n):
        start = fake.date_between(start_date="-5y", end_date="today")
        is_active = rng.random() > NULL_END_DATE_RATE
        end = None if is_active else start + timedelta(days=rng.randint(180, 1095))
        policy_type = rng.choice(POLICY_TYPES)
        prem_lo, prem_hi = _PREMIUM_RANGE[policy_type]
        rows.append(
            {
                "policy_id": uuid.uuid4(),
                "customer_id": rng.choice(customer_ids),
                "agent_id": rng.choice(agent_ids),
                "policy_type": policy_type,
                "premium_amt": round(rng.uniform(prem_lo, prem_hi), 2),
                "start_date": start,
                "end_date": end,
                "is_active": is_active,
                "created_at": datetime.combine(start, datetime.min.time()),
            }
        )
    return rows


def _sample_claim_amount(policy_type: str, rng: random.Random) -> float:
    """Return a claim amount sampled from the policy-type-specific lognormal.

    Uses the seeded ``rng`` instance so the overall dataset is fully
    reproducible from ``--seed``.  Falls back to the health distribution for
    any unrecognised policy_type rather than raising KeyError.
    """
    mu, sigma = _CLAIM_LOGNORM.get(policy_type, _CLAIM_LOGNORM["health"])
    raw = rng.lognormvariate(mu, sigma)
    cap = _CLAIM_CAP.get(policy_type, 1_000_000.0)
    return round(max(_CLAIM_FLOOR, min(raw, cap)), 2)


def generate_claims(
    n: int,
    policy_rows: list[dict],
    fake: Faker,
    rng: random.Random,
) -> list[dict]:
    rows = []
    adjuster_pool = [uuid.uuid4() for _ in range(50)]  # 50 adjusters

    for _ in range(n):
        policy = rng.choice(policy_rows)
        policy_start: date = policy["start_date"]
        claim_date = fake.date_between(
            start_date=policy_start,
            end_date=date.today(),
        )
        status = rng.choice(CLAIM_STATUSES)
        resolved_at = None
        if status in ("approved", "rejected", "paid"):
            resolved_at = datetime.combine(
                claim_date + timedelta(days=rng.randint(1, 90)),
                datetime.min.time(),
            )

        rows.append(
            {
                "claim_id": uuid.uuid4(),
                "policy_id": policy["policy_id"],
                "claim_date": claim_date,
                "claim_amount": (
                    None
                    if rng.random() < NULL_CLAIM_AMOUNT_RATE
                    else _sample_claim_amount(policy["policy_type"], rng)
                ),
                "claim_status": status,
                "adjuster_id": (
                    None if rng.random() < NULL_ADJUSTER_ID_RATE else rng.choice(adjuster_pool)
                ),
                "resolved_at": resolved_at,
                "notes": fake.sentence() if rng.random() < 0.3 else None,
            }
        )
    return rows


def generate_payments(
    claim_rows: list[dict],
    rng: random.Random,
) -> list[dict]:
    """One payment row per approved or paid claim."""
    rows = []
    for claim in claim_rows:
        if claim["claim_status"] not in ("approved", "paid"):
            continue
        if claim["claim_amount"] is None:
            continue
        paid_at = (claim["resolved_at"] or datetime.now(tz=UTC)) + timedelta(
            days=rng.randint(1, 14)
        )
        rows.append(
            {
                "payment_id": uuid.uuid4(),
                "claim_id": claim["claim_id"],
                "paid_amount": round(float(claim["claim_amount"]) * rng.uniform(0.5, 1.0), 2),
                "paid_at": paid_at,
                "payment_method": rng.choice(PAYMENT_METHODS),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Bulk insert helper
# ---------------------------------------------------------------------------

BATCH_SIZE = 1_000


def _bulk_insert(
    engine: Engine,
    table: Table,
    rows: list[dict],
    desc: str,
) -> None:
    if not rows:
        return
    with engine.begin() as conn:
        for i in tqdm(range(0, len(rows), BATCH_SIZE), desc=desc, unit="batch"):
            batch = rows[i : i + BATCH_SIZE]
            conn.execute(table.insert(), batch)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate(db_url: str, scale: str, seed: int = 42, drop_existing: bool = False) -> None:
    cfg = SCALES[scale]
    engine = create_engine(db_url)
    metadata = MetaData()
    tables = build_schema(metadata)

    print(f"[generate] Scale: {scale} | seed: {seed}")
    print(f"[generate] Connecting to: {db_url.split('@')[-1]}")

    if drop_existing:
        metadata.drop_all(engine, checkfirst=True)
        print("[generate] Dropped existing tables.")

    metadata.create_all(engine, checkfirst=True)
    print("[generate] Schema created.")

    if scale == "schema-only":
        print("[generate] Schema-only mode. No data inserted.")
        return

    fake = Faker("en_GB")
    Faker.seed(seed)
    rng = random.Random(seed)

    print(f"[generate] Generating {cfg.n_customers:,} customers…")
    customers = generate_customers(cfg.n_customers, fake, rng)
    _bulk_insert(engine, tables["customers"], customers, "customers")

    print(f"[generate] Generating {cfg.n_agents:,} agents…")
    agents = generate_agents(cfg.n_agents, fake, rng)
    _bulk_insert(engine, tables["agents"], agents, "agents")

    customer_ids = [r["customer_id"] for r in customers]
    agent_ids = [r["agent_id"] for r in agents]

    print(f"[generate] Generating {cfg.n_policies:,} policies…")
    policies = generate_policies(cfg.n_policies, customer_ids, agent_ids, fake, rng)
    _bulk_insert(engine, tables["policies"], policies, "policies")

    print(f"[generate] Generating {cfg.n_claims:,} claims…")
    claims = generate_claims(cfg.n_claims, policies, fake, rng)
    _bulk_insert(engine, tables["claims"], claims, "claims")

    print("[generate] Generating payments…")
    payments = generate_payments(claims, rng)
    _bulk_insert(engine, tables["payments"], payments, "payments")

    # Summary
    with engine.connect() as conn:
        for tname in ["customers", "agents", "policies", "claims", "payments"]:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {tname}")).scalar()
            print(f"  {tname:<12}: {count:>10,} rows")

    # Null rate validation
    with engine.connect() as conn:
        claim_nulls = conn.execute(
            text("SELECT COUNT(*) FROM claims WHERE claim_amount IS NULL")
        ).scalar()
        adj_nulls = conn.execute(
            text("SELECT COUNT(*) FROM claims WHERE adjuster_id IS NULL")
        ).scalar()
        total_claims = conn.execute(text("SELECT COUNT(*) FROM claims")).scalar()
        print(
            f"\n[validate] claim_amount null rate: "
            f"{claim_nulls / total_claims:.1%} (target ~{NULL_CLAIM_AMOUNT_RATE:.0%})"
        )
        print(
            f"[validate] adjuster_id null rate:  "
            f"{adj_nulls / total_claims:.1%} (target ~{NULL_ADJUSTER_ID_RATE:.0%})"
        )

    print("\n[generate] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic insurance data.")
    parser.add_argument("--db-url", required=True, help="SQLAlchemy database URL.")
    parser.add_argument(
        "--scale",
        choices=list(SCALES.keys()),
        default="full",
        help="Data scale preset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop and recreate all tables before inserting.",
    )
    args = parser.parse_args()
    generate(args.db_url, args.scale, args.seed, args.drop_existing)
