"""
extractor.py — one job: DecisionRecord → GraphElements.

Calls the LLM once per decision with a structured-output prompt.
Stateless. No side effects. Easy to test.

Usage:
    elements = extract(record, client)
    all_elements = extract_all(records, client)
"""

from __future__ import annotations

import json
import time
import re
from typing import Optional

from models import (
    DecisionRecord,
    ExtractedCrossReference,
    ExtractedEvent,
    ExtractedFactor,
    ExtractedReversal,
    GraphElements,
)

DEFAULT_MODEL = "claude-sonnet-5-5"


def call_claude(client, model: str, system: str, prompt: str, max_tokens: int = 8000) -> str:
    """
    Send one prompt and return the reply text.

    Runs at low effort (this is structured extraction, not open-ended reasoning) and
    turns on server-side refusal fallback. Text is read by block type, because a
    response can start with thinking blocks. Raises ValueError on a refusal or an
    empty reply, so callers can retry.
    """
    response = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,  # room for thinking plus the JSON reply
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        raise ValueError(f"Model declined the request (request {response._request_id})")
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise ValueError(f"Empty reply (stop_reason={response.stop_reason})")
    return text


def derive_entities(records: list[DecisionRecord]) -> list[str]:
    """
    Derive unique entity names from input decisions, preserving first-seen order.
    Makes the system generic — works on any dataset, not just the 8 hardcoded companies.
    """
    seen = set()
    entities = []
    for record in records:
        name = (record.entity or "").strip()
        if name and name not in seen:
            seen.add(name)
            entities.append(name)
    return entities


# Fallback for tests and direct calls that don't pass records
_FALLBACK_ENTITIES = [
    "Corvex Payments",
    "Thistledown Foods",
    "Kestrion Avionics",
    "Solvane Energy Storage",
    "Orbitra Networks",
    "Pinehollow Biologics",
    "Halvard Freight Rail",
    "Tessellate Data Systems",
]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial analyst assistant that extracts structured data from
investment decision rationales. You always return valid JSON and nothing else.

Your job is to read one investment decision rationale and extract:
1. factors — the 2-4 key themes or conditions that drove this decision
2. events — discrete named occurrences referenced (earnings, contracts, regulatory actions, etc.)
3. reversals — if this decision changes or references a prior position on the SAME entity
4. cross_references — other portfolio companies mentioned as influencing context
5. decision_theme — one sentence capturing the core logic

FACTOR CATEGORIES (use exactly one): macro | fundamental | management | catalyst | risk | valuation
FACTOR SENTIMENT: positive | negative | neutral (relative to the direction of the decision)
EVENT TYPES (use exactly one): earnings | management_change | contract | regulatory | competitor | market | activist | clinical_trial | accounting
REVERSAL TYPES: reverses | reinforces | updates

RULES:
- Factor names must be short and canonical (2-4 words). Use EXACTLY these names when they apply:
  "management credibility", "revenue momentum", "customer concentration", "lithium price",
  "gross margin", "backhaul cost", "valuation pullback", "activist catalyst", "activist exit",
  "satellite capacity", "supplier dependence", "binary clinical risk", "regulatory risk",
  "competitive pressure", "free cash flow", "leadership change", "fuel cost"
  For anything else, keep names to 2-4 words, lowercase, no filler words.
- Events must be specific named occurrences, not generic descriptions.
  Good: "Northvale Grocers bankruptcy", "Phase 3 readout", "second airframer contract", "Ironbridge activist exit"
  Bad:  "negative news", "a contract announcement", "management changes"
- Reversals: look for language like "reversing our earlier", "re-entering", "exiting again",
  "adding to", "returning to", "our original thesis", "we previously passed".
  When found, identify which prior decision on this same entity is being reversed using the
  prior decisions list provided. Set referenced_decision_id to the most recent prior decision
  on the same entity that this one is changing.
- cross_references are ONLY for other portfolio companies explicitly mentioned.
- Return ONLY a JSON object. No markdown, no explanation."""

USER_PROMPT_TEMPLATE = """Decision ID: {decision_id}
Entity: {entity}
Action: {decision}
Timestamp: {timestamp}
Rationale: {rationale}
Outcome: {outcome}

Prior decisions on {entity} (for reversal detection — use these IDs):
{prior_same_entity}

All portfolio companies (for cross-reference detection):
{entities}

Return JSON matching this exact schema:
{{
  "decision_id": "{decision_id}",
  "factors": [
    {{"name": "string", "category": "macro|fundamental|management|catalyst|risk|valuation", "sentiment": "positive|negative|neutral"}}
  ],
  "events": [
    {{"name": "string", "event_type": "earnings|management_change|contract|regulatory|competitor|market|activist|clinical_trial|accounting"}}
  ],
  "reversals": [
    {{"referenced_decision_id": "D##", "reversal_type": "reverses|reinforces|updates", "reason": "one sentence"}}
  ],
  "cross_references": [
    {{"entity": "string", "reason": "one sentence"}}
  ],
  "decision_theme": "string"
}}"""


def build_prompt(
    record: DecisionRecord,
    all_decision_ids: Optional[list[str]] = None,
    all_records: Optional[list] = None,
    known_entities: Optional[list[str]] = None,
) -> str:
    # Pass prior decisions on the SAME entity only — much more useful for
    # reversal detection than a flat list of all IDs.
    if all_records:
        ts = record.timestamp if isinstance(record.timestamp, str) else record.timestamp.isoformat()
        prior_same = [
            f"  {r.id} ({r.decision}, {(r.timestamp if isinstance(r.timestamp, str) else r.timestamp.isoformat())[:10]})"
            for r in all_records
            if r.entity == record.entity and r.id != record.id
            and (r.timestamp if isinstance(r.timestamp, str) else r.timestamp.isoformat()) < ts
        ]
        prior_same_str = "\n".join(prior_same) if prior_same else "  (none — first decision on this entity)"
    else:
        prior_same_str = "  " + ", ".join(all_decision_ids or []) if all_decision_ids else "  (none)"

    entities = known_entities or _FALLBACK_ENTITIES

    return USER_PROMPT_TEMPLATE.format(
        decision_id=record.id,
        entity=record.entity,
        decision=record.decision,
        timestamp=record.timestamp if isinstance(record.timestamp, str) else record.timestamp.isoformat(),
        rationale=record.rationale,
        outcome=record.outcome or "Unknown",
        entities="\n".join(f"- {e}" for e in entities),
        prior_same_entity=prior_same_str,
    )


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract(
    record: DecisionRecord,
    client,  # anthropic.Anthropic client
    all_decision_ids: Optional[list[str]] = None,
    all_records: Optional[list] = None,
    known_entities: Optional[list[str]] = None,
    model: str = DEFAULT_MODEL,
    retries: int = 2,
) -> GraphElements:
    """
    Extract graph elements from one DecisionRecord.
    Retries on JSON parse failure up to `retries` times.
    """
    prompt = build_prompt(
        record,
        all_decision_ids,
        all_records=all_records,
        known_entities=known_entities,
    )

    for attempt in range(retries + 1):
        try:
            raw = call_claude(client, model, SYSTEM_PROMPT, prompt)

            # Strip markdown fences if the model adds them despite instructions
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)
            return _parse_elements(data, record.id, known_entities=known_entities)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to extract elements for {record.id} after {retries + 1} attempts: {e}"
                )
            time.sleep(1)


def _parse_elements(
    data: dict,
    decision_id: str,
    known_entities: Optional[list[str]] = None,   # Optional — falls back to _FALLBACK_ENTITIES
) -> GraphElements:
    """Parse raw LLM JSON into GraphElements. Lenient — skips malformed sub-items."""
    _entities = known_entities or _FALLBACK_ENTITIES

    factors = []
    for f in data.get("factors", []):
        try:
            factors.append(ExtractedFactor(**f))
        except Exception:
            pass

    events = []
    for e in data.get("events", []):
        try:
            events.append(ExtractedEvent(**e))
        except Exception:
            pass

    reversals = []
    for r in data.get("reversals", []):
        try:
            reversals.append(ExtractedReversal(**r))
        except Exception:
            pass

    cross_refs = []
    for c in data.get("cross_references", []):
        try:
            if c.get("entity") in _entities and c.get("entity") != "":
                cross_refs.append(ExtractedCrossReference(**c))
        except Exception:
            pass

    return GraphElements(
        decision_id=decision_id,
        factors=factors,
        events=events,
        reversals=reversals,
        cross_references=cross_refs,
        decision_theme=data.get("decision_theme", ""),
    )


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_all(
    records: list[DecisionRecord],
    client,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> list[GraphElements]:
    """
    Extract graph elements for all records.
    Derives entity names from the dataset — works on any input, not just the
    hardcoded 8 companies. Passes all_records so each call can see prior
    decisions on the same entity for accurate reversal detection.
    """
    all_ids = [r.id for r in records]
    known_entities = derive_entities(records)
    results = []

    for i, record in enumerate(records):
        if verbose:
            print(f"  Extracting {record.id} ({i + 1}/{len(records)}) — {record.entity} {record.decision}...")
        elements = extract(
            record,
            client,
            all_decision_ids=all_ids,
            all_records=records,
            known_entities=known_entities,
            model=model,
        )
        results.append(elements)

    return results


# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------

def load_decisions(path: str) -> list[DecisionRecord]:
    """Load decisions.json and return validated DecisionRecord list."""
    with open(path) as f:
        raw = json.load(f)
    return [DecisionRecord(**d) for d in raw["decisions"]]