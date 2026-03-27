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

def derive_entities(records: list[DecisionRecord]) -> list[str]:
    """
    Derive unique entity names from input decisions, preserving first-seen order.
    """
    seen = set()
    entities = []
    for record in records:
        name = (record.entity or "").strip()
        if name and name not in seen:
            seen.add(name)
            entities.append(name)
    return entities

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial analyst assistant that extracts structured data from
investment decision rationales. You always return valid JSON and nothing else.

Your job is to read one investment decision rationale and extract:
1. factors — the 2-4 key themes or conditions that drove this decision
2. events — discrete named occurrences referenced (earnings, contracts, regulatory actions, etc.)
3. reversals — if this decision explicitly reverses or references a prior decision by ID
4. cross_references — other portfolio companies mentioned as influencing context
5. decision_theme — one sentence capturing the core logic

FACTOR CATEGORIES (use exactly one): macro | fundamental | management | catalyst | risk | valuation
FACTOR SENTIMENT: positive | negative | neutral (relative to the direction of the decision)
EVENT TYPES (use exactly one): earnings | management_change | contract | regulatory | competitor | market | activist | clinical_trial | accounting
REVERSAL TYPES: reverses | reinforces | updates

RULES:
- Factor names must be short and canonical (2-4 words). Use consistent names across decisions.
  Good: "management credibility", "customer concentration", "lithium price", "revenue momentum"
  Bad:  "concerns about management", "the fact that lithium prices fell"
- Events must be specific, not generic.
  Good: "Northvale Grocers bankruptcy", "Phase 3 readout", "Ironbridge activist exit", "second airframer contract"
  Bad:  "negative news", "a contract", "some management changes"
- Only extract reversals when the rationale explicitly calls back to a prior decision.
- cross_references are ONLY for other companies in the portfolio being mentioned as context.
- Return ONLY a JSON object. No markdown, no explanation."""

USER_PROMPT_TEMPLATE = """Decision ID: {decision_id}
Entity: {entity}
Action: {decision}
Timestamp: {timestamp}
Rationale: {rationale}
Outcome: {outcome}

All portfolio companies (for cross-reference detection):
{entities}

Prior decision IDs in this portfolio (for reversal detection): {all_ids}

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
    {{"referenced_decision_id": "D##", "reversal_type": "reverses|reinforces|updates", "reason": "string"}}
  ],
  "cross_references": [
    {{"entity": "string", "reason": "string"}}
  ],
  "decision_theme": "string"
}}"""


def build_prompt(
    record: DecisionRecord,
    known_entities: list[str],
    all_decision_ids: Optional[list[str]] = None,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        decision_id=record.id,
        entity=record.entity,
        decision=record.decision,
        timestamp=record.timestamp if isinstance(record.timestamp, str) else record.timestamp.isoformat(),
        rationale=record.rationale,
        outcome=record.outcome or "Unknown",
        entities="\n".join(f"- {e}" for e in known_entities),
        all_ids=", ".join(all_decision_ids or []),
    )


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract(
    record: DecisionRecord,
    client,  # anthropic.Anthropic client
    all_decision_ids: Optional[list[str]] = None,
    known_entities: Optional[list[str]] = None,
    model: str = "claude-sonnet-4-20250514",
    retries: int = 2,
) -> GraphElements:
    """
    Extract graph elements from one DecisionRecord.
    Retries on JSON parse failure up to `retries` times.
    """
    known_entities = known_entities or [record.entity]
    prompt = build_prompt(record, known_entities, all_decision_ids)

    for attempt in range(retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Strip markdown fences if the model adds them despite instructions
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)
            return _parse_elements(data, record.id, known_entities)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to extract elements for {record.id} after {retries + 1} attempts: {e}"
                )
            time.sleep(1)


def _parse_elements(data: dict, decision_id: str, known_entities: list[str]) -> GraphElements:
    """Parse raw LLM JSON into GraphElements. Lenient — skips malformed sub-items."""
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
            # Only accept entities we actually know about
            if c.get("entity") in known_entities and c.get("entity") != "":
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
    model: str = "claude-sonnet-4-20250514",
    verbose: bool = True,
) -> list[GraphElements]:
    """
    Extract graph elements for all records.
    Passes all decision IDs so the LLM can detect reversals accurately.
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
