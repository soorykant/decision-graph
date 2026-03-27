"""
models.py — all data shapes for the Decision Graph.

One file. No logic. Just types.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Raw input
# ---------------------------------------------------------------------------

class DecisionRecord(BaseModel):
    """One row from decisions.json exactly as given."""

    id: str
    timestamp: datetime
    entity: str
    decision: str                        # Buy | Sell | Hold | Pass
    rationale: str
    outcome: Optional[str] = None


# ---------------------------------------------------------------------------
# Graph node & edge types
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    DECISION = "decision"   # the 52 records
    ENTITY   = "entity"     # 8 companies
    FACTOR   = "factor"     # recurring themes: "management credibility", "lithium price"
    EVENT    = "event"      # discrete occurrences: "Northvale Grocers bankruptcy"


class EdgeType(str, Enum):
    ABOUT      = "about"       # Decision  → Entity   (always present)
    INFLUENCED = "influenced"  # Factor    → Decision (LLM-extracted)
    REVERSES   = "reverses"    # Decision  → Decision (explicit callbacks)
    REFERENCES = "references"  # Decision  → Entity   (cross-company mentions)
    CAUSED     = "caused"      # Event     → Factor   (e.g. bankruptcy → concentration risk)


# ---------------------------------------------------------------------------
# What the LLM extracts from a single rationale
# ---------------------------------------------------------------------------

class ExtractedFactor(BaseModel):
    """A recurring theme or condition that drove the decision."""

    name: str = Field(
        description="Short canonical name, e.g. 'management credibility', 'lithium price', "
                    "'customer concentration'. Use consistent names across decisions."
    )
    category: str = Field(
        description="One of: macro | fundamental | management | catalyst | risk | valuation"
    )
    sentiment: str = Field(
        description="One of: positive | negative | neutral — relative to the decision made"
    )


class ExtractedEvent(BaseModel):
    """A discrete, named occurrence referenced in the rationale."""

    name: str = Field(
        description="Short name, e.g. 'Northvale Grocers bankruptcy', 'Phase 3 readout', "
                    "'Ironbridge activist exit'. Specific, not generic."
    )
    event_type: str = Field(
        description="One of: earnings | management_change | contract | regulatory | "
                    "competitor | market | activist | clinical_trial | accounting"
    )


class ExtractedReversal(BaseModel):
    """When this decision explicitly reverses or references a prior decision."""

    referenced_decision_id: str = Field(
        description="The ID of the prior decision being reversed or referenced, e.g. 'D6'. "
                    "Only include if explicitly called out in the rationale."
    )
    reversal_type: str = Field(
        description="One of: reverses | reinforces | updates"
    )
    reason: str = Field(
        description="One sentence on why."
    )


class ExtractedCrossReference(BaseModel):
    """When this decision references events at a *different* entity."""

    entity: str = Field(
        description="The other company being referenced, e.g. 'Orbitra Networks'."
    )
    reason: str = Field(
        description="One sentence on what the cross-reference is about."
    )


class GraphElements(BaseModel):
    """
    Everything the LLM extracts from a single DecisionRecord rationale.
    This is the contract between extractor.py and graph_builder.py.
    """

    decision_id: str
    factors: list[ExtractedFactor] = Field(
        default_factory=list,
        description="Key factors that drove this decision. Typically 2–4."
    )
    events: list[ExtractedEvent] = Field(
        default_factory=list,
        description="Discrete named events referenced. Typically 0–3."
    )
    reversals: list[ExtractedReversal] = Field(
        default_factory=list,
        description="Prior decisions this one explicitly reverses or references. Often 0–1."
    )
    cross_references: list[ExtractedCrossReference] = Field(
        default_factory=list,
        description="Other entities mentioned as influencing context. Often 0–2."
    )
    decision_theme: str = Field(
        description="One sentence capturing the core logic of this decision."
    )


# ---------------------------------------------------------------------------
# Query results
# ---------------------------------------------------------------------------

class LineageNode(BaseModel):
    """One hop in a decision's ancestry or forward impact."""

    node_id: str
    node_type: NodeType
    label: str
    edge_type: EdgeType
    direction: str   # "upstream" | "downstream"
    depth: int


class LineageResult(BaseModel):
    """Full ancestry + forward impact for a decision."""

    decision_id: str
    upstream: list[LineageNode]    # factors/events/decisions that led here
    downstream: list[LineageNode]  # decisions this one influenced


class PatternQueryResult(BaseModel):
    """Result of a pattern query like 'what factors drove Sell decisions?'"""

    query: str
    matches: list[dict]            # flexible — each query returns different shapes
    summary: str                   # one-sentence human-readable answer


class WhatIfResult(BaseModel):
    """Result of a what-if counterfactual query."""

    factor_name: str
    hypothetical_change: str
    affected_decision_ids: list[str]
    impact_assessment: str         # LLM-generated reasoning
    confidence: str                # "high" | "medium" | "low"


# ---------------------------------------------------------------------------
# Graph persistence
# ---------------------------------------------------------------------------

class GraphMetadata(BaseModel):
    """Stored alongside the serialised graph."""

    built_at: datetime
    total_nodes: int
    total_edges: int
    node_type_counts: dict[str, int]
    edge_type_counts: dict[str, int]
    entities: list[str]
