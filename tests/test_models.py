"""
test_models.py — validate all Pydantic models load and validate correctly.
"""

import pytest
from models import (
    DecisionRecord,
    EdgeType,
    ExtractedCrossReference,
    ExtractedEvent,
    ExtractedFactor,
    ExtractedReversal,
    GraphElements,
    GraphMetadata,
    LineageResult,
    NodeType,
    PatternQueryResult,
    WhatIfResult,
)


class TestDecisionRecord:
    def test_basic_fields(self):
        r = DecisionRecord(
            id="D1",
            timestamp="2024-01-15T10:00:00Z",
            entity="Corvex Payments",
            decision="Buy",
            rationale="Strong revenue growth.",
            outcome="Stock +60%.",
        )
        assert r.id == "D1"
        assert r.entity == "Corvex Payments"
        assert r.decision == "Buy"

    def test_optional_outcome(self):
        r = DecisionRecord(
            id="D37",
            timestamp="2025-01-22T09:00:00Z",
            entity="Corvex Payments",
            decision="Pass",
            rationale="SEC overhang too risky.",
        )
        assert r.outcome is None

    def test_all_decision_types(self):
        for action in ["Buy", "Sell", "Hold", "Pass"]:
            r = DecisionRecord(
                id="D1", timestamp="2024-01-01T00:00:00Z",
                entity="Test Co", decision=action, rationale="Test."
            )
            assert r.decision == action


class TestExtractedFactor:
    def test_valid_factor(self):
        f = ExtractedFactor(
            name="management credibility",
            category="management",
            sentiment="negative",
        )
        assert f.name == "management credibility"
        assert f.category == "management"
        assert f.sentiment == "negative"

    def test_all_categories(self):
        for cat in ["macro", "fundamental", "management", "catalyst", "risk", "valuation"]:
            f = ExtractedFactor(name="test", category=cat, sentiment="neutral")
            assert f.category == cat


class TestExtractedEvent:
    def test_valid_event(self):
        e = ExtractedEvent(name="Northvale Grocers bankruptcy", event_type="accounting")
        assert e.name == "Northvale Grocers bankruptcy"

    def test_all_event_types(self):
        for et in ["earnings", "management_change", "contract", "regulatory",
                   "competitor", "market", "activist", "clinical_trial", "accounting"]:
            e = ExtractedEvent(name="test", event_type=et)
            assert e.event_type == et


class TestExtractedReversal:
    def test_reversal(self):
        rv = ExtractedReversal(
            referenced_decision_id="D6",
            reversal_type="reverses",
            reason="second airframer contract reduces customer concentration.",
        )
        assert rv.referenced_decision_id == "D6"
        assert rv.reversal_type == "reverses"


class TestGraphElements:
    def test_complete_elements(self):
        g = GraphElements(
            decision_id="D17",
            factors=[
                ExtractedFactor(name="customer concentration", category="risk", sentiment="positive"),
            ],
            events=[
                ExtractedEvent(name="second airframer contract", event_type="contract"),
            ],
            reversals=[
                ExtractedReversal(referenced_decision_id="D6", reversal_type="reverses", reason="Diversification met."),
            ],
            cross_references=[],
            decision_theme="Reversing the D6 pass after the second airframer contract.",
        )
        assert g.decision_id == "D17"
        assert len(g.factors) == 1
        assert len(g.events) == 1
        assert len(g.reversals) == 1
        assert g.reversals[0].referenced_decision_id == "D6"

    def test_empty_elements(self):
        g = GraphElements(
            decision_id="D8",
            factors=[],
            events=[],
            reversals=[],
            cross_references=[],
            decision_theme="Passing on biotech binary.",
        )
        assert g.decision_id == "D8"
        assert len(g.factors) == 0


class TestNodeAndEdgeTypes:
    def test_node_types(self):
        assert NodeType.DECISION.value == "decision"
        assert NodeType.ENTITY.value == "entity"
        assert NodeType.FACTOR.value == "factor"
        assert NodeType.EVENT.value == "event"

    def test_edge_types(self):
        assert EdgeType.ABOUT.value == "about"
        assert EdgeType.INFLUENCED.value == "influenced"
        assert EdgeType.REVERSES.value == "reverses"
        assert EdgeType.REFERENCES.value == "references"
        assert EdgeType.CAUSED.value == "caused"

    def test_node_type_enumeration(self):
        values = [e.value for e in NodeType]
        assert "decision" in values
        assert "entity" in values
        assert "factor" in values
        assert "event" in values

    def test_edge_type_enumeration(self):
        values = [e.value for e in EdgeType]
        assert len(values) == 5


class TestWhatIfResult:
    def test_whatif_result(self):
        r = WhatIfResult(
            factor_name="lithium price",
            hypothetical_change="drops from $80 to $50",
            affected_decision_ids=["D4", "D13", "D20"],
            impact_assessment="Three Solvane Energy Storage decisions would be materially affected.",
            confidence="high",
        )
        assert r.factor_name == "lithium price"
        assert len(r.affected_decision_ids) == 3
        assert r.confidence == "high"
