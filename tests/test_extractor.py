"""
test_extractor.py — validate LLM extraction logic without making API calls.

Tests: data loading, prompt construction, response parsing,
       cross-reference filtering, reversal detection.
"""

import pytest
from extractor import (
    _FALLBACK_ENTITIES,
    derive_entities,
    _parse_elements,
    build_prompt,
    load_decisions,
)
from models import GraphElements


class TestLoadDecisions:
    def test_loads_all_records(self):
        records = load_decisions("decisions.json")
        assert len(records) == 52

    def test_record_ids_sequential(self):
        records = load_decisions("decisions.json")
        ids = [r.id for r in records]
        assert ids[0] == "D1"
        assert ids[-1] == "D52"

    def test_all_decision_types_present(self):
        records = load_decisions("decisions.json")
        actions = {r.decision for r in records}
        assert actions == {"Buy", "Sell", "Hold", "Pass"}

    def test_all_entities_covered(self):
        records = load_decisions("decisions.json")
        entities = {r.entity for r in records}
        assert len(entities) == 8

    def test_rationale_non_empty(self):
        records = load_decisions("decisions.json")
        for r in records:
            assert r.rationale.strip(), f"Empty rationale for {r.id}"


class TestBuildPrompt:
    def test_prompt_contains_decision_id(self, all_records):
        prompt = build_prompt(all_records[0], [r.id for r in all_records])
        assert "D1" in prompt

    def test_prompt_contains_entity(self, all_records):
        prompt = build_prompt(all_records[0], [r.id for r in all_records])
        assert "Corvex" in prompt

    def test_prompt_contains_all_decision_ids(self, all_records):
        all_ids = [r.id for r in all_records]
        prompt = build_prompt(all_records[0], all_ids)
        assert "D52" in prompt

    def test_prompt_contains_known_entities(self, all_records):
        prompt = build_prompt(all_records[0], [])
        for entity in _FALLBACK_ENTITIES:
            assert entity in prompt

    def test_prompt_contains_rationale(self, all_records):
        prompt = build_prompt(all_records[0], [])
        # rationale from D1 contains these phrases
        assert "45%" in prompt and "mid-market merchants" in prompt


class TestParseElements:
    def test_basic_parse(self):
        raw = {
            "decision_id": "D1",
            "factors": [
                {"name": "revenue momentum", "category": "fundamental", "sentiment": "positive"}
            ],
            "events": [],
            "reversals": [],
            "cross_references": [],
            "decision_theme": "Strong growth thesis.",
        }
        elements = _parse_elements(raw, "D1")
        assert isinstance(elements, GraphElements)
        assert elements.decision_id == "D1"
        assert len(elements.factors) == 1
        assert elements.factors[0].name == "revenue momentum"

    def test_multiple_factors(self):
        raw = {
            "decision_id": "D22",
            "factors": [
                {"name": "management credibility", "category": "management", "sentiment": "negative"},
                {"name": "revenue momentum", "category": "fundamental", "sentiment": "negative"},
            ],
            "events": [{"name": "revenue restatement", "event_type": "accounting"}],
            "reversals": [],
            "cross_references": [],
            "decision_theme": "Exiting on accounting fraud signals.",
        }
        elements = _parse_elements(raw, "D22")
        assert len(elements.factors) == 2
        assert len(elements.events) == 1
        assert elements.events[0].name == "revenue restatement"

    def test_reversal_parsed(self):
        raw = {
            "decision_id": "D17",
            "factors": [{"name": "customer concentration", "category": "risk", "sentiment": "positive"}],
            "events": [{"name": "second airframer contract", "event_type": "contract"}],
            "reversals": [
                {
                    "referenced_decision_id": "D6",
                    "reversal_type": "reverses",
                    "reason": "Diversification condition now met.",
                }
            ],
            "cross_references": [],
            "decision_theme": "Reversing original pass.",
        }
        elements = _parse_elements(raw, "D17")
        assert len(elements.reversals) == 1
        assert elements.reversals[0].referenced_decision_id == "D6"
        assert elements.reversals[0].reversal_type == "reverses"

    def test_cross_reference_valid_entity_kept(self):
        raw = {
            "decision_id": "D12",
            "factors": [{"name": "backhaul cost advantage", "category": "fundamental", "sentiment": "positive"}],
            "events": [],
            "reversals": [],
            "cross_references": [
                {"entity": "Orbitra Networks", "reason": "Half the cost of terrestrial fibre."}
            ],
            "decision_theme": "Adding Tessellate on Orbitra cost advantage.",
        }
        elements = _parse_elements(raw, "D12")
        assert len(elements.cross_references) == 1
        assert elements.cross_references[0].entity == "Orbitra Networks"

    def test_cross_reference_unknown_entity_filtered(self):
        raw = {
            "decision_id": "D12",
            "factors": [{"name": "backhaul cost", "category": "fundamental", "sentiment": "positive"}],
            "events": [],
            "reversals": [],
            "cross_references": [
                {"entity": "SomeFakeCorp Inc.", "reason": "Should be filtered."},
                {"entity": "Orbitra Networks", "reason": "Should be kept."},
            ],
            "decision_theme": "Test.",
        }
        elements = _parse_elements(raw, "D12")
        assert len(elements.cross_references) == 1
        assert elements.cross_references[0].entity == "Orbitra Networks"

    def test_malformed_factor_does_not_crash(self):
        """Parser must never raise on malformed input — at worst it skips bad items."""
        raw = {
            "decision_id": "D1",
            "factors": [
                {"name": "good factor", "category": "macro", "sentiment": "positive"},
                {"bad_key": "missing name field"},  # malformed
            ],
            "events": [],
            "reversals": [],
            "cross_references": [],
            "decision_theme": "Test resilience.",
        }
        elements = _parse_elements(raw, "D1")
        assert isinstance(elements, GraphElements)
        assert len(elements.factors) >= 1

    def test_empty_response_graceful(self):
        raw = {
            "decision_id": "D8",
            "factors": [],
            "events": [],
            "reversals": [],
            "cross_references": [],
            "decision_theme": "",
        }
        elements = _parse_elements(raw, "D8")
        assert elements.decision_id == "D8"
        assert len(elements.factors) == 0

    def test_known_entities_list_complete(self):
        assert len(_FALLBACK_ENTITIES) == 8
        assert "Corvex Payments" in _FALLBACK_ENTITIES
        assert "Orbitra Networks" in _FALLBACK_ENTITIES
        assert "Tessellate Data Systems" in _FALLBACK_ENTITIES

    def test_derived_entities_match_the_dataset(self):
        entities = derive_entities(load_decisions("decisions.json"))
        assert len(entities) == len(set(entities)) == 8
        assert set(entities) == set(_FALLBACK_ENTITIES)
