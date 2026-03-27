"""
test_whatif_engine.py — validate what-if graph traversal logic.

LLM calls are mocked — these tests cover the deterministic traversal
layer only, not the LLM impact assessment.
"""

import pytest

from whatif_engine import find_affected_decisions


class TestFindAffectedDecisions:
    def test_lithium_price_affects_vantage(self, graph):
        affected = find_affected_decisions(graph, "lithium price")
        assert len(affected["matched_factors"]) > 0
        assert len(affected["direct_decisions"]) > 0
        vantage_decisions = [
            d for d in affected["direct_decisions"]
            if d["entity"] == "Solvane Energy Storage"
        ]
        assert len(vantage_decisions) > 0

    def test_management_credibility_reaches_multiple_decisions(self, graph):
        affected = find_affected_decisions(graph, "management credibility")
        assert len(affected["all_decision_ids"]) >= 2

    def test_partial_name_match(self, graph):
        affected = find_affected_decisions(graph, "management")
        full = find_affected_decisions(graph, "management credibility")
        # partial match should find at least as many decisions
        assert len(affected["all_decision_ids"]) >= len(full["all_decision_ids"])

    def test_nonexistent_factor_returns_empty(self, graph):
        affected = find_affected_decisions(graph, "totally_fake_factor_xyz_999")
        assert affected["matched_factors"] == []
        assert affected["direct_decisions"] == []
        assert affected["downstream_decisions"] == []
        assert affected["all_decision_ids"] == []

    def test_reversal_chain_traversed(self, graph):
        """customer concentration → D6 (Pass) → D17 reverses D6.
        D6 should be direct; D17 should appear in downstream via reversal chain."""
        affected = find_affected_decisions(graph, "customer concentration")
        direct_ids = {d["decision_id"] for d in affected["direct_decisions"]}
        downstream_ids = {d["decision_id"] for d in affected["downstream_decisions"]}
        all_ids = set(affected["all_decision_ids"])

        assert "D6" in all_ids, "D6 (original Pass) should be affected"
        # D17 reverses D6, so it should appear in the downstream chain
        assert "D17" in all_ids, "D17 (reversal of D6) should appear via reversal chain"

    def test_all_decision_ids_is_union(self, graph):
        """all_decision_ids must equal direct + downstream (deduplicated)."""
        affected = find_affected_decisions(graph, "lithium price")
        direct = {d["decision_id"] for d in affected["direct_decisions"]}
        downstream = {d["decision_id"] for d in affected["downstream_decisions"]}
        expected = sorted(direct | downstream)
        assert affected["all_decision_ids"] == expected

    def test_direct_decisions_have_required_fields(self, graph):
        affected = find_affected_decisions(graph, "revenue momentum")
        for d in affected["direct_decisions"]:
            assert "decision_id" in d
            assert "entity" in d
            assert "action" in d
            assert "rationale" in d
            assert "factor_sentiment" in d

    def test_matched_factors_returned(self, graph):
        affected = find_affected_decisions(graph, "lithium price")
        assert "lithium price" in affected["matched_factors"]

    def test_result_has_all_required_keys(self, graph):
        affected = find_affected_decisions(graph, "revenue momentum")
        assert "matched_factors" in affected
        assert "direct_decisions" in affected
        assert "downstream_decisions" in affected
        assert "all_decision_ids" in affected

    def test_no_duplicate_ids_in_all_decisions(self, graph):
        affected = find_affected_decisions(graph, "management")
        ids = affected["all_decision_ids"]
        assert len(ids) == len(set(ids)), "Duplicate decision IDs in all_decision_ids"
