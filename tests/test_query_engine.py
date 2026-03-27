"""
test_query_engine.py — validate all six query functions.

Tests: correct results, edge cases, pattern_query dispatcher routing.
"""

import pytest

from query_engine import (
    cross_entity_links,
    decisions_by_factor,
    factors_for_decision_type,
    lineage,
    pattern_query,
    reversed_decisions,
)


class TestFactorsForDecisionType:
    def test_sell_decisions_have_factors(self, graph):
        result = factors_for_decision_type(graph, "Sell")
        assert result["decision_count"] > 0
        assert len(result["factors"]) > 0

    def test_buy_decisions_most_common(self, graph):
        result = factors_for_decision_type(graph, "Buy")
        # revenue momentum should appear in Buy decisions
        factor_names = [f["factor"] for f in result["factors"]]
        assert "revenue momentum" in factor_names

    def test_factors_ranked_by_count(self, graph):
        result = factors_for_decision_type(graph, "Buy")
        counts = [f["decision_count"] for f in result["factors"]]
        assert counts == sorted(counts, reverse=True), "Factors not sorted by count"

    def test_unknown_action_returns_empty(self, graph):
        result = factors_for_decision_type(graph, "Merge")
        assert result["decision_count"] == 0
        assert result["factors"] == []

    def test_case_insensitive(self, graph):
        r1 = factors_for_decision_type(graph, "sell")
        r2 = factors_for_decision_type(graph, "Sell")
        r3 = factors_for_decision_type(graph, "SELL")
        assert r1["decision_count"] == r2["decision_count"] == r3["decision_count"]

    def test_top_n_respected(self, graph):
        result = factors_for_decision_type(graph, "Buy", top_n=3)
        assert len(result["factors"]) <= 3

    def test_result_has_required_keys(self, graph):
        result = factors_for_decision_type(graph, "Sell")
        assert "query" in result
        assert "decision_type" in result
        assert "decision_count" in result
        assert "factors" in result
        assert "summary" in result

    def test_factor_entry_has_required_keys(self, graph):
        result = factors_for_decision_type(graph, "Sell")
        if result["factors"]:
            f = result["factors"][0]
            assert "factor" in f
            assert "category" in f
            assert "decision_count" in f
            assert "influenced_decisions" in f


class TestDecisionsByFactor:
    def test_management_credibility_found(self, graph):
        result = decisions_by_factor(graph, "management credibility")
        assert result["decision_count"] >= 1

    def test_partial_match_works(self, graph):
        result = decisions_by_factor(graph, "management")
        # Should match both "management credibility" and "management quality"
        assert result["decision_count"] >= 2
        assert len(result["matched_factors"]) >= 2

    def test_nonexistent_factor_graceful(self, graph):
        result = decisions_by_factor(graph, "totally_fake_factor_xyz")
        assert result["decision_count"] == 0
        assert result["decisions"] == []
        assert "decision_count" in result  # key must exist even for empty result

    def test_lithium_price_linked_to_vantage(self, graph):
        result = decisions_by_factor(graph, "lithium price")
        entities = {d["entity"] for d in result["decisions"]}
        assert "Solvane Energy Storage" in entities

    def test_decisions_sorted_by_timestamp(self, graph):
        result = decisions_by_factor(graph, "revenue momentum")
        timestamps = [d["timestamp"] for d in result["decisions"]]
        assert timestamps == sorted(timestamps)

    def test_action_breakdown_present(self, graph):
        result = decisions_by_factor(graph, "management credibility")
        assert "action_breakdown" in result
        assert isinstance(result["action_breakdown"], dict)

    def test_result_includes_sentiment(self, graph):
        result = decisions_by_factor(graph, "management credibility")
        for d in result["decisions"]:
            assert "factor_sentiment" in d


class TestLineage:
    def test_d22_upstream_factors(self, graph):
        result = lineage(graph, "D22")
        assert result["decision"]["action"] == "Sell"
        factor_labels = [u["label"] for u in result["upstream"] if u["type"] == "factor"]
        assert "management credibility" in factor_labels

    def test_d22_upstream_events(self, graph):
        result = lineage(graph, "D22")
        event_labels = [u["label"] for u in result["upstream"] if u["type"] == "event"]
        assert "revenue restatement" in event_labels

    def test_d17_has_reversal_downstream(self, graph):
        """D17 reverses D6 — D6 should appear in D17's downstream."""
        result = lineage(graph, "D17")
        downstream_ids = [d.get("node_id", "") for d in result["downstream"]]
        assert any("D6" in did for did in downstream_ids)

    def test_nonexistent_decision_returns_error(self, graph):
        result = lineage(graph, "D999")
        assert "error" in result

    def test_lineage_result_has_required_keys(self, graph):
        result = lineage(graph, "D1")
        assert "decision_id" in result
        assert "decision" in result
        assert "upstream" in result
        assert "downstream" in result
        assert "summary" in result

    def test_decision_dict_has_action(self, graph):
        result = lineage(graph, "D1")
        assert "action" in result["decision"]
        assert result["decision"]["action"] == "Buy"

    def test_upstream_nodes_have_depth(self, graph):
        result = lineage(graph, "D22")
        for node in result["upstream"]:
            assert "depth" in node
            assert node["depth"] >= 1

    def test_case_insensitive_id(self, graph):
        r1 = lineage(graph, "D22")
        r2 = lineage(graph, "d22")
        assert r1["decision_id"] == r2["decision_id"]


class TestCrossEntityLinks:
    def test_atlas_helios_connected(self, graph):
        result = cross_entity_links(graph)
        pairs = [(c["entity_a"], c["entity_b"]) for c in result["connections"]]
        atlas_helios = any(
            ("Tessellate Data Systems" in p and "Orbitra Networks" in p)
            for p in pairs
        )
        assert atlas_helios, "Tessellate/Orbitra connection not found"

    def test_connections_sorted_by_strength(self, graph):
        result = cross_entity_links(graph)
        strengths = [c["connection_strength"] for c in result["connections"]]
        assert strengths == sorted(strengths, reverse=True)

    def test_result_has_required_keys(self, graph):
        result = cross_entity_links(graph)
        assert "total_connections" in result
        assert "connections" in result
        assert "summary" in result

    def test_connection_entry_has_required_keys(self, graph):
        result = cross_entity_links(graph)
        if result["connections"]:
            c = result["connections"][0]
            assert "entity_a" in c
            assert "entity_b" in c
            assert "connection_strength" in c
            assert "shared_factors" in c

    def test_no_self_connections(self, graph):
        result = cross_entity_links(graph)
        for c in result["connections"]:
            assert c["entity_a"] != c["entity_b"]


class TestReversedDecisions:
    def test_d17_reverses_d6(self, graph):
        result = reversed_decisions(graph)
        assert result["count"] >= 1
        ids = [(r["reversing_decision"], r["reversed_decision"]) for r in result["reversals"]]
        assert ("D17", "D6") in ids

    def test_reversals_sorted_by_timestamp(self, graph):
        result = reversed_decisions(graph)
        timestamps = [r["reversing_timestamp"] for r in result["reversals"]]
        assert timestamps == sorted(timestamps)

    def test_reversal_has_required_keys(self, graph):
        result = reversed_decisions(graph)
        if result["reversals"]:
            r = result["reversals"][0]
            assert "reversing_decision" in r
            assert "reversed_decision" in r
            assert "reversal_type" in r
            assert "reason" in r


class TestPatternQueryDispatcher:
    def test_routes_sell_to_factors(self, graph):
        result = pattern_query(graph, "what factors drove sell decisions")
        assert result.get("decision_type") == "Sell"

    def test_routes_buy_to_factors(self, graph):
        result = pattern_query(graph, "factors behind buy decisions")
        assert result.get("decision_type") == "Buy"

    def test_routes_factor_name_to_decisions_by_factor(self, graph):
        result = pattern_query(graph, "management credibility")
        assert "matched_factors" in result

    def test_routes_decision_id_to_lineage(self, graph):
        result = pattern_query(graph, "tell me about D22")
        assert result.get("decision_id") == "D22"

    def test_routes_reversals_keyword(self, graph):
        result = pattern_query(graph, "show all reversals")
        assert "reversals" in result

    def test_routes_cross_entity(self, graph):
        result = pattern_query(graph, "cross entity connections")
        assert "connections" in result

    def test_unknown_query_falls_back_to_factor_search(self, graph):
        result = pattern_query(graph, "something completely unknown xyz")
        # Falls back to decisions_by_factor which returns empty gracefully
        assert "matched_factors" in result or "decision_count" in result
