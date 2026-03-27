"""
test_graph_builder.py — validate graph construction, factor normalisation,
                         node/edge types, and save/load round-trip.
"""

import json
import os
import tempfile

import networkx as nx
import pytest

from graph_builder import (
    FactorNormalizer,
    build_graph,
    decision_node_id,
    entity_node_id,
    factor_node_id,
    graph_summary,
    load_graph,
    save_graph,
)
from models import EdgeType, NodeType


class TestNodeIdHelpers:
    def test_decision_node_id(self):
        assert decision_node_id("D1") == "decision:D1"
        assert decision_node_id("D52") == "decision:D52"

    def test_entity_node_id_normalises(self):
        assert entity_node_id("Corvex Payments") == "entity:corvex_payments"
        assert entity_node_id("Tessellate Data Systems") == "entity:tessellate_data_systems"

    def test_factor_node_id_normalises(self):
        assert factor_node_id("Management Credibility") == "factor:management_credibility"
        assert factor_node_id("lithium price") == "factor:lithium_price"


class TestFactorNormalizer:
    def test_exact_match_deduped(self):
        norm = FactorNormalizer()
        c1 = norm.normalize("management credibility")
        c2 = norm.normalize("management credibility")
        assert c1 == c2

    def test_near_duplicate_deduped(self):
        norm = FactorNormalizer(threshold=0.92)
        c1 = norm.normalize("lithium price")
        c2 = norm.normalize("lithium prices")  # ~0.963 similarity
        assert c1 == c2

    def test_distinct_concepts_not_merged(self):
        norm = FactorNormalizer(threshold=0.92)
        c1 = norm.normalize("management credibility")
        c2 = norm.normalize("management quality")  # 0.75 similarity — distinct
        assert c1 != c2

    def test_trailing_whitespace_ignored(self):
        norm = FactorNormalizer()
        c1 = norm.normalize("revenue momentum")
        c2 = norm.normalize("  revenue momentum  ")
        assert c1 == c2

    def test_case_insensitive(self):
        norm = FactorNormalizer()
        c1 = norm.normalize("Revenue Momentum")
        c2 = norm.normalize("revenue momentum")
        assert c1 == c2

    def test_all_canonicals_property(self):
        norm = FactorNormalizer()
        norm.normalize("lithium price")
        norm.normalize("management credibility")
        norm.normalize("lithium prices")  # deduped with lithium price at default threshold
        assert len(norm.all_canonicals) <= 2


class TestBuildGraph:
    def test_graph_is_directed(self, graph):
        assert isinstance(graph, nx.DiGraph)

    def test_all_node_types_present(self, graph):
        node_types = {d.get("type") for _, d in graph.nodes(data=True)}
        assert NodeType.DECISION in node_types
        assert NodeType.ENTITY in node_types
        assert NodeType.FACTOR in node_types

    def test_52_decision_nodes(self, graph):
        decisions = [n for n, d in graph.nodes(data=True) if d.get("type") == NodeType.DECISION]
        assert len(decisions) == 52

    def test_8_entity_nodes(self, graph):
        entities = [n for n, d in graph.nodes(data=True) if d.get("type") == NodeType.ENTITY]
        assert len(entities) == 8

    def test_all_edge_types_present(self, graph):
        edge_types = {d.get("type") for _, _, d in graph.edges(data=True)}
        assert EdgeType.ABOUT in edge_types
        assert EdgeType.INFLUENCED in edge_types

    def test_every_decision_has_about_edge(self, graph):
        """Every decision node must connect to exactly one entity via ABOUT."""
        for node, data in graph.nodes(data=True):
            if data.get("type") != NodeType.DECISION:
                continue
            about_edges = [
                (u, v, d) for u, v, d in graph.out_edges(node, data=True)
                if d.get("type") == EdgeType.ABOUT
            ]
            assert len(about_edges) == 1, f"{node} has {len(about_edges)} ABOUT edges (expected 1)"

    def test_factor_dedup_across_decisions(self, graph):
        """Same factor mentioned in multiple decisions → single node."""
        factor_labels = [
            d.get("label", "") for _, d in graph.nodes(data=True)
            if d.get("type") == NodeType.FACTOR
        ]
        # revenue momentum appears in many decisions but must be one node
        assert factor_labels.count("revenue momentum") == 1

    def test_reversal_edges_present(self, graph):
        """D17 reverses D6 — REVERSES edge must exist."""
        d17 = decision_node_id("D17")
        d6 = decision_node_id("D6")
        assert graph.has_node(d17)
        assert graph.has_node(d6)
        reversal_edges = [
            (u, v, d) for u, v, d in graph.edges(data=True)
            if d.get("type") == EdgeType.REVERSES
            and u == d17 and v == d6
        ]
        assert len(reversal_edges) == 1

    def test_cross_reference_edges(self, graph):
        """D2 references Tessellate Data Systems → REFERENCES edge should exist."""
        d2 = decision_node_id("D2")
        atlas = entity_node_id("Tessellate Data Systems")
        assert graph.has_node(d2)
        assert graph.has_node(atlas)
        refs = [
            d for u, v, d in graph.edges(data=True)
            if u == d2 and v == atlas and d.get("type") == EdgeType.REFERENCES
        ]
        assert len(refs) == 1

    def test_decision_node_attributes(self, graph):
        """Decision nodes must carry all required fields."""
        d1 = decision_node_id("D1")
        assert graph.has_node(d1)
        data = graph.nodes[d1]
        assert data["type"] == NodeType.DECISION
        assert data["entity"] == "Corvex Payments"
        assert data["action"] == "Buy"
        assert "rationale" in data
        assert "timestamp" in data

    def test_management_credibility_shared(self, graph):
        """management credibility must connect to D22, D28, D31 (all in mock extractions)."""
        fnode = factor_node_id("management credibility")
        assert graph.has_node(fnode), "management credibility factor node missing"
        influenced = [
            graph.nodes[s].get("decision_id")
            for s in graph.successors(fnode)
            if graph.nodes[s].get("type") == NodeType.DECISION
        ]
        assert "D22" in influenced
        assert "D28" in influenced


class TestGraphSummaryAndPersistence:
    def test_graph_summary_format(self, graph):
        summary = graph_summary(graph)
        assert "nodes" in summary.lower()
        assert "edges" in summary.lower()

    def test_save_and_reload(self, graph):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            meta = save_graph(graph, path)
            assert meta.total_nodes == graph.number_of_nodes()
            assert meta.total_edges == graph.number_of_edges()

            G2 = load_graph(path)
            assert G2.number_of_nodes() == graph.number_of_nodes()
            assert G2.number_of_edges() == graph.number_of_edges()

        finally:
            os.unlink(path)

    def test_reload_restores_enum_types(self, graph):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_graph(graph, path)
            G2 = load_graph(path)

            node_types = {d.get("type") for _, d in G2.nodes(data=True)}
            assert NodeType.DECISION in node_types
            assert NodeType.ENTITY in node_types

            edge_types = {d.get("type") for _, _, d in G2.edges(data=True)}
            assert EdgeType.ABOUT in edge_types

        finally:
            os.unlink(path)

    def test_metadata_counts_correct(self, graph):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            meta = save_graph(graph, path)
            assert meta.node_type_counts.get("decision") == 52
            assert meta.node_type_counts.get("entity") == 8
            assert len(meta.entities) == 8
        finally:
            os.unlink(path)
