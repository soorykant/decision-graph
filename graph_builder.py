"""
graph_builder.py — GraphElements → NetworkX DiGraph.

One job: take all extracted elements, build a clean directed graph,
normalize entity names, and persist to JSON.

Usage:
    graph = build_graph(all_elements, all_records)
    save_graph(graph, "graph.json")
    graph = load_graph("graph.json")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Optional

import networkx as nx

from models import (
    DecisionRecord,
    EdgeType,
    GraphElements,
    GraphMetadata,
    NodeType,
)


# ---------------------------------------------------------------------------
# Node ID helpers — keep IDs predictable and collision-free
# ---------------------------------------------------------------------------

def decision_node_id(decision_id: str) -> str:
    return f"decision:{decision_id}"


def entity_node_id(entity_name: str) -> str:
    # Normalise to lowercase slug for consistent IDs
    slug = entity_name.lower().replace(" ", "_").replace(".", "")
    return f"entity:{slug}"


def factor_node_id(factor_name: str) -> str:
    slug = factor_name.lower().replace(" ", "_").replace("/", "_")
    return f"factor:{slug}"


def event_node_id(event_name: str) -> str:
    slug = event_name.lower().replace(" ", "_").replace("/", "_")
    return f"event:{slug}"


# ---------------------------------------------------------------------------
# Factor name normalisation — fuzzy dedup without a second LLM call
# ---------------------------------------------------------------------------

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class FactorNormalizer:
    """
    Keeps a registry of canonical factor names.
    When a new name is close enough to an existing one (>= threshold),
    it maps to the existing canonical name instead of creating a new node.
    """

    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold
        self._canonical: dict[str, str] = {}  # raw_name → canonical_name

    def normalize(self, name: str) -> str:
        name = name.strip().lower()

        # Exact match first
        if name in self._canonical:
            return self._canonical[name]

        # Fuzzy match against known canonicals
        for known in list(self._canonical.values()):
            if _similarity(name, known) >= self.threshold:
                self._canonical[name] = known
                return known

        # New canonical
        self._canonical[name] = name
        return name

    @property
    def all_canonicals(self) -> set[str]:
        return set(self._canonical.values())


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    all_elements: list[GraphElements],
    all_records: list[DecisionRecord],
) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from all extracted elements.

    Node attributes:
        type (NodeType), label (str), + type-specific fields

    Edge attributes:
        type (EdgeType), + optional metadata
    """
    G = nx.DiGraph()
    normalizer = FactorNormalizer()

    # Index records by id for quick lookup
    records_by_id: dict[str, DecisionRecord] = {r.id: r for r in all_records}

    # -----------------------------------------------------------------------
    # Pass 1: add all nodes
    # -----------------------------------------------------------------------
    for record in all_records:
        # Decision node
        G.add_node(
            decision_node_id(record.id),
            type=NodeType.DECISION,
            label=f"{record.decision} {record.entity} ({record.id})",
            decision_id=record.id,
            entity=record.entity,
            action=record.decision,
            timestamp=record.timestamp.isoformat() if hasattr(record.timestamp, "isoformat") else str(record.timestamp),
            rationale=record.rationale,
            outcome=record.outcome or "",
        )

        # Entity node (idempotent — multiple decisions share the same entity)
        eid = entity_node_id(record.entity)
        if not G.has_node(eid):
            G.add_node(
                eid,
                type=NodeType.ENTITY,
                label=record.entity,
                entity=record.entity,
            )

    for elements in all_elements:
        # Factor nodes
        for factor in elements.factors:
            canonical = normalizer.normalize(factor.name)
            fid = factor_node_id(canonical)
            if not G.has_node(fid):
                G.add_node(
                    fid,
                    type=NodeType.FACTOR,
                    label=canonical,
                    category=factor.category,
                )

        # Event nodes
        for event in elements.events:
            evid = event_node_id(event.name)
            if not G.has_node(evid):
                G.add_node(
                    evid,
                    type=NodeType.EVENT,
                    label=event.name,
                    event_type=event.event_type,
                )

    # -----------------------------------------------------------------------
    # Pass 2: add all edges
    # -----------------------------------------------------------------------
    for elements in all_elements:
        did = decision_node_id(elements.decision_id)
        record = records_by_id.get(elements.decision_id)
        if record is None:
            continue

        # ABOUT: Decision → Entity
        eid = entity_node_id(record.entity)
        G.add_edge(did, eid, type=EdgeType.ABOUT)

        # INFLUENCED: Factor → Decision
        for factor in elements.factors:
            canonical = normalizer.normalize(factor.name)
            fid = factor_node_id(canonical)
            G.add_edge(
                fid, did,
                type=EdgeType.INFLUENCED,
                sentiment=factor.sentiment,
                category=factor.category,
            )

        # CAUSED: Event → Decision (events influence decisions directly)
        for event in elements.events:
            evid = event_node_id(event.name)
            G.add_edge(
                evid, did,
                type=EdgeType.INFLUENCED,  # events influence decisions too
                event_type=event.event_type,
            )

        # REVERSES: Decision → Decision
        for reversal in elements.reversals:
            prior_did = decision_node_id(reversal.referenced_decision_id)
            if G.has_node(prior_did):
                G.add_edge(
                    did, prior_did,
                    type=EdgeType.REVERSES,
                    reversal_type=reversal.reversal_type,
                    reason=reversal.reason,
                )

        # REFERENCES: Decision → Entity (cross-company)
        for xref in elements.cross_references:
            xeid = entity_node_id(xref.entity)
            if G.has_node(xeid) and xeid != eid:  # don't self-reference
                G.add_edge(
                    did, xeid,
                    type=EdgeType.REFERENCES,
                    reason=xref.reason,
                )

    return G


# ---------------------------------------------------------------------------
# Graph persistence
# ---------------------------------------------------------------------------

def save_graph(G: nx.DiGraph, path: str) -> GraphMetadata:
    """Serialise graph + metadata to JSON."""
    # Count node and edge types
    node_type_counts: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        key = t.value if hasattr(t, "value") else str(t)
        node_type_counts[key] = node_type_counts.get(key, 0) + 1

    edge_type_counts: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        t = data.get("type", "unknown")
        key = t.value if hasattr(t, "value") else str(t)
        edge_type_counts[key] = edge_type_counts.get(key, 0) + 1

    entities = [
        data["entity"]
        for _, data in G.nodes(data=True)
        if data.get("type") == NodeType.ENTITY
    ]

    metadata = GraphMetadata(
        built_at=datetime.now(timezone.utc),
        total_nodes=G.number_of_nodes(),
        total_edges=G.number_of_edges(),
        node_type_counts=node_type_counts,
        edge_type_counts=edge_type_counts,
        entities=entities,
    )

    # Serialise — convert enums to strings for JSON
    graph_data = nx.node_link_data(G)
    _stringify_enums(graph_data)

    payload = {
        "metadata": metadata.model_dump() if hasattr(metadata, "model_dump") else _metadata_to_dict(metadata),
        "graph": graph_data,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return metadata


def load_graph(path: str) -> nx.DiGraph:
    """Load a previously saved graph from JSON."""
    with open(path) as f:
        payload = json.load(f)

    G = nx.node_link_graph(payload["graph"])

    # Restore enum types on nodes and edges
    for _, data in G.nodes(data=True):
        if "type" in data:
            try:
                data["type"] = NodeType(data["type"])
            except ValueError:
                pass

    for _, _, data in G.edges(data=True):
        if "type" in data:
            try:
                data["type"] = EdgeType(data["type"])
            except ValueError:
                pass

    return G


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stringify_enums(obj):
    """Recursively convert Enum values to their .value string in a dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, "value"):
                obj[k] = v.value
            else:
                _stringify_enums(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if hasattr(v, "value"):
                obj[i] = v.value
            else:
                _stringify_enums(v)


def _metadata_to_dict(meta: GraphMetadata) -> dict:
    return {
        "built_at": str(meta.built_at),
        "total_nodes": meta.total_nodes,
        "total_edges": meta.total_edges,
        "node_type_counts": meta.node_type_counts,
        "edge_type_counts": meta.edge_type_counts,
        "entities": meta.entities,
    }


def graph_summary(G: nx.DiGraph) -> str:
    """One-paragraph human-readable graph summary."""
    node_counts: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        key = t.value if hasattr(t, "value") else str(t)
        node_counts[key] = node_counts.get(key, 0) + 1

    edge_counts: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        t = data.get("type", "unknown")
        key = t.value if hasattr(t, "value") else str(t)
        edge_counts[key] = edge_counts.get(key, 0) + 1

    lines = [
        f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
        f"Nodes: {node_counts}",
        f"Edges: {edge_counts}",
    ]
    return "\n".join(lines)