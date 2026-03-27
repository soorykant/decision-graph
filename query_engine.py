"""
query_engine.py — pure graph traversal. No LLM. No side effects.

Five functions, each answering one type of question:
    1. factors_for_decision_type   — what factors drove Buy / Sell / Hold / Pass?
    2. decisions_by_factor         — every decision a factor touched
    3. lineage                     — full upstream + downstream for a decision
    4. cross_entity_links          — which companies are connected through shared decisions
    5. pattern_query               — dispatch by intent (wraps the above)

All functions accept a nx.DiGraph and return plain dicts / lists
so they're easy to serialise to JSON and serve from FastAPI.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx

from models import EdgeType, NodeType


# ---------------------------------------------------------------------------
# 1. What factors drove a decision type? (Buy / Sell / Hold / Pass)
# ---------------------------------------------------------------------------

def factors_for_decision_type(
    G: nx.DiGraph,
    decision_type: str,          # "Buy" | "Sell" | "Hold" | "Pass"
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Return the factors most commonly influencing a given decision action.

    Works by:
      - finding all Decision nodes with action == decision_type
      - walking their in-edges to find Factor predecessors
      - counting and ranking
    """
    decision_type = decision_type.strip().title()

    # Collect all matching decision node IDs
    target_decisions = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == NodeType.DECISION
        and d.get("action", "").title() == decision_type
    ]

    if not target_decisions:
        return {
            "query": f"Factors driving {decision_type} decisions",
            "decision_type": decision_type,
            "decision_count": 0,
            "factors": [],
            "summary": f"No {decision_type} decisions found in the graph.",
        }

    # Count how many decisions each factor influences
    factor_counts: dict[str, int] = defaultdict(int)
    factor_meta: dict[str, dict] = {}
    factor_decisions: dict[str, list[str]] = defaultdict(list)

    for dec_node in target_decisions:
        for pred in G.predecessors(dec_node):
            pred_data = G.nodes[pred]
            if pred_data.get("type") == NodeType.FACTOR:
                label = pred_data.get("label", pred)
                factor_counts[label] += 1
                factor_decisions[label].append(
                    G.nodes[dec_node].get("decision_id", dec_node)
                )
                if label not in factor_meta:
                    factor_meta[label] = {
                        "category": pred_data.get("category", "unknown"),
                        "node_id": pred,
                    }

    ranked = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    factors_out = []
    for factor_name, count in ranked:
        factors_out.append({
            "factor": factor_name,
            "category": factor_meta[factor_name]["category"],
            "decision_count": count,
            "influenced_decisions": sorted(set(factor_decisions[factor_name])),
        })

    top_factor = factors_out[0]["factor"] if factors_out else "none"
    summary = (
        f"Across {len(target_decisions)} {decision_type} decisions, "
        f"'{top_factor}' was the most influential factor "
        f"(appeared in {factors_out[0]['decision_count']} decisions)."
        if factors_out else f"No factors found for {decision_type} decisions."
    )

    return {
        "query": f"Factors driving {decision_type} decisions",
        "decision_type": decision_type,
        "decision_count": len(target_decisions),
        "factors": factors_out,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 2. Every decision a factor touched
# ---------------------------------------------------------------------------

def decisions_by_factor(
    G: nx.DiGraph,
    factor_name: str,            # partial match supported
) -> dict[str, Any]:
    """
    Find all decisions influenced by a factor (exact or partial name match).

    Useful for: "show me every decision where management credibility was a factor"
    """
    factor_name_lower = factor_name.lower()

    # Find matching factor nodes (partial match)
    matching_factors = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == NodeType.FACTOR
        and factor_name_lower in d.get("label", "").lower()
    ]

    if not matching_factors:
        return {
            "query": f"Decisions influenced by '{factor_name}'",
            "matched_factors": [],
            "decision_count": 0,
            "decisions": [],
            "action_breakdown": {},
            "summary": f"No factors matching '{factor_name}' found.",
        }

    all_decisions = []
    matched_factor_labels = []

    for fnode in matching_factors:
        flabel = G.nodes[fnode].get("label", fnode)
        matched_factor_labels.append(flabel)

        for succ in G.successors(fnode):
            succ_data = G.nodes[succ]
            if succ_data.get("type") == NodeType.DECISION:
                edge_data = G.edges[fnode, succ]
                all_decisions.append({
                    "decision_id": succ_data.get("decision_id", succ),
                    "entity": succ_data.get("entity", ""),
                    "action": succ_data.get("action", ""),
                    "timestamp": succ_data.get("timestamp", ""),
                    "outcome": succ_data.get("outcome", ""),
                    "factor_sentiment": edge_data.get("sentiment", ""),
                    "matched_factor": flabel,
                })

    # Sort by timestamp
    all_decisions.sort(key=lambda x: x["timestamp"])

    # Count outcomes
    action_counts: dict[str, int] = defaultdict(int)
    for d in all_decisions:
        action_counts[d["action"]] += 1

    summary = (
        f"Factor '{factor_name}' (matched: {', '.join(matched_factor_labels)}) "
        f"influenced {len(all_decisions)} decisions: "
        + ", ".join(f"{v} {k}" for k, v in sorted(action_counts.items()))
        + "."
    )

    return {
        "query": f"Decisions influenced by '{factor_name}'",
        "matched_factors": matched_factor_labels,
        "decision_count": len(all_decisions),
        "decisions": all_decisions,
        "action_breakdown": dict(action_counts),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 3. Full lineage for a decision
# ---------------------------------------------------------------------------

def lineage(
    G: nx.DiGraph,
    decision_id: str,            # e.g. "D22"
    max_depth: int = 3,
) -> dict[str, Any]:
    """
    For a given decision, return:
      - upstream: factors, events, and prior decisions that led to it
      - downstream: decisions this one influenced (via REVERSES edges)
      - entity context: other entities referenced

    This is the 'decision archaeology' function.
    """
    target_node = f"decision:{decision_id}"

    if not G.has_node(target_node):
        return {
            "decision_id": decision_id,
            "error": f"Decision {decision_id} not found in graph.",
        }

    node_data = G.nodes[target_node]

    # --- Upstream: BFS through predecessors ---
    upstream = []
    visited = {target_node}
    queue = [(target_node, 0)]

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        for pred in G.predecessors(current):
            if pred in visited:
                continue
            visited.add(pred)
            pred_data = G.nodes[pred]
            node_type = pred_data.get("type", "")
            edge_data = G.edges[pred, current]

            entry: dict[str, Any] = {
                "node_id": pred,
                "type": node_type.value if hasattr(node_type, "value") else str(node_type),
                "label": pred_data.get("label", pred),
                "edge_type": edge_data.get("type", "").value
                    if hasattr(edge_data.get("type", ""), "value")
                    else str(edge_data.get("type", "")),
                "depth": depth + 1,
            }

            # Type-specific enrichment
            if node_type == NodeType.FACTOR:
                entry["category"] = pred_data.get("category", "")
                entry["sentiment"] = edge_data.get("sentiment", "")
            elif node_type == NodeType.DECISION:
                entry["action"] = pred_data.get("action", "")
                entry["entity"] = pred_data.get("entity", "")
                entry["timestamp"] = pred_data.get("timestamp", "")
                # Continue BFS for decision chains
                queue.append((pred, depth + 1))
            elif node_type == NodeType.EVENT:
                entry["event_type"] = pred_data.get("event_type", "")

            upstream.append(entry)

    # --- Downstream: decisions that reference this one ---
    downstream = []
    for succ in G.successors(target_node):
        succ_data = G.nodes[succ]
        edge_data = G.edges[target_node, succ]
        edge_type = edge_data.get("type", "")
        edge_type_val = edge_type.value if hasattr(edge_type, "value") else str(edge_type)

        if edge_type_val == EdgeType.REVERSES.value:
            downstream.append({
                "node_id": succ,
                "type": "decision",
                "label": succ_data.get("label", succ),
                "action": succ_data.get("action", ""),
                "entity": succ_data.get("entity", ""),
                "timestamp": succ_data.get("timestamp", ""),
                "reversal_type": edge_data.get("reversal_type", ""),
                "reason": edge_data.get("reason", ""),
            })

    # Also check: which decisions have REVERSES edges pointing TO this node
    for pred in G.predecessors(target_node):
        pred_data = G.nodes[pred]
        edge_data = G.edges[pred, target_node]
        edge_type = edge_data.get("type", "")
        edge_type_val = edge_type.value if hasattr(edge_type, "value") else str(edge_type)

        if edge_type_val == EdgeType.REVERSES.value:
            # pred reversed target — add to downstream of target
            downstream.append({
                "node_id": pred,
                "type": "decision",
                "label": pred_data.get("label", pred),
                "action": pred_data.get("action", ""),
                "entity": pred_data.get("entity", ""),
                "timestamp": pred_data.get("timestamp", ""),
                "reversal_type": edge_data.get("reversal_type", ""),
                "reason": edge_data.get("reason", ""),
                "direction": "was_reversed_by",
            })

    # --- Cross-entity references ---
    cross_refs = []
    for succ in G.successors(target_node):
        succ_data = G.nodes[succ]
        edge_data = G.edges[target_node, succ]
        edge_type = edge_data.get("type", "")
        edge_type_val = edge_type.value if hasattr(edge_type, "value") else str(edge_type)

        if edge_type_val == EdgeType.REFERENCES.value:
            cross_refs.append({
                "entity": succ_data.get("entity", succ_data.get("label", "")),
                "reason": edge_data.get("reason", ""),
            })

    # --- Build summary ---
    factor_names = [u["label"] for u in upstream if u["type"] == "factor"]
    event_names = [u["label"] for u in upstream if u["type"] == "event"]

    summary_parts = [
        f"Decision {decision_id} ({node_data.get('action')} {node_data.get('entity')}):",
        f"  Driven by {len(factor_names)} factor(s): {', '.join(factor_names) or 'none'}.",
    ]
    if event_names:
        summary_parts.append(f"  Referenced events: {', '.join(event_names)}.")
    if downstream:
        summary_parts.append(f"  {len(downstream)} related decision(s) found.")
    if cross_refs:
        entities = [c["entity"] for c in cross_refs]
        summary_parts.append(f"  Cross-references: {', '.join(entities)}.")

    return {
        "decision_id": decision_id,
        "decision": {
            "action": node_data.get("action", ""),
            "entity": node_data.get("entity", ""),
            "timestamp": node_data.get("timestamp", ""),
            "rationale": node_data.get("rationale", ""),
            "outcome": node_data.get("outcome", ""),
        },
        "upstream": sorted(upstream, key=lambda x: x["depth"]),
        "downstream": downstream,
        "cross_references": cross_refs,
        "summary": "\n".join(summary_parts),
    }


# ---------------------------------------------------------------------------
# 4. Cross-entity links — which companies are connected?
# ---------------------------------------------------------------------------

def cross_entity_links(G: nx.DiGraph) -> dict[str, Any]:
    """
    Surface connections between portfolio companies:
      - Shared factors (both companies influenced by same factor)
      - Direct cross-references (Decision about A references Company B)
      - Shared events

    Returns a list of entity pairs with their connection types.
    """
    entities = {
        n: d for n, d in G.nodes(data=True)
        if d.get("type") == NodeType.ENTITY
    }

    # Map each factor/event to the entities whose decisions it influenced
    factor_to_entities: dict[str, set[str]] = defaultdict(set)
    event_to_entities: dict[str, set[str]] = defaultdict(set)

    for fnode, fdata in G.nodes(data=True):
        if fdata.get("type") not in (NodeType.FACTOR, NodeType.EVENT):
            continue

        for dec_node in G.successors(fnode):
            dec_data = G.nodes[dec_node]
            if dec_data.get("type") == NodeType.DECISION:
                entity = dec_data.get("entity", "")
                if entity:
                    if fdata.get("type") == NodeType.FACTOR:
                        factor_to_entities[fdata.get("label", fnode)].add(entity)
                    else:
                        event_to_entities[fdata.get("label", fnode)].add(entity)

    # Build entity-pair connections
    connections: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"shared_factors": [], "shared_events": [], "direct_references": []}
    )

    # Shared factors
    for factor_label, ents in factor_to_entities.items():
        ents_list = sorted(ents)
        for i, e1 in enumerate(ents_list):
            for e2 in ents_list[i + 1:]:
                key = (e1, e2)
                connections[key]["shared_factors"].append(factor_label)

    # Shared events
    for event_label, ents in event_to_entities.items():
        ents_list = sorted(ents)
        for i, e1 in enumerate(ents_list):
            for e2 in ents_list[i + 1:]:
                key = (e1, e2)
                connections[key]["shared_events"].append(event_label)

    # Direct cross-references (Decision → Entity via REFERENCES edge)
    for dec_node, dec_data in G.nodes(data=True):
        if dec_data.get("type") != NodeType.DECISION:
            continue
        dec_entity = dec_data.get("entity", "")

        for succ in G.successors(dec_node):
            edge_data = G.edges[dec_node, succ]
            edge_type = edge_data.get("type", "")
            edge_type_val = edge_type.value if hasattr(edge_type, "value") else str(edge_type)

            if edge_type_val == EdgeType.REFERENCES.value:
                ref_entity = G.nodes[succ].get("entity", G.nodes[succ].get("label", ""))
                if ref_entity and ref_entity != dec_entity:
                    pair = tuple(sorted([dec_entity, ref_entity]))
                    connections[pair]["direct_references"].append({
                        "from_decision": dec_data.get("decision_id", dec_node),
                        "reason": edge_data.get("reason", ""),
                    })

    # Format output — only pairs with at least one connection
    result = []
    for (e1, e2), conn in connections.items():
        total_links = (
            len(conn["shared_factors"])
            + len(conn["shared_events"])
            + len(conn["direct_references"])
        )
        if total_links == 0:
            continue
        result.append({
            "entity_a": e1,
            "entity_b": e2,
            "shared_factors": conn["shared_factors"],
            "shared_events": conn["shared_events"],
            "direct_references": conn["direct_references"],
            "connection_strength": total_links,
        })

    result.sort(key=lambda x: x["connection_strength"], reverse=True)

    summary = (
        f"Found {len(result)} entity pair connections. "
        f"Strongest: {result[0]['entity_a']} ↔ {result[0]['entity_b']} "
        f"({result[0]['connection_strength']} shared signals)."
        if result else "No cross-entity connections found."
    )

    return {
        "query": "Cross-entity connections",
        "total_connections": len(result),
        "connections": result,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 5. Reversed decisions — find all explicit reversals in the graph
# ---------------------------------------------------------------------------

def reversed_decisions(G: nx.DiGraph) -> dict[str, Any]:
    """
    Find all decisions that explicitly reversed a prior decision.
    Useful for: 'show me all position changes and why they happened'
    """
    reversals = []

    for u, v, data in G.edges(data=True):
        edge_type = data.get("type", "")
        edge_type_val = edge_type.value if hasattr(edge_type, "value") else str(edge_type)

        if edge_type_val == EdgeType.REVERSES.value:
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            reversals.append({
                "reversing_decision": u_data.get("decision_id", u),
                "reversing_action": u_data.get("action", ""),
                "reversing_entity": u_data.get("entity", ""),
                "reversing_timestamp": u_data.get("timestamp", ""),
                "reversed_decision": v_data.get("decision_id", v),
                "reversed_action": v_data.get("action", ""),
                "reversal_type": data.get("reversal_type", ""),
                "reason": data.get("reason", ""),
            })

    reversals.sort(key=lambda x: x["reversing_timestamp"])

    summary = (
        f"Found {len(reversals)} explicit decision reversal(s)."
        if reversals else "No reversals found."
    )

    return {
        "query": "All decision reversals",
        "count": len(reversals),
        "reversals": reversals,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 6. pattern_query — dispatch by keyword intent
# ---------------------------------------------------------------------------

DISPATCH_MAP = {
    # decision type queries
    "buy":    ("factors_for_decision_type", "Buy"),
    "sell":   ("factors_for_decision_type", "Sell"),
    "hold":   ("factors_for_decision_type", "Hold"),
    "pass":   ("factors_for_decision_type", "Pass"),
    # structural queries
    "reversal":   ("reversed_decisions", None),
    "reversed":   ("reversed_decisions", None),
    "cross":      ("cross_entity_links", None),
    "connection": ("cross_entity_links", None),
    "link":       ("cross_entity_links", None),
}


def pattern_query(
    G: nx.DiGraph,
    query: str,
    decision_id: str | None = None,
) -> dict[str, Any]:
    """
    Lightweight dispatcher. Routes natural-language queries to the right function.

    Examples:
        pattern_query(G, "what factors drove sell decisions")
        pattern_query(G, "management credibility")   → decisions_by_factor
        pattern_query(G, "lineage D22")              → lineage
        pattern_query(G, "cross entity links")       → cross_entity_links
        pattern_query(G, "all reversals")            → reversed_decisions
    """
    q_lower = query.lower()

    # Explicit lineage request
    if "lineage" in q_lower and decision_id:
        return lineage(G, decision_id)

    # Decision ID embedded in query (e.g. "lineage D22" or "tell me about D22")
    import re
    id_match = re.search(r"\bD(\d+)\b", query, re.IGNORECASE)
    if id_match:
        return lineage(G, f"D{id_match.group(1)}")

    # Keyword dispatch
    for keyword, (fn_name, arg) in DISPATCH_MAP.items():
        if keyword in q_lower:
            if fn_name == "factors_for_decision_type":
                return factors_for_decision_type(G, arg)
            elif fn_name == "reversed_decisions":
                return reversed_decisions(G)
            elif fn_name == "cross_entity_links":
                return cross_entity_links(G)

    # Default: treat the query as a factor name search
    return decisions_by_factor(G, query)