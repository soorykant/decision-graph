"""
whatif_engine.py — counterfactual reasoning.

"If revenue growth changed from 30% to 10%, which decisions would be affected?"

Steps:
  1. Find all Factor nodes matching the query (partial name match)
  2. Traverse INFLUENCED edges forward to find directly affected Decisions
  3. Traverse REVERSES edges to find downstream decisions
  4. Call the LLM once with the full context to reason about impact
  5. Return structured WhatIfResult

One LLM call. Deterministic graph traversal. Simple.
"""

from __future__ import annotations

import json
from typing import Any

import networkx as nx

from models import EdgeType, NodeType, WhatIfResult


# ---------------------------------------------------------------------------
# Graph traversal — pure, no LLM
# ---------------------------------------------------------------------------

def find_affected_decisions(
    G: nx.DiGraph,
    factor_name: str,
    max_depth: int = 2,
) -> dict[str, Any]:
    """
    Find all decisions directly or transitively affected by a factor.

    Returns a dict with:
        matched_factors   — factor node labels that matched
        direct_decisions  — decisions the factor directly influences
        downstream_decisions — decisions influenced via REVERSES chains
        all_decision_ids  — union of both sets
    """
    factor_name_lower = factor_name.lower()

    # Match factor nodes by partial name
    matched_factor_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == NodeType.FACTOR
        and factor_name_lower in d.get("label", "").lower()
    ]

    if not matched_factor_nodes:
        return {
            "matched_factors": [],
            "direct_decisions": [],
            "downstream_decisions": [],
            "all_decision_ids": [],
        }

    matched_labels = [G.nodes[n].get("label", n) for n in matched_factor_nodes]

    # Direct: decisions this factor influences
    direct_decisions = []
    direct_decision_nodes = set()

    for fnode in matched_factor_nodes:
        for succ in G.successors(fnode):
            succ_data = G.nodes[succ]
            if succ_data.get("type") == NodeType.DECISION:
                direct_decision_nodes.add(succ)
                direct_decisions.append({
                    "decision_id": succ_data.get("decision_id", succ),
                    "entity": succ_data.get("entity", ""),
                    "action": succ_data.get("action", ""),
                    "timestamp": succ_data.get("timestamp", ""),
                    "rationale": succ_data.get("rationale", ""),
                    "outcome": succ_data.get("outcome", ""),
                    "factor_sentiment": G.edges[fnode, succ].get("sentiment", ""),
                })

    # Downstream: decisions connected via REVERSES edges
    downstream_decisions = []
    visited = set(direct_decision_nodes)
    queue = list(direct_decision_nodes)

    while queue:
        current = queue.pop(0)

        # Check successors for REVERSES edges (both directions)
        for neighbor in list(G.successors(current)) + list(G.predecessors(current)):
            if neighbor in visited:
                continue

            neighbor_data = G.nodes[neighbor]
            if neighbor_data.get("type") != NodeType.DECISION:
                continue

            # Check edge in both directions for REVERSES
            edge_data = {}
            if G.has_edge(current, neighbor):
                edge_data = G.edges[current, neighbor]
            elif G.has_edge(neighbor, current):
                edge_data = G.edges[neighbor, current]

            edge_type = edge_data.get("type", "")
            edge_type_val = edge_type.value if hasattr(edge_type, "value") else str(edge_type)

            if edge_type_val == EdgeType.REVERSES.value:
                visited.add(neighbor)
                queue.append(neighbor)
                downstream_decisions.append({
                    "decision_id": neighbor_data.get("decision_id", neighbor),
                    "entity": neighbor_data.get("entity", ""),
                    "action": neighbor_data.get("action", ""),
                    "timestamp": neighbor_data.get("timestamp", ""),
                    "rationale": neighbor_data.get("rationale", ""),
                    "connection": edge_data.get("reversal_type", "reversal chain"),
                })

    all_ids = sorted(set(
        [d["decision_id"] for d in direct_decisions]
        + [d["decision_id"] for d in downstream_decisions]
    ))

    return {
        "matched_factors": matched_labels,
        "direct_decisions": direct_decisions,
        "downstream_decisions": downstream_decisions,
        "all_decision_ids": all_ids,
    }


# ---------------------------------------------------------------------------
# LLM impact assessment
# ---------------------------------------------------------------------------

WHATIF_SYSTEM = """You are a financial analyst assistant. You reason carefully about
how changing a key assumption would affect historical investment decisions.

Be specific and concise. Reference actual decisions by ID when relevant.
Return only valid JSON, nothing else."""

WHATIF_PROMPT = """Factor: "{factor_name}"
Hypothetical change: "{hypothetical_change}"

Affected decisions ({count} total):
{decisions_summary}

For each affected decision, assess:
1. Would the conclusion (Buy/Sell/Hold/Pass) likely change?
2. Why or why not?

Return JSON:
{{
  "impact_assessment": "2-4 sentence overall assessment of how this change would ripple through these decisions",
  "per_decision": [
    {{
      "decision_id": "D##",
      "would_change": true/false,
      "reasoning": "one sentence"
    }}
  ],
  "confidence": "high" | "medium" | "low",
  "summary": "one sentence bottom line"
}}"""


def assess_impact(
    factor_name: str,
    hypothetical_change: str,
    affected: dict[str, Any],
    client,
    model: str = "claude-sonnet-4-20250514",
) -> dict[str, Any]:
    """Call LLM once to reason about impact. Returns parsed JSON dict."""

    all_decisions = affected["direct_decisions"] + affected["downstream_decisions"]

    if not all_decisions:
        return {
            "impact_assessment": f"No decisions in the graph are influenced by '{factor_name}'.",
            "per_decision": [],
            "confidence": "low",
            "summary": "No affected decisions found.",
        }

    # Build a compact decisions summary for the prompt
    lines = []
    for d in all_decisions[:15]:  # cap at 15 to stay within token budget
        lines.append(
            f"  {d['decision_id']} — {d['action']} {d['entity']} "
            f"({d.get('timestamp','')[:10]}): {d['rationale'][:120]}..."
        )

    prompt = WHATIF_PROMPT.format(
        factor_name=factor_name,
        hypothetical_change=hypothetical_change,
        count=len(all_decisions),
        decisions_summary="\n".join(lines),
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=WHATIF_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    import re
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "impact_assessment": raw[:500],
            "per_decision": [],
            "confidence": "low",
            "summary": "Could not parse structured response.",
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def whatif(
    G: nx.DiGraph,
    factor_name: str,
    hypothetical_change: str,
    client,
    model: str = "claude-sonnet-4-20250514",
) -> WhatIfResult:
    """
    Full what-if pipeline:
        1. Find affected decisions via graph traversal
        2. Call LLM for impact assessment
        3. Return WhatIfResult

    Example:
        result = whatif(G, "lithium price", "rises from $15k to $25k per tonne", client)
    """
    # Step 1: graph traversal (free)
    affected = find_affected_decisions(G, factor_name)

    # Step 2: LLM impact assessment (one call)
    llm_result = assess_impact(
        factor_name=factor_name,
        hypothetical_change=hypothetical_change,
        affected=affected,
        client=client,
        model=model,
    )

    return WhatIfResult(
        factor_name=factor_name,
        hypothetical_change=hypothetical_change,
        affected_decision_ids=affected["all_decision_ids"],
        impact_assessment=llm_result.get("impact_assessment", ""),
        confidence=llm_result.get("confidence", "low"),
    )