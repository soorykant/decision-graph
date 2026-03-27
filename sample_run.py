"""
sample_run.py — complete end-to-end run for submission.

What this script does:
  1. Loads decisions.json
  2. Calls the LLM once per decision to extract graph elements (52 calls)
  3. Builds the NetworkX graph
  4. Saves the raw graph to graph.json
  5. Runs all query types against the graph
  6. Saves everything — graph stats, all query results — to example_output.json

Usage:
    export ANTHROPIC_API_KEY=sk-ant-your-key-here
    python sample_run.py

Output files:
    graph.json          — the built graph (raw, reloadable)
    example_output.json — full run output for submission
"""

import json
import os
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Check API key upfront — fail fast with a clear message
# ---------------------------------------------------------------------------
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set.")
    print("Run:  export ANTHROPIC_API_KEY=sk-ant-your-key-here")
    sys.exit(1)

import anthropic

from extractor import extract_all, load_decisions
from graph_builder import build_graph, graph_summary, save_graph
from query_engine import (
    cross_entity_links,
    decisions_by_factor,
    factors_for_decision_type,
    lineage,
    pattern_query,
    reversed_decisions,
)
from whatif_engine import find_affected_decisions

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH   = "decisions.json"
GRAPH_PATH  = "graph.json"
OUTPUT_PATH = "example_output.json"
MODEL       = "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------

print("=" * 60)
print("Decision Graph — Full Sample Run")
print("=" * 60)
print()

print(f"[1/4] Loading {DATA_PATH}...")
records = load_decisions(DATA_PATH)
print(f"      Loaded {len(records)} decisions across "
      f"{len({r.entity for r in records})} entities.\n")

# ---------------------------------------------------------------------------
# Step 2: Extract graph elements (LLM calls)
# ---------------------------------------------------------------------------

print(f"[2/4] Extracting graph elements via {MODEL}...")
print(f"      This makes one API call per decision ({len(records)} total).")
print(f"      Expected time: ~2-3 minutes.\n")

client = anthropic.Anthropic()
t0 = time.time()

all_elements = extract_all(
    records,
    client,
    model=MODEL,
    verbose=True,   # prints each decision as it's extracted
)

elapsed = time.time() - t0
print(f"\n      Extraction complete in {elapsed:.0f}s.\n")

# ---------------------------------------------------------------------------
# Step 3: Build graph
# ---------------------------------------------------------------------------

print("[3/4] Building graph...")
G = build_graph(all_elements, records)
metadata = save_graph(G, GRAPH_PATH)

print(f"      {graph_summary(G)}")
print(f"      Saved to {GRAPH_PATH}.\n")

# ---------------------------------------------------------------------------
# Step 4: Run all queries and save output
# ---------------------------------------------------------------------------

print("[4/4] Running all queries and saving output...")

output = {
    "run_metadata": {
        "generated_at": datetime.now().isoformat(),
        "dataset": DATA_PATH,
        "model": MODEL,
        "extraction_time_seconds": round(elapsed, 1),
        "graph_stats": {
            "total_nodes": metadata.total_nodes,
            "total_edges": metadata.total_edges,
            "node_type_counts": metadata.node_type_counts,
            "edge_type_counts": metadata.edge_type_counts,
            "entities": metadata.entities,
        },
    },

    "graph_summary": graph_summary(G),

    "queries": {

        # --- What factors drove each decision type? ---
        "factors_buy":  factors_for_decision_type(G, "Buy"),
        "factors_sell": factors_for_decision_type(G, "Sell"),
        "factors_hold": factors_for_decision_type(G, "Hold"),
        "factors_pass": factors_for_decision_type(G, "Pass"),

        # --- Every decision a key factor touched ---
        "by_factor_management_credibility": decisions_by_factor(G, "management credibility"),
        "by_factor_lithium_price":              decisions_by_factor(G, "lithium price"),
        "by_factor_revenue_momentum":       decisions_by_factor(G, "revenue momentum"),
        "by_factor_customer_concentration": decisions_by_factor(G, "customer concentration"),

        # --- Full lineage for notable decisions ---
        "lineage_D22": lineage(G, "D22"),  # Corvex Sell — revenue restatement
        "lineage_D34": lineage(G, "D34"),  # Orbitra final exit — inventory fraud
        "lineage_D17": lineage(G, "D17"),  # Kestrion reversal of D6 — second airframer contract
        "lineage_D43": lineage(G, "D43"),  # Second Solvane exit — EPA undisclosed
        "lineage_D45": lineage(G, "D45"),  # Corvex turnaround buy — new CEO

        # --- Structural queries ---
        "cross_entity_links": cross_entity_links(G),
        "all_reversals":      reversed_decisions(G),

        # --- What-if: affected decisions (traversal layer, no LLM) ---
        "whatif_management_credibility": find_affected_decisions(G, "management credibility"),
        "whatif_lithium_price":              find_affected_decisions(G, "lithium price"),
        "whatif_customer_concentration": find_affected_decisions(G, "customer concentration"),
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Run complete.")
print("=" * 60)
print(f"  graph.json          → {metadata.total_nodes} nodes, {metadata.total_edges} edges")
print(f"  example_output.json → {len(output['queries'])} queries captured")
print()

rev = output["queries"]["all_reversals"]
print(f"  Reversals detected  : {rev['count']}")
for r in rev["reversals"]:
    print(f"    {r['reversing_decision']} ({r['reversing_action']:4s}) "
          f"{r['reversal_type']:10s} {r['reversed_decision']} — {r['reason'][:60]}")

mc = output["queries"]["by_factor_management_credibility"]
print(f"\n  Management credibility: {mc['decision_count']} decisions")
print(f"    Breakdown: {mc['action_breakdown']}")

top = output["queries"]["cross_entity_links"]["connections"][0]
print(f"\n  Strongest cross-entity link:")
print(f"    {top['entity_a']} <-> {top['entity_b']} "
      f"(strength {top['connection_strength']})")
if top["shared_factors"]:
    print(f"    Shared factors: {', '.join(top['shared_factors'][:4])}")

print()
print(f"  Output files ready for submission:")
print(f"    {GRAPH_PATH}")
print(f"    {OUTPUT_PATH}")
print()