"""
api.py — FastAPI application for the Decision Graph.

Endpoints:
    POST /build              — extract + build graph from decisions.json
    GET  /graph/summary      — node/edge counts and metadata
    POST /query              — pattern query (all query types)
    GET  /lineage/{id}       — full lineage for a decision
    GET  /factors/{action}   — factors driving Buy/Sell/Hold/Pass
    GET  /cross-links        — cross-entity connections
    GET  /reversals          — all decision reversals
    POST /whatif             — what-if counterfactual
    GET  /stream/build       — SSE: build graph with live progress updates

Run with:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncGenerator, Optional

import anthropic
import networkx as nx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from extractor import derive_entities, extract_all, load_decisions
from graph_builder import build_graph, graph_summary, load_graph, save_graph
from query_engine import (
    cross_entity_links,
    decisions_by_factor,
    factors_for_decision_type,
    lineage,
    pattern_query,
    reversed_decisions,
)
from whatif_engine import find_affected_decisions, whatif

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Decision Graph",
    description="Investment decision knowledge graph — extract, query, and reason about decisions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# State — graph lives in memory, persisted to disk
# ---------------------------------------------------------------------------

GRAPH_PATH = Path("graph.json")
DATA_PATH = Path("decisions.json")

_graph: Optional[nx.DiGraph] = None


def get_graph() -> nx.DiGraph:
    global _graph
    if _graph is not None:
        return _graph
    if GRAPH_PATH.exists():
        _graph = load_graph(str(GRAPH_PATH))
        return _graph
    raise HTTPException(
        status_code=404,
        detail="Graph not built yet. POST /build first.",
    )


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY not set in environment.",
        )
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class BuildRequest(BaseModel):
    data_path: str = "decisions.json"
    model: str = "claude-sonnet-4-20250514"


class BuildResponse(BaseModel):
    status: str
    total_nodes: int
    total_edges: int
    node_type_counts: dict
    edge_type_counts: dict
    entities: list[str]


class QueryRequest(BaseModel):
    query: str
    decision_id: Optional[str] = None


class WhatIfRequest(BaseModel):
    factor_name: str
    hypothetical_change: str
    model: str = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# POST /build — extract + build graph
# ---------------------------------------------------------------------------

@app.post("/build", response_model=BuildResponse)
async def build(req: BuildRequest):
    """
    Load decisions.json, run LLM extraction on each record,
    build the graph, and persist to graph.json.

    This is the slow endpoint (~1-2 min for 52 decisions).
    Use GET /stream/build for live progress.
    """
    global _graph

    if not Path(req.data_path).exists():
        raise HTTPException(status_code=404, detail=f"Data file not found: {req.data_path}")

    client = get_client()
    records = load_decisions(req.data_path)
    all_elements = extract_all(records, client, model=req.model, verbose=False)

    _graph = build_graph(all_elements, records)
    metadata = save_graph(_graph, str(GRAPH_PATH))

    return BuildResponse(
        status="ok",
        total_nodes=metadata.total_nodes,
        total_edges=metadata.total_edges,
        node_type_counts=metadata.node_type_counts,
        edge_type_counts=metadata.edge_type_counts,
        entities=metadata.entities,
    )


# ---------------------------------------------------------------------------
# GET /graph/summary
# ---------------------------------------------------------------------------

@app.get("/graph/summary")
async def get_summary():
    """Return node/edge counts and a human-readable summary."""
    G = get_graph()
    return {"summary": graph_summary(G)}


# ---------------------------------------------------------------------------
# POST /query — dispatch any query type
# ---------------------------------------------------------------------------

@app.post("/query")
async def query(req: QueryRequest):
    """
    Dispatch a natural-language query to the right query function.

    Examples:
        {"query": "what factors drove sell decisions"}
        {"query": "management credibility"}
        {"query": "lineage D22"}
        {"query": "all reversals"}
        {"query": "cross entity links"}
    """
    G = get_graph()
    return pattern_query(G, req.query, decision_id=req.decision_id)


# ---------------------------------------------------------------------------
# GET /lineage/{decision_id}
# ---------------------------------------------------------------------------

@app.get("/lineage/{decision_id}")
async def get_lineage(decision_id: str, max_depth: int = Query(default=3, ge=1, le=5)):
    """Full upstream ancestry and downstream impact for a single decision."""
    G = get_graph()
    result = lineage(G, decision_id.upper(), max_depth=max_depth)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ---------------------------------------------------------------------------
# GET /factors/{action}
# ---------------------------------------------------------------------------

@app.get("/factors/{action}")
async def get_factors(action: str, top_n: int = Query(default=10, ge=1, le=50)):
    """
    Factors that most influenced a decision type.
    action: Buy | Sell | Hold | Pass
    """
    valid = {"buy", "sell", "hold", "pass"}
    if action.lower() not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{action}'. Must be one of: {valid}",
        )
    G = get_graph()
    return factors_for_decision_type(G, action, top_n=top_n)


# ---------------------------------------------------------------------------
# GET /decisions/by-factor
# ---------------------------------------------------------------------------

@app.get("/decisions/by-factor")
async def get_decisions_by_factor(factor: str = Query(..., description="Factor name (partial match)")):
    """All decisions influenced by a given factor."""
    G = get_graph()
    return decisions_by_factor(G, factor)


# ---------------------------------------------------------------------------
# GET /cross-links
# ---------------------------------------------------------------------------

@app.get("/cross-links")
async def get_cross_links():
    """Cross-entity connections — which companies are linked through shared factors or decisions."""
    G = get_graph()
    return cross_entity_links(G)


# ---------------------------------------------------------------------------
# GET /reversals
# ---------------------------------------------------------------------------

@app.get("/reversals")
async def get_reversals():
    """All explicit decision reversals in the graph."""
    G = get_graph()
    return reversed_decisions(G)


# ---------------------------------------------------------------------------
# POST /whatif
# ---------------------------------------------------------------------------

@app.post("/whatif")
async def run_whatif(req: WhatIfRequest):
    """
    Counterfactual: how would decisions change if a factor shifted?

    Example body:
        {
          "factor_name": "lithium price",
          "hypothetical_change": "rises from $15k to $25k per tonne"
        }
    """
    G = get_graph()
    client = get_client()

    # First return the affected decisions (free — no LLM)
    affected = find_affected_decisions(G, req.factor_name)

    if not affected["all_decision_ids"]:
        raise HTTPException(
            status_code=404,
            detail=f"No decisions found influenced by factor '{req.factor_name}'.",
        )

    result = whatif(
        G,
        factor_name=req.factor_name,
        hypothetical_change=req.hypothetical_change,
        client=client,
        model=req.model,
    )

    return {
        "factor_name": result.factor_name,
        "hypothetical_change": result.hypothetical_change,
        "affected_decision_ids": result.affected_decision_ids,
        "matched_factors": affected["matched_factors"],
        "direct_decisions": len(affected["direct_decisions"]),
        "downstream_decisions": len(affected["downstream_decisions"]),
        "impact_assessment": result.impact_assessment,
        "confidence": result.confidence,
    }


# ---------------------------------------------------------------------------
# GET /stream/build — SSE streaming build with live progress
# ---------------------------------------------------------------------------

async def _stream_build_events(
    data_path: str,
    model: str,
) -> AsyncGenerator[str, None]:
    """
    Async generator yielding SSE events during graph build.
    Each event is a JSON-encoded progress update.
    """

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    try:
        # Step 1: load data
        yield sse("progress", {"step": "loading", "message": "Loading decisions.json...", "pct": 0})
        await asyncio.sleep(0)

        if not Path(data_path).exists():
            yield sse("error", {"message": f"File not found: {data_path}"})
            return

        records = load_decisions(data_path)
        yield sse("progress", {
            "step": "loaded",
            "message": f"Loaded {len(records)} decisions.",
            "pct": 5,
            "total": len(records),
        })
        await asyncio.sleep(0)

        # Step 2: extract (one record at a time for streaming)
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        all_ids = [r.id for r in records]
        known_entities = derive_entities(records)
        all_elements = []

        from extractor import extract

        for i, record in enumerate(records):
            yield sse("progress", {
                "step": "extracting",
                "message": f"Extracting {record.id} — {record.entity} {record.decision}",
                "pct": 5 + int(80 * i / len(records)),
                "current": i + 1,
                "total": len(records),
                "decision_id": record.id,
                "entity": record.entity,
                "action": record.decision,
            })
            await asyncio.sleep(0)

            # Run extraction in thread to avoid blocking event loop
            elements = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda r=record: extract(
                    r,
                    client,
                    all_decision_ids=all_ids,
                    known_entities=known_entities,
                    model=model,
                ),
            )
            all_elements.append(elements)

            # Emit what was extracted
            yield sse("extracted", {
                "decision_id": record.id,
                "factors": [f.name for f in elements.factors],
                "events": [e.name for e in elements.events],
                "reversals": [rv.referenced_decision_id for rv in elements.reversals],
                "cross_refs": [x.entity for x in elements.cross_references],
            })
            await asyncio.sleep(0)

        # Step 3: build graph
        yield sse("progress", {"step": "building", "message": "Building graph...", "pct": 87})
        await asyncio.sleep(0)

        global _graph
        _graph = build_graph(all_elements, records)

        # Step 4: save
        yield sse("progress", {"step": "saving", "message": "Saving graph.json...", "pct": 95})
        await asyncio.sleep(0)

        metadata = save_graph(_graph, str(GRAPH_PATH))

        # Step 5: done
        yield sse("done", {
            "step": "done",
            "message": "Graph built successfully.",
            "pct": 100,
            "total_nodes": metadata.total_nodes,
            "total_edges": metadata.total_edges,
            "node_type_counts": metadata.node_type_counts,
            "edge_type_counts": metadata.edge_type_counts,
        })

    except Exception as e:
        yield sse("error", {"message": str(e)})


@app.get("/stream/build")
async def stream_build(
    data_path: str = Query(default="decisions.json"),
    model: str = Query(default="claude-sonnet-4-20250514"),
):
    """
    SSE endpoint: build graph with live progress events.

    Event types:
        progress  — step name, message, pct (0-100)
        extracted — per-decision extraction result (factors, events, reversals)
        done      — final metadata
        error     — error message

    Connect with EventSource in JS:
        const es = new EventSource('/stream/build');
        es.addEventListener('progress', e => console.log(JSON.parse(e.data)));
        es.addEventListener('extracted', e => console.log(JSON.parse(e.data)));
        es.addEventListener('done', e => { console.log(JSON.parse(e.data)); es.close(); });
        es.addEventListener('error', e => es.close());
    """
    return StreamingResponse(
        _stream_build_events(data_path, model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    graph_built = GRAPH_PATH.exists() or _graph is not None
    return {"status": "ok", "graph_built": graph_built}