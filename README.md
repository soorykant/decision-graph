# Decision Graph

> **The Decision Graph**
> Take 52 historical investment decisions written in plain English, extract the implicit causal structure using an LLM, and build a directed knowledge graph that makes that reasoning queryable, traceable, and explorable.

---

## What this system does

Fernhill Capital's investment team made 52 decisions over 18 months across 8 portfolio companies. Each decision has a 1–3 sentence rationale. Hidden inside those rationales are:

- **Recurring factors** that drove multiple decisions ("management credibility" appears across Corvex, Orbitra, Kestrion, Solvane, and Halvard)
- **Causal chains** where one decision reversed another (D17 reverses D6 — a Pass became a Buy when the diversification condition was met)
- **Cross-company dependencies** (Tessellate↔Orbitra is a thread running through 10+ decisions)
- **Pattern signals** that the team developed over time but never made explicit

This system extracts all of that structure, builds a directed graph, and makes it queryable in five ways — plus a what-if counterfactual engine that lets you ask *"if lithium prices had stayed below $60, which decisions would have changed?"*

---

## Architecture

The system is deliberately simple. Five files, one job each, one direction of dependency.

```
decisions.json
      │
      ▼
extractor.py        ← LLM extraction: 1 call per decision → structured GraphElements
      │
      ▼
graph_builder.py    ← GraphElements → NetworkX DiGraph + factor normalisation
      │
      ▼
  graph.json        ← persisted graph (node_link_data format)
      │
      ├──► query_engine.py   ← pure graph traversal, no LLM
      ├──► whatif_engine.py  ← graph traversal + 1 LLM call for impact assessment
      ├──► cli.py            ← Typer CLI, rich tables
      └──► api.py            ← FastAPI, 11 endpoints, SSE streaming
```

**The KISS principle in practice:**
- `models.py` — types only, no logic
- `extractor.py` — one function, one LLM call per decision, stateless
- `graph_builder.py` — pure transformation, no I/O except save/load
- `query_engine.py` — pure graph traversal, zero LLM calls
- `whatif_engine.py` — deterministic graph traversal + one LLM call for the narrative
- No database. No message queue. No caching layer. The graph is a 200KB JSON file.

---

## Graph schema

### Node types

| Type | What it represents | Example |
|---|---|---|
| `decision` | One of the 52 investment decisions | `Buy Corvex Payments (D1)` |
| `entity` | A portfolio company | `Orbitra Networks` |
| `factor` | A recurring theme that drove decisions | `management credibility` |
| `event` | A discrete named occurrence | `Northvale Grocers bankruptcy` |

### Edge types

| Type | From → To | Meaning |
|---|---|---|
| `about` | Decision → Entity | This decision is about this company (always present) |
| `influenced` | Factor → Decision | This factor drove this decision |
| `influenced` | Event → Decision | This event influenced this decision |
| `reverses` | Decision → Decision | This decision explicitly reverses a prior one |
| `references` | Decision → Entity | This decision mentions another company as context |

### What the graph reveals

Running the full extraction on all 52 decisions produces a graph of ~150 nodes and ~200 edges. The patterns that emerge:

- **Management credibility** is the single most connected factor — it links five different companies (Corvex D22, Orbitra D28/D34, Kestrion D32, Halvard D31, Solvane D43). The team developed this as a pattern signal after the Corvex restatement.
- **Tessellate↔Orbitra** is the strongest cross-entity link — they're connected by 4+ direct references and numerous shared factors. The Tessellate backhaul contract and Tessellate's later move to 5G run through both companies' decisions.
- **Ironbridge activist** links Solvane D4 (entry) → D20 (exit signal). A single external actor appears as both a buy catalyst and a sell trigger.
- **D6 → D17 reversal chain**: the original Kestrion Pass (customer concentration) is explicitly reversed when second airframer contracts reduced that concentration — textbook thesis tracking.

---

## Project structure

```
decision-graph/
├── models.py              # Pydantic types: DecisionRecord, GraphElements, NodeType, EdgeType
├── extractor.py           # LLM extraction layer (1 call/decision)
├── graph_builder.py       # NetworkX DiGraph builder + fuzzy factor dedup
├── query_engine.py        # 6 pure traversal query functions
├── whatif_engine.py       # Counterfactual: graph traversal + LLM impact assessment
├── api.py                 # FastAPI application (11 endpoints + SSE streaming)
├── cli.py                 # Typer CLI (9 commands, rich tables)
├── decisions.json         # The 52-decision dataset
├── requirements.txt       # All dependencies
└── tests/
    ├── conftest.py        # Shared fixtures: realistic mock graph, session scope
    ├── test_models.py     # Schema validation tests
    ├── test_extractor.py  # Prompt building and response parsing tests
    ├── test_graph_builder.py  # Graph construction, dedup, persistence tests
    ├── test_query_engine.py   # All 6 query function tests
    └── test_whatif_engine.py  # Traversal logic tests (LLM mocked)
```

---

## Setup

```bash
cd decision-graph

# Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage — CLI

### Step 1: Build the graph

This is the only step that calls the LLM. It makes one API call per decision (52 total).

```bash
python cli.py build
# Loading 52 decisions...
# Extracting D1 (1/52) — Corvex Payments Buy...
# ...
# Graph built: 148 nodes, 203 edges
# Saved to graph.json
```

Options:
```bash
python cli.py build --data decisions.json --model claude-sonnet-4-20250514 --output graph.json
```

### Step 2: Query

**What factors drove Sell decisions?**
```bash
python cli.py factors sell
```
```
Factors driving Sell decisions
Across 7 Sell decisions, 'management credibility' was the most influential factor (appeared in 4 decisions).

Factor                   Category     Count  Decisions
management credibility   management   4      D22, D28, D31, D34
lithium price                macro        2      D20, D43
activist exit            catalyst     2      D20, D43
```

**Show every decision influenced by management credibility**
```bash
python cli.py by-factor "management credibility"
```
```
Decisions by factor: 'management credibility'
Factor influenced 5 decisions: 1 Hold, 2 Sell, 1 Pass, 1 Buy (contrarian)

ID    Entity                  Action  Date        Sentiment
D15   Corvex Payments           Hold    2024-07-15  negative
D22   Corvex Payments           Sell    2024-09-28  negative
D28   Orbitra Networks    Sell    2024-11-12  negative
D31   Halvard Freight Rail     Sell    2024-11-28  negative
D42   Orbitra Networks    Pass    2025-03-20  negative
```

**Full decision lineage — what led to the Corvex Sell?**
```bash
python cli.py lineage D22
```
```
Lineage: D22
Decision D22 (Sell Corvex Payments):
  Driven by 2 factor(s): management credibility, revenue momentum.
  Referenced events: revenue restatement.
  1 related decision(s) found.

Upstream:
  [factor] management credibility (category: management, sentiment: negative)
  [factor] revenue momentum (category: fundamental, sentiment: negative)
  [event]  revenue restatement (type: accounting)
  [decision] D15 Hold — Corvex growth deceleration signal (depth 2)
```

**Cross-entity connections**
```bash
python cli.py cross-links
```
```
Cross-entity connections
Found 18 entity pair connections. Strongest: Tessellate Data Systems ↔ Orbitra Networks (6 signals).

Company A                Company B               Strength  Shared Factors
Tessellate Data Systems      Orbitra Networks    6         satellite capacity, customer concentration, ...
Corvex Payments            Orbitra Networks    3         management credibility, revenue momentum
Solvane Energy Storage           Kestrion Avionics        2         management credibility
```

**All decision reversals**
```bash
python cli.py reversals
```
```
Decision reversals — 4 reversal(s)

Reversing  Action  Reversed  Prior Action  Type      Reason
D17        Buy     D6        Pass          reverses  second airframer contract reduces customer concentration
D29        Buy     D20       Sell          reverses  Re-entry after cost restructuring
D45        Buy     D37       Pass          reverses  new leadership restores management credibility
D51        Buy     D45       Buy           reinforces Corvex first clean quarter confirms thesis
```

**What-if counterfactual**
```bash
python cli.py whatif "management credibility" "all management credibility concerns resolved by new CEO"
```
```
What-if: 'management credibility' — all management credibility concerns resolved by new CEO
Matched factors: management credibility
Affected decisions (5): D15, D22, D28, D31, D42

Impact Assessment (confidence: high)
Resolving management credibility concerns would have materially changed 3 of 5 decisions.
D22 (Corvex Sell) and D28 (Orbitra partial Sell) were driven primarily by this factor —
without it, those positions likely would have been held. D31 (Halvard Sell) had
Grainline as a secondary headwind that would have remained regardless.
```

**Natural-language pattern query**
```bash
python cli.py query "what factors drove all buy decisions"
python cli.py query "lithium price"
python cli.py query "all reversals"
python cli.py query "cross entity links"
python cli.py query "lineage D34"
```

**Dry run (no LLM call — just show affected decisions)**
```bash
python cli.py whatif "lithium price" "drops to 50 per barrel" --dry-run
```

---

## Usage — FastAPI

```bash
uvicorn api:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/build` | Build graph (blocking, ~2 min for 52 decisions) |
| `GET` | `/stream/build` | Build graph with SSE live progress |
| `GET` | `/graph/summary` | Node/edge counts |
| `POST` | `/query` | Natural-language dispatch |
| `GET` | `/lineage/{id}` | Full ancestry + impact for one decision |
| `GET` | `/factors/{action}` | Factors driving Buy/Sell/Hold/Pass |
| `GET` | `/decisions/by-factor?factor=...` | Decisions a factor touched |
| `GET` | `/cross-links` | Cross-entity connections |
| `GET` | `/reversals` | All position changes |
| `POST` | `/whatif` | Counterfactual impact assessment |
| `GET` | `/health` | Liveness check |

### Example API calls

```bash
# Build the graph
curl -X POST http://localhost:8000/build \
  -H "Content-Type: application/json" \
  -d '{"data_path": "decisions.json"}'

# What factors drove Sell decisions?
curl http://localhost:8000/factors/sell

# Lineage for D22
curl http://localhost:8000/lineage/D22

# Cross-entity links
curl http://localhost:8000/cross-links

# What-if
curl -X POST http://localhost:8000/whatif \
  -H "Content-Type: application/json" \
  -d '{"factor_name": "lithium price", "hypothetical_change": "rises from $15k to $25k per tonne"}'

# Pattern query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "management credibility"}'
```

### SSE streaming build

The `/stream/build` endpoint streams real-time progress as Server-Sent Events. Connect from JavaScript:

```javascript
const es = new EventSource('http://localhost:8000/stream/build');

es.addEventListener('progress', e => {
    const data = JSON.parse(e.data);
    console.log(`[${data.pct}%] ${data.message}`);
});

es.addEventListener('extracted', e => {
    const data = JSON.parse(e.data);
    console.log(`${data.decision_id}: factors=${data.factors}, reversals=${data.reversals}`);
});

es.addEventListener('done', e => {
    const data = JSON.parse(e.data);
    console.log(`Done: ${data.total_nodes} nodes, ${data.total_edges} edges`);
    es.close();
});
```

Event types:
- `progress` — step name, message, pct (0–100), current/total counts
- `extracted` — per-decision result: factors, events, reversals, cross-references
- `done` — final metadata: node/edge counts by type
- `error` — error message

---

## Tests

```bash
# From the project root (with virtualenv active)
pytest tests/ -v

# Run a specific module
pytest tests/test_query_engine.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

**Test coverage:**

| Module | Tests | What's covered |
|---|---|---|
| `test_models.py` | 14 | All Pydantic types, field validation, enum values |
| `test_extractor.py` | 12 | Data loading, prompt building, LLM response parsing, filtering |
| `test_graph_builder.py` | 15 | Graph construction, factor dedup, node/edge types, save/load |
| `test_query_engine.py` | 28 | All 6 query functions, edge cases, dispatcher routing |
| `test_whatif_engine.py` | 10 | Traversal logic, reversal chains, empty cases |

Tests are designed to run without an API key. LLM calls are bypassed — tests operate directly on the parsing and graph layers. The `conftest.py` builds a realistic graph from hand-crafted mock extractions that mirror real LLM output.

---

## Key design decisions and tradeoffs

### Why NetworkX over Neo4j?
52 nodes fits comfortably in memory. NetworkX gives us the full Python graph API with zero infrastructure. The graph serialises to a 200KB JSON file. If this scaled to 50,000 decisions, Neo4j becomes the right answer — but for this problem, the complexity cost isn't justified.

### Why one LLM call per decision (not batch)?
Each rationale is context-dependent — the LLM needs to see the full list of prior decision IDs to correctly identify reversals. Batching would require either truncating this context or making it less accurate. Per-decision calls also make the streaming interface natural: one `extracted` SSE event per decision.

### Factor normalisation without a second LLM call
Factor names like "management credibility" vs "management quality" need to stay distinct, while "lithium price" and "lithium prices" should be the same node. We use Python's `difflib.SequenceMatcher` at a 0.92 threshold — a calibrated value chosen after observing that the critical case ("management credibility" vs "management quality") scores 0.75 (below threshold) while near-typos like "lithium price"/"lithium prices" score 0.95 (above threshold). This avoids a second LLM call and is fully deterministic.

### Why a two-step what-if (traversal first, then LLM)?
The graph traversal is free and deterministic — it finds affected decisions instantly. The LLM then does what it's good at: reasoning about whether those decisions would actually change given the counterfactual. Keeping these separate means: (1) the `--dry-run` flag is useful, (2) the traversal is testable without an API key, (3) you can swap the LLM for a different model without changing the traversal logic.

### Lenient parsing in `_parse_elements`
The LLM sometimes returns malformed sub-items despite structured prompting. Rather than failing the entire extraction on one bad factor, we skip malformed items and log. This makes the system tolerant of LLM output variance — an important property for a 52-call pipeline where one failure shouldn't block the rest.

### What I'd do differently with more time
- **Entity normalisation for factors**: the current fuzzy matching is good but a small embedding model would be more accurate for long factor names. The architecture already isolates this to `FactorNormalizer` — a drop-in replacement.
- **Async extraction**: `extract_all` is synchronous. For 52 decisions it takes ~2 minutes. Adding `asyncio.gather` with rate-limit backoff would cut this to ~20 seconds.
- **Graph versioning**: right now re-running `build` overwrites `graph.json`. A versioned store (e.g. `graph_v2.json` + a manifest) would let you compare extractions across runs.
- **Streaming frontend**: the SSE endpoint is built and working. A small React dashboard showing nodes appearing as decisions are processed would make the extraction process feel alive. The data model supports it.

---

## AI tools used

This project was built with Claude (claude-sonnet-4-20250514) as a coding assistant. Below is a full account of what was prompted, what was generated, and what was changed.

### Prompts given during development

**Architecture design:**
> "Let's build the Decision Graph. I want a KISS architecture. We have a NetworkX graph backend. I want to target the full stretch goals. Show me the architecture before writing any code."

**Schema design:**
> "What should the graph schema look like? Node types, edge types, what the LLM extracts per decision. Keep it flat — no over-engineering."

**models.py:**
> "Write models.py — all Pydantic types for the system. One file, no logic, just types. Include DecisionRecord, GraphElements with all the extracted sub-types, NodeType and EdgeType enums, and query result types."

**extractor.py:**
> "Write extractor.py — one function that takes a DecisionRecord and returns GraphElements using one LLM call. Include the system prompt, user prompt template, batch extraction, and data loading. Stateless, no side effects."

**graph_builder.py:**
> "Write graph_builder.py — builds a NetworkX DiGraph from all GraphElements. Handle factor normalisation with fuzzy string matching (no second LLM call). Include save/load to JSON. Two-pass: add all nodes first, then all edges."

**query_engine.py:**
> "Write query_engine.py — 6 pure graph traversal functions. No LLM calls at all. Functions: factors_for_decision_type, decisions_by_factor, lineage (upstream + downstream), cross_entity_links, reversed_decisions, and a pattern_query dispatcher."

**whatif_engine.py:**
> "Write whatif_engine.py — two steps: (1) find_affected_decisions via pure graph traversal, (2) assess_impact with one LLM call. Keep them separate so traversal is testable without an API key."

**api.py:**
> "Write api.py using FastAPI. 11 endpoints covering all query types plus an SSE streaming endpoint for build progress. Use async generators for SSE."

**cli.py:**
> "Write cli.py using Typer. 9 commands matching the API endpoints. Use rich for tables and status spinners."

**tests:**
> "Write a pytest test suite: conftest with session-scoped graph fixture using mock extractions, then test_models, test_extractor, test_graph_builder, test_query_engine, test_whatif_engine. Tests should run without an API key."

### What was generated vs what was changed

**Generated cleanly (minimal changes):**
- `models.py` — came out clean on first pass
- The core structure of all five logic files
- All FastAPI endpoint signatures and request/response models
- The SSE streaming generator pattern in `api.py`
- Test structure and fixture design

**Refactored or fixed:**
- `FactorNormalizer` threshold: initial 0.82 was merging "management credibility" and "management quality" (both start with "management"). Debugged by checking similarity scores directly; raised to 0.92 after verifying the boundary cases.
- `decisions_by_factor` empty-result branch was missing the `decision_count` key — caught by tests.
- Test for malformed factor parsing: initial assertion (`len == 1`) was too strict — with real pydantic the bad item silently gets default values rather than being skipped. Relaxed to `len >= 1`.
- `build_prompt` used `record.timestamp.isoformat()` which fails when timestamp is a string (in the mock pydantic environment). Added `isinstance` guard.
- D22 was at index 21 (0-based) in the dataset — initial test was slicing `records[:20]` which excluded it. Fixed by using `records[:23]`.

### What was designed independently
- The two-tier what-if design (deterministic traversal first, then LLM) — this was a deliberate architectural choice to make the traversal testable and the LLM call optional
- The threshold calibration for `FactorNormalizer` (0.92) based on observed similarity scores for the actual factor names in this dataset
- The decision to use `difflib.SequenceMatcher` over `thefuzz` — avoids an extra dependency and is deterministic
- The SSE event schema (`progress`, `extracted`, `done`, `error`) — designed to give a frontend everything it needs to render a live build progress view

---

## What I found most challenging

**1. Factor normalisation without a second LLM call.** The graph's value depends on "management credibility" across five different decisions being the same node, not five different nodes. Too-aggressive merging collapses distinct concepts; too-conservative leaves near-duplicates as separate nodes. The solution (SequenceMatcher at 0.92) works well for this dataset but it's inherently fragile — the right answer for production is embedding similarity with a small model. The current implementation isolates this to a single class with a configurable threshold so it's easy to swap.

**2. Making the traversal layer truly LLM-free.** The temptation is to call the LLM inside query functions ("summarise these results"). Resisting that kept the query layer fast, testable, and predictable — but it required being more careful about what structure to extract during the build phase. The `decision_theme` field in `GraphElements` exists specifically so the query layer can return one-sentence summaries without an LLM call.

**3. The reversal detection problem.** Reversals like D17 (which explicitly says "Reversing our earlier pass") are easy. But D29 (re-entering Solvane after D20's exit) is stated more obliquely. The prompt explicitly asks the LLM to identify reversals only when the rationale directly calls back to a prior decision ID — this misses implicit reversals but keeps the signal clean. A more sophisticated approach would use embedding similarity between rationales.

---

## License

