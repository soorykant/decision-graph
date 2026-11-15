# Decision Graph

Investment teams record every decision with a short written rationale, but the reasoning that links those decisions (the recurring factors, the reversals, the dependencies between companies) is never written down. It stays in people's heads.

This project recovers that hidden structure. It takes 52 historical investment decisions written in plain English, uses an LLM to extract the implicit causal structure, and builds a directed knowledge graph that makes the reasoning queryable, traceable, and explorable. A what-if engine on top of it answers counterfactual questions.

---

## Quickstart — full end-to-end run

One file does everything:

```bash
cd decision-graph

# 1. Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Set API key
export ANTHROPIC_API_KEY=sk-ant-your-key-here

# 3. Run
python sample_run.py

# No API key? Build the same graph from the hand-written extractions instead
python sample_run.py --offline
```

`sample_run.py` does the following in order:

1. Loads `decisions.json` (52 records)
2. Calls `claude-sonnet-5-5` once per decision to extract structured graph elements (~2–3 min, prints live progress). With `--offline` it reads `extractions.json` instead and makes no API calls
3. Builds the NetworkX graph and saves it to `graph.json`
4. Runs 18 queries across all query types
5. Saves everything to `example_output.json`
6. Prints a summary — node/edge counts, reversals found, management credibility breakdown, strongest cross-entity link

> `graph.json` and `example_output.json` are already included, so you can skip `sample_run.py` and query the pre-built graph directly (for example `python cli.py lineage D22`). They were built with `--offline` from `extractions.json`: extractions written by hand in the exact JSON shape the extraction prompt asks the LLM for, and passed through the same parser. They are not the output of a live LLM run. Run `python sample_run.py` with an API key to rebuild them from real extractions.

---

## What this system does

The included dataset is synthetic: Fernhill Capital, a fictional investment team, made 52 decisions over 18 months across 8 fictional companies. All companies, people and figures are invented. Each decision has a 1–3 sentence rationale. Hidden inside those rationales are patterns that were never made explicit:

- **Recurring factors** — "management credibility" drives exits at Corvex, Orbitra, Halvard, Solvane and Pinehollow
- **Causal chains** — D17 reverses D6 (a Pass became a Buy once Kestrion won a second airframer and its customer concentration fell)
- **Cross-company dependencies** — Orbitra↔Tessellate is a thread running through 10 decisions, and Solvane↔Halvard through 6 more
- **Thesis evolution** — after two credibility failures the team adopts a leadership rule at D39 and applies it at D41, D45, D48 and D52, though the rule exists only inside rationale text

This system extracts all of that structure, builds a directed graph, and makes it queryable — plus a what-if engine for counterfactual reasoning.

---

## Architecture

Five files, one job each, one direction of dependency.

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
      ├──► query_engine.py   ← pure graph traversal, zero LLM calls
      ├──► whatif_engine.py  ← traversal + 1 LLM call for impact assessment
      ├──► cli.py            ← Typer CLI, 9 commands, rich tables
      └──► api.py            ← FastAPI, 11 endpoints, SSE streaming
```

- `models.py` — types only, no logic
- `extractor.py` — one LLM call per decision, stateless
- `graph_builder.py` — pure transformation, no I/O except save/load
- `query_engine.py` — pure graph traversal, zero LLM calls
- `whatif_engine.py` — deterministic traversal + one LLM call for the narrative
- No database. No message queue. The graph is a 200KB JSON file.

---

## Project structure

```
decision-graph/
├── models.py              # Pydantic types — zero logic
├── extractor.py           # LLM extraction (1 call/decision)
├── graph_builder.py       # NetworkX graph construction + fuzzy factor dedup
├── query_engine.py        # 6 pure traversal functions
├── whatif_engine.py       # Counterfactual: traversal + LLM impact assessment
├── api.py                 # FastAPI — 11 endpoints + SSE streaming
├── cli.py                 # Typer CLI — 9 commands
├── sample_run.py          # End-to-end run script (builds graph + runs all queries)
├── decisions.json         # The 52-decision synthetic dataset
├── extractions.json       # Hand-written extractions for --offline runs
├── graph.json             # Pre-built graph (from --offline)
├── example_output.json    # Full query output (from --offline)
├── requirements.txt
├── pytest.ini
└── tests/
    ├── conftest.py        # Session-scoped graph fixture (no API key needed)
    ├── test_models.py
    ├── test_extractor.py
    ├── test_graph_builder.py
    ├── test_query_engine.py
    └── test_whatif_engine.py
```

---

## Graph schema

### Node types


| Type       | Represents                             | Example                  |
| ---------- | -------------------------------------- | ------------------------ |
| `decision` | One of the 52 investment decisions     | `Buy Corvex Payments (D1)` |
| `entity`   | A portfolio company                    | `Orbitra Networks`       |
| `factor`   | A recurring theme that drove decisions | `management credibility` |
| `event`    | A discrete named occurrence            | `Northvale Grocers bankruptcy` |


### Edge types


| Type         | From → To               | Meaning                                           |
| ------------ | ----------------------- | ------------------------------------------------- |
| `about`      | Decision → Entity       | This decision is about this company               |
| `influenced` | Factor/Event → Decision | This factor or event drove this decision          |
| `reverses`   | Decision → Decision     | This decision explicitly reverses a prior one     |
| `references` | Decision → Entity       | This decision mentions another company as context |


---

## CLI usage

```bash
# Build the graph (the only step requiring an API key)
python cli.py build

# What factors drove Sell decisions?
python cli.py factors sell

# Every decision influenced by management credibility
python cli.py by-factor "management credibility"

# Full lineage for the Corvex revenue-restatement exit
python cli.py lineage D22

# Which companies are connected and how?
python cli.py cross-links

# All position reversals and their reasons
python cli.py reversals

# What-if: counterfactual impact (LLM-assessed)
python cli.py whatif "lithium price" "rises from 15k to 25k per tonne"

# Natural-language dispatch (routes to the right query automatically)
python cli.py query "factors driving sell decisions"
python cli.py query "cross entity links"
```

---

## API usage

```bash
uvicorn api:app --reload --port 8000
# Interactive docs at http://localhost:8000/docs
```


| Method | Endpoint                          | Description                           |
| ------ | --------------------------------- | ------------------------------------- |
| `POST` | `/build`                          | Build graph (blocking)                |
| `GET`  | `/stream/build`                   | Build with SSE live progress          |
| `GET`  | `/graph/summary`                  | Node/edge counts                      |
| `POST` | `/query`                          | Natural-language dispatch             |
| `GET`  | `/lineage/{id}`                   | Full ancestry + impact for a decision |
| `GET`  | `/factors/{action}`               | Factors driving Buy/Sell/Hold/Pass    |
| `GET`  | `/decisions/by-factor?factor=...` | Decisions a factor touched            |
| `GET`  | `/cross-links`                    | Cross-entity connections              |
| `GET`  | `/reversals`                      | All position reversals                |
| `POST` | `/whatif`                         | Counterfactual impact assessment      |
| `GET`  | `/health`                         | Liveness check                        |


**SSE streaming** — `/stream/build` emits one event per decision as it's extracted:

```javascript
const es = new EventSource('http://localhost:8000/stream/build');
es.addEventListener('progress',  e => console.log(JSON.parse(e.data)));  // pct, message
es.addEventListener('extracted', e => console.log(JSON.parse(e.data)));  // factors, events, reversals
es.addEventListener('done',      e => { console.log(JSON.parse(e.data)); es.close(); });
```

---

## Tests

Run without an API key. `conftest.py` builds a realistic graph from mock extractions so the suite is fast and deterministic.

```bash
pytest tests/ -v
```


| Module                  | Tests | Covers                                                               |
| ----------------------- | ----- | -------------------------------------------------------------------- |
| `test_models.py`        | 15    | Schema validation, enum values                                       |
| `test_extractor.py`     | 19    | Data loading, prompt building, response parsing, cross-ref filtering |
| `test_graph_builder.py` | 24    | Graph construction, factor dedup, node/edge types, save/load         |
| `test_query_engine.py`  | 38    | All 6 query functions, edge cases, dispatcher routing                |
| `test_whatif_engine.py` | 10    | Traversal logic, reversal chains, empty cases                        |


---

## Key design decisions

**NetworkX over Neo4j.** 52 decisions fits in memory. NetworkX gives the full Python graph API with zero infrastructure. The graph serialises to a 200KB JSON file. If this scaled to 50,000 decisions, Neo4j becomes the right answer.

**One LLM call per decision, not batch.** Each rationale needs the full list of prior decision IDs to correctly detect reversals. Batching would truncate that context. Per-decision calls also make SSE streaming natural — one `extracted` event per decision.

**Factor normalisation without a second LLM call.** "Management credibility" and "management quality" must stay distinct; "lithium price" and "lithium prices" must be the same node. `difflib.SequenceMatcher` at a 0.92 threshold handles this deterministically. The critical boundary — "management credibility" vs "management quality" scores 0.75 (below), "lithium price" vs "lithium prices" scores 0.96 (above). Isolated to `FactorNormalizer` so it's a drop-in swap for embeddings.

**Two-step what-if.** Graph traversal finds affected decisions instantly for free. The LLM then reasons about whether those decisions would actually change. Separating them means `--dry-run` is useful, traversal is testable without an API key, and the LLM is swappable.

**Lenient parsing.** The LLM occasionally returns a malformed sub-item despite structured prompting. `_parse_elements` skips invalid items rather than failing a 52-call pipeline on one bad factor.

**What I'd change with more time.** Async extraction (`asyncio.gather`) would cut the 2-minute build to ~20 seconds. Embedding-based factor normalisation would be more accurate than fuzzy string matching. Graph versioning would let you compare extractions across runs instead of overwriting `graph.json`.

---

## AI tools used

Built with Claude (`claude-sonnet-4-20250514`) as a coding assistant throughout.

### Prompts

```
"Let's build the Decision Graph. KISS architecture, NetworkX backend,
full stretch goals. Show me the architecture before writing any code."

"What should the graph schema look like? Node types, edge types, what the LLM
extracts per decision. Keep it flat — no over-engineering."

"Write models.py — all Pydantic types. One file, no logic. DecisionRecord,
GraphElements with all extracted sub-types, NodeType and EdgeType enums,
and query result types."

"Write extractor.py — DecisionRecord → GraphElements via one LLM call.
System prompt, user prompt template, batch extraction, data loading. Stateless."

"Write graph_builder.py — GraphElements → NetworkX DiGraph. Factor normalisation
with fuzzy string matching, no second LLM call. Save/load to JSON.
Two-pass: nodes first, then edges."

"Write query_engine.py — 6 pure graph traversal functions, zero LLM calls:
factors_for_decision_type, decisions_by_factor, lineage, cross_entity_links,
reversed_decisions, pattern_query dispatcher."

"Write whatif_engine.py — two steps: (1) find_affected_decisions via pure graph
traversal, (2) assess_impact with one LLM call. Keep them separate so traversal
is testable without an API key."

"Write api.py using FastAPI — 11 endpoints plus SSE streaming for build progress."

"Write cli.py using Typer — 9 commands matching the API. Rich tables and spinners."

"Write pytest tests: conftest with session-scoped graph fixture using mock
extractions, test_models, test_extractor, test_graph_builder, test_query_engine,
test_whatif_engine. Must run without an API key."

"Write sample_run.py — single self-contained script: load data, extract all 52
decisions, build graph, save graph.json, run all queries, save example_output.json,
print summary."
```

### What was generated vs what was changed

Generated cleanly: `models.py`, core structure of all five logic files, FastAPI endpoint signatures, SSE generator pattern, test structure and fixture design.

Fixed or refactored:

- `FactorNormalizer` threshold raised 0.82 → 0.92 after "management credibility" and "management quality" were incorrectly merged. Debugged by checking similarity scores directly against the actual factor names.
- `decisions_by_factor` empty-result branch was missing the `decision_count` key — caught by tests.
- `build_prompt` used `record.timestamp.isoformat()` which fails when timestamp is a raw string. Added `isinstance` guard.
- D22 is at dataset index 21 — initial test sliced `records[:20]` and excluded it. Fixed to `records[:23]`.
- Malformed factor test assertion `len == 1` was too strict with real pydantic. Relaxed to `len >= 1`.

Designed independently: the two-step what-if architecture, the 0.92 threshold calibrated against actual factor names in this dataset, using `difflib` over `thefuzz` (avoids a dependency, fully deterministic), the SSE event schema (`progress` / `extracted` / `done` / `error`).

---

## What I found most challenging

**Factor normalisation.** The graph's value depends on "management credibility" across five different companies being the same node. Too aggressive and distinct concepts collapse; too conservative and near-duplicates fragment. The 0.92 threshold works for this dataset, but embeddings would be more robust in production. The architecture isolates this to `FactorNormalizer` — a drop-in swap.

**Keeping the query layer LLM-free.** The temptation is to call the LLM inside query functions to summarise results. Resisting that made the layer fast, testable, and deterministic — but required front-loading more structure into the extraction phase. The `decision_theme` field in `GraphElements` exists specifically so queries can return one-sentence summaries without an LLM call.

**Reversal detection.** Explicit reversals ("Reversing our earlier pass") are easy. Implicit ones (D45 re-entering Corvex, which cites the leadership rule but never D22's exit) are stated obliquely. The prompt identifies reversals only when the rationale directly references a prior decision ID — this keeps the signal clean but misses some. Embedding similarity between rationales would catch more.