"""
conftest.py — shared pytest fixtures for all test modules.

Builds a realistic in-memory graph from mock-extracted elements
so tests run without an LLM API key or network.
"""

import sys
import types
from datetime import datetime

import pytest

# ---------------------------------------------------------------------------
# Pydantic mock — lets tests run without installing pydantic
# (real pydantic is available when running with pip install -r requirements.txt)
# ---------------------------------------------------------------------------
try:
    import pydantic  # noqa: F401
except ImportError:
    mock_pydantic = types.ModuleType("pydantic")

    class _BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return vars(self)

    def _Field(description="", default=None, default_factory=None, **kw):
        return default

    mock_pydantic.BaseModel = _BaseModel
    mock_pydantic.Field = _Field
    sys.modules["pydantic"] = mock_pydantic


from extractor import _parse_elements, load_decisions  # noqa: E402
from graph_builder import build_graph  # noqa: E402
from models import NodeType, EdgeType  # noqa: E402


# ---------------------------------------------------------------------------
# Shared raw extractions — mirrors realistic LLM output
# ---------------------------------------------------------------------------

MOCK_EXTRACTIONS = {
    "D1": {
        "f": [("revenue momentum", "fundamental", "positive"), ("management quality", "management", "positive")],
        "e": [("card-network CTO hire", "management_change")],
        "rv": [],
        "xr": [],
    },
    "D2": {
        "f": [("satellite capacity", "macro", "positive"), ("gross margin expansion", "fundamental", "positive")],
        "e": [],
        "rv": [],
        "xr": [("Tessellate Data Systems", "backhaul contract underpins revenue visibility")],
    },
    "D4": {
        "f": [("lithium price", "macro", "positive"), ("activist catalyst", "catalyst", "positive")],
        "e": [("Ironbridge activist entry", "activist")],
        "rv": [],
        "xr": [],
    },
    "D5": {
        "f": [("edge data demand", "macro", "positive"), ("backhaul cost", "fundamental", "positive")],
        "e": [],
        "rv": [],
        "xr": [("Orbitra Networks", "satellite backhaul drives cost advantage")],
    },
    "D6": {
        "f": [("customer concentration", "risk", "negative"), ("airframer delivery delays", "macro", "negative")],
        "e": [],
        "rv": [],
        "xr": [],
    },
    "D13": {
        "f": [("lithium price", "macro", "negative")],
        "e": [],
        "rv": [],
        "xr": [],
    },
    "D15": {
        "f": [("revenue momentum", "fundamental", "negative"), ("management quality", "management", "negative")],
        "e": [("CTO departure", "management_change")],
        "rv": [],
        "xr": [],
    },
    "D17": {
        "f": [("customer concentration", "risk", "positive"), ("valuation pullback", "valuation", "positive")],
        "e": [("second airframer contract", "contract")],
        "rv": [("D6", "reverses", "Customer diversification condition from original Pass has been met.")],
        "xr": [],
    },
    "D20": {
        "f": [("lithium price", "macro", "negative"), ("activist exit", "catalyst", "negative")],
        "e": [("Ironbridge stake sale", "activist")],
        "rv": [],
        "xr": [],
    },
    "D22": {
        "f": [("management credibility", "management", "negative"), ("revenue momentum", "fundamental", "negative")],
        "e": [("revenue restatement", "accounting")],
        "rv": [],
        "xr": [],
    },
    "D28": {
        "f": [("management credibility", "management", "negative"), ("customer concentration", "risk", "negative")],
        "e": [("Tessellate traffic shift", "management_change")],
        "rv": [],
        "xr": [("Tessellate Data Systems", "Tessellate diversifying away from Orbitra")],
    },
    "D31": {
        "f": [("management credibility", "management", "negative"), ("customer concentration", "risk", "negative")],
        "e": [("Grainline in-house fleet", "competitor")],
        "rv": [],
        "xr": [],
    },
}


def _build_elements(records):
    elements = []
    for r in records:
        m = MOCK_EXTRACTIONS.get(
            r.id,
            {"f": [("revenue momentum", "fundamental", "positive")], "e": [], "rv": [], "xr": []},
        )
        elements.append(
            _parse_elements(
                {
                    "decision_id": r.id,
                    "factors": [{"name": f[0], "category": f[1], "sentiment": f[2]} for f in m["f"]],
                    "events": [{"name": e[0], "event_type": e[1]} for e in m["e"]],
                    "reversals": [
                        {"referenced_decision_id": rv[0], "reversal_type": rv[1], "reason": rv[2]}
                        for rv in m["rv"]
                    ],
                    "cross_references": [{"entity": x[0], "reason": x[1]} for x in m["xr"]],
                    "decision_theme": f"Theme for {r.id}",
                },
                r.id,
            )
        )
    return elements


@pytest.fixture(scope="session")
def all_records():
    return load_decisions("decisions.json")


@pytest.fixture(scope="session")
def graph(all_records):
    """Full graph built from all 52 decisions with mock extractions."""
    elements = _build_elements(all_records)
    return build_graph(elements, all_records)


@pytest.fixture(scope="session")
def small_graph():
    """Smaller graph (first 32 decisions) — faster for unit tests."""
    records = load_decisions("decisions.json")[:32]
    elements = _build_elements(records)
    return build_graph(elements, records)
