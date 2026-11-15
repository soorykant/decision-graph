"""
cli.py — command-line interface for the Decision Graph.

Commands:
    build           Extract + build graph from decisions.json
    summary         Print graph node/edge counts
    query           Run a pattern query
    lineage         Show full lineage for a decision
    factors         Show factors driving a decision type
    by-factor       Show decisions influenced by a factor
    cross-links     Show cross-entity connections
    reversals       Show all decision reversals
    whatif          Run a what-if counterfactual

Usage:
    python cli.py build
    python cli.py query "management credibility"
    python cli.py lineage D22
    python cli.py factors sell
    python cli.py whatif "lithium price" "rises from 15k to 25k per tonne"
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import anthropic
import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from extractor import extract_all, load_decisions
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

app = typer.Typer(help="Decision Graph CLI", add_completion=False)
console = Console()

GRAPH_PATH = Path("graph.json")
DATA_PATH = Path("decisions.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_or_die() -> object:
    """Load graph from disk or exit with a clear message."""
    if not GRAPH_PATH.exists():
        rprint("[red]Graph not found.[/red] Run [bold]python cli.py build[/bold] first.")
        raise typer.Exit(1)
    return load_graph(str(GRAPH_PATH))


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        rprint("[red]ANTHROPIC_API_KEY not set.[/red] Export it before running.")
        raise typer.Exit(1)
    return anthropic.Anthropic(api_key=api_key)


def print_json(data: dict):
    rprint(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

@app.command()
def build(
    data_path: str = typer.Option("decisions.json", "--data", "-d", help="Path to decisions.json"),
    model: str = typer.Option("claude-sonnet-5-5", "--model", "-m", help="LLM model"),
    output: str = typer.Option("graph.json", "--output", "-o", help="Output graph file"),
):
    """Extract graph elements from all decisions and build the graph."""
    if not Path(data_path).exists():
        rprint(f"[red]File not found:[/red] {data_path}")
        raise typer.Exit(1)

    client = get_client()

    with console.status(f"Loading {data_path}..."):
        records = load_decisions(data_path)
    rprint(f"[green]Loaded[/green] {len(records)} decisions.")

    rprint(f"\nExtracting graph elements using [bold]{model}[/bold]...")
    all_elements = extract_all(records, client, model=model, verbose=True)

    with console.status("Building graph..."):
        G = build_graph(all_elements, records)

    with console.status(f"Saving to {output}..."):
        metadata = save_graph(G, output)

    rprint(f"\n[bold green]Graph built successfully.[/bold green]")
    rprint(f"  Nodes : {metadata.total_nodes}")
    rprint(f"  Edges : {metadata.total_edges}")
    rprint(f"  Types : {metadata.node_type_counts}")
    rprint(f"  Saved : {output}")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

@app.command()
def summary():
    """Print graph summary (node/edge counts)."""
    G = load_or_die()
    rprint(graph_summary(G))


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

@app.command()
def query(
    q: str = typer.Argument(..., help="Natural language query"),
    decision_id: Optional[str] = typer.Option(None, "--id", help="Decision ID for lineage queries"),
    raw: bool = typer.Option(False, "--raw", help="Print raw JSON"),
):
    """
    Run a pattern query against the graph.

    Examples:\n
        python cli.py query "factors driving sell decisions"\n
        python cli.py query "management credibility"\n
        python cli.py query "lineage D22"\n
        python cli.py query "all reversals"\n
        python cli.py query "cross entity links"
    """
    G = load_or_die()
    result = pattern_query(G, q, decision_id=decision_id)

    if raw:
        print_json(result)
        return

    # Pretty print summary
    rprint(f"\n[bold]Query:[/bold] {q}")
    if "summary" in result:
        rprint(f"[green]{result['summary']}[/green]\n")
    print_json(result)


# ---------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------

@app.command(name="lineage")
def lineage_cmd(
    decision_id: str = typer.Argument(..., help="Decision ID, e.g. D22"),
    depth: int = typer.Option(3, "--depth", "-d", help="Max traversal depth"),
):
    """Show full upstream ancestry and downstream impact for a decision."""
    G = load_or_die()
    result = lineage(G, decision_id.upper(), max_depth=depth)

    if "error" in result:
        rprint(f"[red]{result['error']}[/red]")
        raise typer.Exit(1)

    rprint(f"\n[bold]Lineage: {decision_id.upper()}[/bold]")
    rprint(f"[green]{result['summary']}[/green]\n")
    print_json(result)


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------

@app.command()
def factors(
    action: str = typer.Argument(..., help="Decision type: buy | sell | hold | pass"),
    top_n: int = typer.Option(10, "--top", "-n"),
):
    """Show factors most commonly driving a decision type."""
    G = load_or_die()
    result = factors_for_decision_type(G, action, top_n=top_n)

    rprint(f"\n[bold]Factors driving {action.title()} decisions[/bold]")
    rprint(f"[green]{result['summary']}[/green]\n")

    if result["factors"]:
        table = Table("Factor", "Category", "Decision Count", "Decisions")
        for f in result["factors"]:
            table.add_row(
                f["factor"],
                f["category"],
                str(f["decision_count"]),
                ", ".join(f["influenced_decisions"][:5]),
            )
        console.print(table)


# ---------------------------------------------------------------------------
# by-factor
# ---------------------------------------------------------------------------

@app.command(name="by-factor")
def by_factor(
    factor_name: str = typer.Argument(..., help="Factor name (partial match)"),
):
    """Show all decisions influenced by a factor."""
    G = load_or_die()
    result = decisions_by_factor(G, factor_name)

    rprint(f"\n[bold]Decisions by factor: '{factor_name}'[/bold]")
    rprint(f"[green]{result['summary']}[/green]\n")

    if result["decisions"]:
        table = Table("ID", "Entity", "Action", "Date", "Sentiment")
        for d in result["decisions"]:
            table.add_row(
                d["decision_id"],
                d["entity"],
                d["action"],
                d["timestamp"][:10],
                d["factor_sentiment"],
            )
        console.print(table)


# ---------------------------------------------------------------------------
# cross-links
# ---------------------------------------------------------------------------

@app.command(name="cross-links")
def cross_links():
    """Show cross-entity connections in the portfolio."""
    G = load_or_die()
    result = cross_entity_links(G)

    rprint(f"\n[bold]Cross-entity connections[/bold]")
    rprint(f"[green]{result['summary']}[/green]\n")

    if result["connections"]:
        table = Table("Company A", "Company B", "Strength", "Shared Factors")
        for c in result["connections"][:15]:
            table.add_row(
                c["entity_a"],
                c["entity_b"],
                str(c["connection_strength"]),
                ", ".join(c["shared_factors"][:3]),
            )
        console.print(table)


# ---------------------------------------------------------------------------
# reversals
# ---------------------------------------------------------------------------

@app.command()
def reversals():
    """Show all explicit decision reversals."""
    G = load_or_die()
    result = reversed_decisions(G)

    rprint(f"\n[bold]Decision reversals[/bold]")
    rprint(f"[green]{result['summary']}[/green]\n")

    if result["reversals"]:
        table = Table("Reversing", "Action", "Reversed", "Prior Action", "Type", "Reason")
        for r in result["reversals"]:
            table.add_row(
                r["reversing_decision"],
                r["reversing_action"],
                r["reversed_decision"],
                r["reversed_action"],
                r["reversal_type"],
                r["reason"][:60],
            )
        console.print(table)


# ---------------------------------------------------------------------------
# whatif
# ---------------------------------------------------------------------------

@app.command(name="whatif")
def whatif_cmd(
    factor_name: str = typer.Argument(..., help="Factor to change, e.g. 'lithium price'"),
    change: str = typer.Argument(..., help="The hypothetical change, e.g. 'drops from $80 to $50'"),
    model: str = typer.Option("claude-sonnet-5-5", "--model"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show affected decisions without calling LLM"),
):
    """Run a what-if counterfactual analysis."""
    G = load_or_die()

    # Always show affected decisions first (free)
    affected = find_affected_decisions(G, factor_name)

    if not affected["all_decision_ids"]:
        rprint(f"[yellow]No decisions found influenced by '{factor_name}'.[/yellow]")
        raise typer.Exit(0)

    rprint(f"\n[bold]What-if:[/bold] '{factor_name}' — {change}")
    rprint(f"[dim]Matched factors:[/dim] {', '.join(affected['matched_factors'])}")
    rprint(f"[dim]Affected decisions ({len(affected['all_decision_ids'])}):[/dim] {', '.join(affected['all_decision_ids'])}")

    if dry_run:
        rprint("\n[dim](dry run — skipping LLM call)[/dim]")
        return

    client = get_client()

    with console.status("Assessing impact..."):
        result = whatif(G, factor_name, change, client, model=model)

    rprint(f"\n[bold]Impact Assessment[/bold] [dim](confidence: {result.confidence})[/dim]")
    rprint(f"[green]{result.impact_assessment}[/green]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()