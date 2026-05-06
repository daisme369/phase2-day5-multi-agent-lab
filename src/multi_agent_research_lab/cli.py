"""Command-line entrypoint for the lab."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import trace_span, export_trace_json
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline — one LLM call handles everything."""

    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    with trace_span("baseline_run", {"query": query}) as span:
        llm = LLMClient()
        system_prompt = (
            "You are a research assistant. Given a query, provide a comprehensive, "
            "well-structured answer. Include relevant details, cite sources where possible, "
            "and aim for approximately 500 words."
        )
        response = llm.complete(system_prompt, query)
        state.final_answer = response.content
        span["attributes"]["tokens"] = (response.input_tokens or 0) + (response.output_tokens or 0)

    console.print(Panel.fit(state.final_answer or "No response", title="[bold cyan]Single-Agent Baseline[/]"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow: Supervisor > Researcher > Analyst > Writer > Critic."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)

    # Display route trace
    route_str = " -> ".join(result.route_history) if result.route_history else "N/A"
    console.print(Panel.fit(route_str, title="[bold yellow]Route Trace[/]"))

    # Display final answer
    console.print(Panel.fit(result.final_answer or "No answer produced", title="[bold cyan]Final Answer[/]"))

    # Display agent summary table
    if result.agent_results:
        table = Table(title="Agent Results Summary")
        table.add_column("Agent", style="cyan")
        table.add_column("Content Preview", style="white", max_width=60)
        table.add_column("Tokens", style="green")

        for ar in result.agent_results:
            tokens = ar.metadata.get("input_tokens", "—")
            content_preview = ar.content[:80] + "..." if len(ar.content) > 80 else ar.content
            table.add_row(ar.agent, content_preview, str(tokens))

        console.print(table)


@app.command()
def benchmark(
    query: Annotated[
        str,
        typer.Option("--query", "-q", help="Research query"),
    ] = "Research GraphRAG state-of-the-art and write a 500-word summary",
) -> None:
    """Run benchmark comparing single-agent baseline vs multi-agent workflow."""

    _init()
    console.print("[bold]Running benchmark: single-agent vs multi-agent[/]\n")

    def single_agent_runner(q: str) -> ResearchState:
        request = ResearchQuery(query=q)
        state = ResearchState(request=request)
        llm = LLMClient()
        system_prompt = (
            "You are a research assistant. Given a query, provide a comprehensive, "
            "well-structured answer with citations."
        )
        response = llm.complete(system_prompt, q)
        state.final_answer = response.content
        from multi_agent_research_lab.core.schemas import AgentName, AgentResult

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        return state

    def multi_agent_runner(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        workflow = MultiAgentWorkflow()
        return workflow.run(state)

    # Run both
    console.print("[dim]Running single-agent baseline...[/]")
    _, baseline_metrics = run_benchmark("single-agent-baseline", query, single_agent_runner)

    console.print("[dim]Running multi-agent workflow...[/]")
    multi_state, multi_metrics = run_benchmark("multi-agent-workflow", query, multi_agent_runner)

    # Generate report
    report = render_markdown_report([baseline_metrics, multi_metrics])

    # Save report
    store = LocalArtifactStore()
    report_path = store.write_text("benchmark_report.md", report)

    # Save trace
    if multi_state.trace:
        trace_path = Path("reports") / "trace.json"
        export_trace_json(multi_state.trace, trace_path)

    # Display results
    console.print()
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Single-Agent", style="yellow")
    table.add_column("Multi-Agent", style="green")

    table.add_row("Latency", f"{baseline_metrics.latency_seconds:.2f}s", f"{multi_metrics.latency_seconds:.2f}s")
    table.add_row(
        "Cost",
        f"${baseline_metrics.estimated_cost_usd:.4f}" if baseline_metrics.estimated_cost_usd else "—",
        f"${multi_metrics.estimated_cost_usd:.4f}" if multi_metrics.estimated_cost_usd else "—",
    )
    table.add_row(
        "Quality",
        f"{baseline_metrics.quality_score:.1f}/10" if baseline_metrics.quality_score is not None else "—",
        f"{multi_metrics.quality_score:.1f}/10" if multi_metrics.quality_score is not None else "—",
    )

    console.print(table)
    console.print(f"\n[green]Report saved to:[/] {report_path}")


if __name__ == "__main__":
    app()
