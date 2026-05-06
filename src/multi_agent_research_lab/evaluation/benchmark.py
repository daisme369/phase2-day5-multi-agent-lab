"""Benchmark skeleton for single-agent vs multi-agent."""

import logging
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, estimate cost, and score quality.

    Metrics collected:
    - latency_seconds: wall-clock time
    - estimated_cost_usd: sum of agent cost estimates
    - quality_score: heuristic based on answer completeness and citation coverage
    - notes: summary of the run
    """

    started = perf_counter()
    try:
        state = runner(query)
        latency = perf_counter() - started
        error = False
    except Exception as exc:
        latency = perf_counter() - started
        logger.error("Benchmark run '%s' failed: %s", run_name, exc)
        state = ResearchState(
            request=__import__(
                "multi_agent_research_lab.core.schemas", fromlist=["ResearchQuery"]
            ).ResearchQuery(query=query)
        )
        state.errors.append(str(exc))
        error = True

    # Estimate cost from agent results
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if cost is not None:
            total_cost += cost
        inp = result.metadata.get("input_tokens")
        out = result.metadata.get("output_tokens")
        if inp:
            total_input_tokens += inp
        if out:
            total_output_tokens += out

    # Quality scoring heuristic (0-10)
    quality_score = _score_quality(state, error)

    # Build notes
    notes_parts = []
    notes_parts.append(f"iterations={state.iteration}")
    notes_parts.append(f"agents={len(state.agent_results)}")
    notes_parts.append(f"sources={len(state.sources)}")
    notes_parts.append(f"tokens={total_input_tokens}+{total_output_tokens}")
    if error:
        notes_parts.append(f"ERRORS={len(state.errors)}")

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost if total_cost > 0 else None,
        quality_score=quality_score,
        notes=", ".join(notes_parts),
    )

    logger.info(
        "Benchmark '%s': latency=%.2fs cost=$%s quality=%.1f/10",
        run_name,
        latency,
        f"{total_cost:.6f}" if total_cost else "N/A",
        quality_score or 0,
    )

    return state, metrics


def _score_quality(state: ResearchState, error: bool) -> float:
    """Heuristic quality score (0-10) based on answer completeness."""

    if error:
        return 0.0

    score = 0.0

    # Has final answer? (0-3 points)
    if state.final_answer:
        word_count = len(state.final_answer.split())
        if word_count >= 300:
            score += 3.0
        elif word_count >= 100:
            score += 2.0
        elif word_count > 0:
            score += 1.0

    # Has research notes? (0-2 points)
    if state.research_notes and len(state.research_notes) > 50:
        score += 2.0

    # Has analysis notes? (0-2 points)
    if state.analysis_notes and len(state.analysis_notes) > 50:
        score += 2.0

    # Has sources? (0-2 points)
    if len(state.sources) >= 3:
        score += 2.0
    elif len(state.sources) >= 1:
        score += 1.0

    # Citation coverage bonus (0-1 point)
    if state.final_answer and "[1]" in state.final_answer:
        score += 1.0

    return min(score, 10.0)
