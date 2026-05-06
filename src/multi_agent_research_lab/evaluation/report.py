"""Benchmark report rendering."""

from datetime import datetime, timezone

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a comprehensive markdown report.

    Includes:
    - Summary table with all metrics
    - Winner analysis
    - Recommendations
    """

    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Benchmark Report",
        "",
        f"_Generated: {now}_",
        "",
        "## Results Summary",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality (/10) | Notes |",
        "|---|---:|---:|---:|---|",
    ]

    for item in metrics:
        cost = "—" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.4f}"
        quality = "—" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |"
        )

    lines.append("")

    # Winner analysis
    if len(metrics) >= 2:
        lines.append("## Analysis")
        lines.append("")

        best_quality = max(metrics, key=lambda m: m.quality_score or 0)
        fastest = min(metrics, key=lambda m: m.latency_seconds)

        lines.append(f"- **Highest quality**: {best_quality.run_name} ({best_quality.quality_score:.1f}/10)")
        lines.append(f"- **Fastest**: {fastest.run_name} ({fastest.latency_seconds:.2f}s)")
        lines.append("")

        # Compare single vs multi
        single = [m for m in metrics if "single" in m.run_name.lower() or "baseline" in m.run_name.lower()]
        multi = [m for m in metrics if "multi" in m.run_name.lower()]

        if single and multi:
            s, m_item = single[0], multi[0]
            speedup = s.latency_seconds / m_item.latency_seconds if m_item.latency_seconds > 0 else 0
            quality_diff = (m_item.quality_score or 0) - (s.quality_score or 0)

            lines.append("### Single-Agent vs Multi-Agent")
            lines.append("")
            lines.append(f"| Metric | Single-Agent | Multi-Agent | Delta |")
            lines.append(f"|---|---:|---:|---:|")
            lines.append(
                f"| Latency | {s.latency_seconds:.2f}s | {m_item.latency_seconds:.2f}s | "
                f"{speedup:.2f}x |"
            )
            lines.append(
                f"| Quality | {s.quality_score or 0:.1f} | {m_item.quality_score or 0:.1f} | "
                f"{quality_diff:+.1f} |"
            )
            lines.append("")

    # Failure modes
    lines.append("## Failure Modes & Mitigations")
    lines.append("")
    lines.append("| Failure Mode | Mitigation |")
    lines.append("|---|---|")
    lines.append("| Agent loops indefinitely | `max_iterations` guard in Supervisor |")
    lines.append("| LLM timeout or rate limit | `tenacity` retry with exponential backoff |")
    lines.append("| Empty research results | Mock search fallback provides baseline data |")
    lines.append("| Missing API keys | Graceful fallback to mock LLM/search |")
    lines.append("| State corruption | Pydantic validation on all state mutations |")
    lines.append("")

    return "\n".join(lines) + "\n"
