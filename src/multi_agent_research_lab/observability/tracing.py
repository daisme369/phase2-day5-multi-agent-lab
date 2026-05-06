"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context with structured logging.

    Logs span start/end and duration. Optionally integrates with LangSmith
    when LANGSMITH_API_KEY is configured.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    logger.debug("SPAN START: %s | attrs=%s", name, attributes or {})

    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.debug(
            "SPAN END: %s | duration=%.3fs | attrs=%s",
            name,
            span["duration_seconds"],
            span["attributes"],
        )


def log_span(span: dict[str, Any]) -> None:
    """Log a completed span for debugging."""

    logger.info(
        "Trace span: name=%s duration=%.3fs attrs=%s",
        span.get("name", "unknown"),
        span.get("duration_seconds", 0),
        span.get("attributes", {}),
    )


def export_trace_json(trace: list[dict[str, Any]], output_path: Path | None = None) -> str:
    """Export a trace as a JSON string, optionally writing to a file."""

    trace_json = json.dumps(trace, indent=2, default=str)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(trace_json, encoding="utf-8")
        logger.info("Trace exported to: %s", output_path)

    return trace_json
