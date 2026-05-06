"""Researcher agent — collects sources and creates research notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.sources`` and ``state.research_notes``.

        Steps:
        1. Search for relevant sources using the query.
        2. Summarize sources into concise research notes via LLM.
        3. Record citations and metadata.
        """

        with trace_span("researcher_run", {"query": state.request.query}) as span:
            # Step 1: Search for sources
            sources = self._search.search(
                query=state.request.query,
                max_results=state.request.max_sources,
            )
            state.sources = sources
            logger.info("Researcher found %d sources", len(sources))

            # Step 2: Summarize sources into research notes
            source_text = "\n\n".join(
                f"[{i + 1}] {s.title}\nURL: {s.url}\nSnippet: {s.snippet}"
                for i, s in enumerate(sources)
            )

            system_prompt = (
                "You are a research assistant. Your job is to synthesize information from "
                "multiple sources into clear, structured research notes. Include citations "
                "using [1], [2], etc. to reference the sources provided."
            )
            user_prompt = (
                f"Research query: {state.request.query}\n\n"
                f"Audience: {state.request.audience}\n\n"
                f"Sources:\n{source_text}\n\n"
                f"Create detailed research notes covering key findings, themes, and "
                f"important details. Cite sources using [N] notation."
            )

            response = self._llm.complete(system_prompt, user_prompt)
            state.research_notes = response.content

            # Record result
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata={
                        "source_count": len(sources),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("researcher_complete", {
                "source_count": len(sources),
                "notes_length": len(response.content),
            })

            span["attributes"]["source_count"] = len(sources)
            logger.info("Researcher created notes: %d chars", len(response.content))

        return state
