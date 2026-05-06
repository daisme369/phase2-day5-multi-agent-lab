"""Analyst agent — turns research notes into structured insights."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.analysis_notes``.

        Steps:
        1. Read research_notes from state.
        2. Extract key claims and compare viewpoints.
        3. Flag weak evidence and identify gaps.
        """

        with trace_span("analyst_run") as span:
            if not state.research_notes:
                state.analysis_notes = "No research notes available to analyze."
                logger.warning("Analyst: no research notes found")
                return state

            system_prompt = (
                "You are a critical analyst. Your job is to:\n"
                "1. Extract and list the KEY CLAIMS from the research notes.\n"
                "2. Compare different viewpoints or approaches mentioned.\n"
                "3. Flag any weak evidence, unsupported claims, or gaps.\n"
                "4. Provide a structured analysis with clear sections.\n"
                "Be objective and thorough."
            )
            user_prompt = (
                f"Original query: {state.request.query}\n\n"
                f"Research notes to analyze:\n{state.research_notes}\n\n"
                f"Provide a structured analysis with:\n"
                f"- Key Claims (numbered list)\n"
                f"- Comparative Analysis (if applicable)\n"
                f"- Evidence Strength Assessment\n"
                f"- Knowledge Gaps\n"
                f"- Synthesis / Summary"
            )

            response = self._llm.complete(system_prompt, user_prompt)
            state.analysis_notes = response.content

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("analyst_complete", {
                "analysis_length": len(response.content),
            })

            span["attributes"]["analysis_length"] = len(response.content)
            logger.info("Analyst created analysis: %d chars", len(response.content))

        return state
