"""Writer agent — produces the final answer with citations."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.final_answer``.

        Synthesizes research_notes and analysis_notes into a clear,
        well-structured response with citations.
        """

        with trace_span("writer_run") as span:
            # Build source reference list
            source_refs = "\n".join(
                f"[{i + 1}] {s.title} — {s.url}"
                for i, s in enumerate(state.sources)
            ) if state.sources else "No sources available."

            system_prompt = (
                "You are a professional technical writer. Your job is to:\n"
                "1. Synthesize the research notes and analysis into a clear, comprehensive answer.\n"
                "2. Structure the response with an introduction, main sections, and conclusion.\n"
                "3. Include citations using [N] notation referencing the provided sources.\n"
                "4. Write for the specified audience level.\n"
                "5. Aim for approximately 500 words.\n"
                "Be clear, accurate, and engaging."
            )
            user_prompt = (
                f"Original query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research notes:\n{state.research_notes or 'None'}\n\n"
                f"Analysis:\n{state.analysis_notes or 'None'}\n\n"
                f"Source references:\n{source_refs}\n\n"
                f"Write a comprehensive, well-structured final answer."
            )

            response = self._llm.complete(system_prompt, user_prompt)
            state.final_answer = response.content

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "word_count": len(response.content.split()),
                    },
                )
            )
            state.add_trace_event("writer_complete", {
                "answer_length": len(response.content),
                "word_count": len(response.content.split()),
            })

            span["attributes"]["word_count"] = len(response.content.split())
            logger.info("Writer produced answer: %d words", len(response.content.split()))

        return state
