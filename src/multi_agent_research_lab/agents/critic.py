"""Critic agent — fact-checking and quality review (bonus)."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings.

        Checks:
        1. Citation coverage — are claims backed by sources?
        2. Factual consistency — do claims match source snippets?
        3. Completeness — does the answer address the original query?
        """

        with trace_span("critic_run") as span:
            if not state.final_answer:
                state.errors.append("Critic: no final answer to review")
                logger.warning("Critic: no final answer found")
                return state

            source_info = "\n".join(
                f"[{i + 1}] {s.title}: {s.snippet[:200]}"
                for i, s in enumerate(state.sources)
            ) if state.sources else "No sources available."

            system_prompt = (
                "You are a fact-checker and quality reviewer. Evaluate the final answer by:\n"
                "1. CITATION CHECK: Are key claims backed by cited sources? Rate coverage 0-100%.\n"
                "2. ACCURACY CHECK: Do claims match the provided source snippets?\n"
                "3. COMPLETENESS CHECK: Does the answer fully address the original query?\n"
                "4. QUALITY SCORE: Rate overall quality 1-10.\n\n"
                "Format your response as:\n"
                "Citation Coverage: X%\n"
                "Accuracy Issues: [list or 'None found']\n"
                "Completeness: [assessment]\n"
                "Quality Score: X/10\n"
                "Recommendations: [list]"
            )
            user_prompt = (
                f"Original query: {state.request.query}\n\n"
                f"Final answer to review:\n{state.final_answer}\n\n"
                f"Available sources:\n{source_info}"
            )

            response = self._llm.complete(system_prompt, user_prompt)

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "review_type": "fact_check",
                    },
                )
            )
            state.add_trace_event("critic_complete", {
                "review_length": len(response.content),
            })

            span["attributes"]["review_length"] = len(response.content)
            logger.info("Critic completed review: %d chars", len(response.content))

        return state
