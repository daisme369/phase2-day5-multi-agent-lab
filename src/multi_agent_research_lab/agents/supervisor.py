"""Supervisor / router agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update ``state.route_history`` with the next route.

        Routing policy:
        1. If no research_notes → route to researcher
        2. If no analysis_notes → route to analyst
        3. If no final_answer → route to writer
        4. Otherwise → done
        Enforces max_iterations guard and uses LLM for intelligent routing.
        """

        with trace_span("supervisor_route", {"iteration": state.iteration}) as span:
            # Guard: max iterations
            if state.iteration >= self._settings.max_iterations:
                logger.warning(
                    "Max iterations (%d) reached, forcing completion",
                    self._settings.max_iterations,
                )
                next_route = "done"
            else:
                next_route = self._decide_route(state)

            state.record_route(next_route)
            state.add_trace_event("supervisor_decision", {
                "next_route": next_route,
                "iteration": state.iteration,
                "has_research": state.research_notes is not None,
                "has_analysis": state.analysis_notes is not None,
                "has_answer": state.final_answer is not None,
            })
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.SUPERVISOR,
                    content=f"Routed to: {next_route}",
                    metadata={"iteration": state.iteration, "route": next_route},
                )
            )

            span["attributes"]["route"] = next_route
            logger.info("Supervisor decided: %s (iteration %d)", next_route, state.iteration)

        return state

    def _decide_route(self, state: ResearchState) -> str:
        """Determine the next agent to run based on current state."""

        # Rule-based routing with state inspection
        if state.research_notes is None:
            return "researcher"
        if state.analysis_notes is None:
            return "analyst"
        if state.final_answer is None:
            return "writer"
        return "done"
