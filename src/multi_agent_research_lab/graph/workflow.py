"""LangGraph workflow — orchestrates the multi-agent pipeline."""

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in ``agents/``.
    """

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()
        self._critic = CriticAgent()
        self._settings = get_settings()

    def build(self) -> StateGraph:
        """Create a LangGraph graph with supervisor routing.

        Graph structure:
            supervisor → (researcher | analyst | writer | done)
            researcher → supervisor
            analyst → supervisor
            writer → critic → END (or supervisor if needs revision)
        """

        # Define state as a dict for LangGraph compatibility
        graph = StateGraph(dict)

        # Add nodes
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("researcher", self._researcher_node)
        graph.add_node("analyst", self._analyst_node)
        graph.add_node("writer", self._writer_node)
        graph.add_node("critic", self._critic_node)

        # Set entry point
        graph.set_entry_point("supervisor")

        # Conditional routing from supervisor
        graph.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )

        # Worker agents loop back to supervisor
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        # Writer goes to critic before finishing
        graph.add_edge("writer", "critic")
        graph.add_edge("critic", END)

        return graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""

        with trace_span("multi_agent_workflow", {"query": state.request.query}) as span:
            graph = self.build()
            compiled = graph.compile()

            # Convert ResearchState to dict for LangGraph
            initial_state = {"_research_state": state}

            logger.info("Starting multi-agent workflow for: %s", state.request.query[:80])
            result = compiled.invoke(initial_state)

            # Extract updated state
            final_state = result["_research_state"]

            span["attributes"]["iterations"] = final_state.iteration
            span["attributes"]["route_history"] = final_state.route_history
            logger.info(
                "Workflow complete: %d iterations, route=%s",
                final_state.iteration,
                " -> ".join(final_state.route_history),
            )

        return final_state

    # ── Node wrappers ─────────────────────────────────────────────

    def _supervisor_node(self, state: dict[str, Any]) -> dict[str, Any]:
        research_state: ResearchState = state["_research_state"]
        research_state = self._supervisor.run(research_state)
        return {"_research_state": research_state}

    def _researcher_node(self, state: dict[str, Any]) -> dict[str, Any]:
        research_state: ResearchState = state["_research_state"]
        research_state = self._researcher.run(research_state)
        return {"_research_state": research_state}

    def _analyst_node(self, state: dict[str, Any]) -> dict[str, Any]:
        research_state: ResearchState = state["_research_state"]
        research_state = self._analyst.run(research_state)
        return {"_research_state": research_state}

    def _writer_node(self, state: dict[str, Any]) -> dict[str, Any]:
        research_state: ResearchState = state["_research_state"]
        research_state = self._writer.run(research_state)
        return {"_research_state": research_state}

    def _critic_node(self, state: dict[str, Any]) -> dict[str, Any]:
        research_state: ResearchState = state["_research_state"]
        research_state = self._critic.run(research_state)
        return {"_research_state": research_state}

    def _route_from_supervisor(self, state: dict[str, Any]) -> str:
        """Extract the last route decision from supervisor."""
        research_state: ResearchState = state["_research_state"]
        if not research_state.route_history:
            return "done"
        return research_state.route_history[-1]
