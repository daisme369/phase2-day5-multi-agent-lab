"""Tests for agent implementations.

Verifies that agents run successfully with mock LLM/search (no API keys needed).
"""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _make_state(query: str = "Explain multi-agent systems") -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query))


def test_supervisor_routes_to_researcher_first() -> None:
    """Supervisor should route to researcher when no research notes exist."""
    state = _make_state()
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "researcher"
    assert result.iteration == 1


def test_supervisor_routes_to_analyst_after_research() -> None:
    """Supervisor should route to analyst after research notes are populated."""
    state = _make_state()
    state.research_notes = "Some research notes."
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "analyst"


def test_supervisor_routes_to_writer_after_analysis() -> None:
    """Supervisor should route to writer after analysis notes are populated."""
    state = _make_state()
    state.research_notes = "Some research notes."
    state.analysis_notes = "Some analysis notes."
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "writer"


def test_supervisor_routes_done_when_complete() -> None:
    """Supervisor should route to done when all fields are populated."""
    state = _make_state()
    state.research_notes = "Notes"
    state.analysis_notes = "Analysis"
    state.final_answer = "Answer"
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"


def test_supervisor_enforces_max_iterations() -> None:
    """Supervisor should force 'done' when max iterations reached."""
    state = _make_state()
    state.iteration = 100  # exceed any reasonable max
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"


def test_researcher_populates_sources_and_notes() -> None:
    """Researcher should populate sources and research_notes."""
    state = _make_state()
    result = ResearcherAgent().run(state)
    assert len(result.sources) > 0
    assert result.research_notes is not None
    assert len(result.research_notes) > 0


def test_analyst_populates_analysis_notes() -> None:
    """Analyst should populate analysis_notes from research_notes."""
    state = _make_state()
    state.research_notes = "Multi-agent systems distribute tasks across specialized agents."
    result = AnalystAgent().run(state)
    assert result.analysis_notes is not None
    assert len(result.analysis_notes) > 0


def test_analyst_handles_missing_research() -> None:
    """Analyst should handle missing research notes gracefully."""
    state = _make_state()
    result = AnalystAgent().run(state)
    assert result.analysis_notes is not None


def test_writer_populates_final_answer() -> None:
    """Writer should produce a final answer from notes."""
    state = _make_state()
    state.research_notes = "Research: Multi-agent systems use specialized agents."
    state.analysis_notes = "Analysis: Key claims are well-supported."
    result = WriterAgent().run(state)
    assert result.final_answer is not None
    assert len(result.final_answer) > 0


def test_critic_reviews_final_answer() -> None:
    """Critic should review the final answer."""
    state = _make_state()
    state.final_answer = "Multi-agent systems use specialized agents [1]."
    result = CriticAgent().run(state)
    # Critic adds an AgentResult but doesn't modify final_answer
    critic_results = [r for r in result.agent_results if r.agent == "critic"]
    assert len(critic_results) == 1


def test_critic_handles_no_answer() -> None:
    """Critic should handle missing final answer gracefully."""
    state = _make_state()
    result = CriticAgent().run(state)
    assert "no final answer" in result.errors[0].lower()
