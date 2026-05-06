"""Search client abstraction for ResearcherAgent."""

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client with Tavily integration and mock fallback."""

    def __init__(self) -> None:
        settings = get_settings()
        self._tavily_client = None

        if settings.tavily_api_key:
            try:
                from tavily import TavilyClient

                self._tavily_client = TavilyClient(api_key=settings.tavily_api_key)
                logger.info("SearchClient: using Tavily provider")
            except ImportError:
                logger.warning("tavily-python package not installed, falling back to mock search")
        else:
            logger.info("SearchClient: no TAVILY_API_KEY found, using mock search")

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Uses Tavily when API key is configured, otherwise returns mock results.
        """

        if self._tavily_client is not None:
            return self._tavily_search(query, max_results)
        return self._mock_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Search using Tavily API."""

        response = self._tavily_client.search(query=query, max_results=max_results)
        results = []
        for item in response.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", "Untitled"),
                    url=item.get("url"),
                    snippet=item.get("content", "")[:500],
                    metadata={"score": item.get("score", 0.0)},
                )
            )
        logger.info("Tavily search returned %d results for: %s", len(results), query[:80])
        return results

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Return mock search results for testing without API keys."""

        mock_sources = [
            SourceDocument(
                title="Building Effective Agents - Anthropic",
                url="https://www.anthropic.com/engineering/building-effective-agents",
                snippet=(
                    "Multi-agent systems work best when each agent has a clear, well-defined role. "
                    "The supervisor pattern routes tasks to specialized workers, enabling better "
                    "separation of concerns and more focused prompt engineering per subtask."
                ),
                metadata={"score": 0.95, "source": "mock"},
            ),
            SourceDocument(
                title="LangGraph Concepts - LangChain",
                url="https://langchain-ai.github.io/langgraph/concepts/",
                snippet=(
                    "LangGraph provides a framework for building stateful, multi-actor applications "
                    "with LLMs. It uses a graph-based approach where nodes represent computation steps "
                    "and edges define the flow between them, supporting conditional routing."
                ),
                metadata={"score": 0.90, "source": "mock"},
            ),
            SourceDocument(
                title="OpenAI Agents SDK - Orchestration",
                url="https://developers.openai.com/api/docs/guides/agents/orchestration",
                snippet=(
                    "Agent orchestration involves coordinating multiple specialized agents to solve "
                    "complex tasks. Key patterns include supervisor-worker, peer-to-peer, and "
                    "hierarchical architectures with handoff mechanisms."
                ),
                metadata={"score": 0.88, "source": "mock"},
            ),
            SourceDocument(
                title="Multi-Agent Systems: A Survey",
                url="https://arxiv.org/abs/2402.01680",
                snippet=(
                    "Recent advances in LLM-based multi-agent systems show improvements in complex "
                    "reasoning tasks. Decomposing problems into sub-tasks handled by specialized "
                    "agents with distinct prompts yields higher quality outputs than monolithic approaches."
                ),
                metadata={"score": 0.85, "source": "mock"},
            ),
            SourceDocument(
                title="Production Guardrails for LLM Agents",
                url="https://docs.smith.langchain.com/",
                snippet=(
                    "Production-grade agent systems require guardrails: max iteration limits, "
                    "timeout enforcement, input/output validation, retry with exponential backoff, "
                    "and comprehensive tracing for debugging and cost monitoring."
                ),
                metadata={"score": 0.82, "source": "mock"},
            ),
        ]

        results = mock_sources[:max_results]
        logger.info("Mock search returned %d results for: %s", len(results), query[:80])
        return results
