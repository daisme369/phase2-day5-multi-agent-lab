"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
Supports: Ollama (self-hosted), OpenAI, and mock fallback.
"""

import logging
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# Approximate pricing per 1M tokens for gpt-4o-mini (as of 2024-Q3)
_COST_PER_1M_INPUT = 0.15
_COST_PER_1M_OUTPUT = 0.60


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client with Ollama, OpenAI, and mock support."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = None
        self._model = None
        self._provider = "mock"

        provider = settings.llm_provider.lower()

        # Auto-detect: try ollama first (self-hosted), then openai, then mock
        if provider == "auto":
            if self._try_init_ollama(settings):
                pass
            elif self._try_init_openai(settings):
                pass
            else:
                logger.info("LLMClient: no provider available, using mock LLM")
        elif provider == "ollama":
            if not self._try_init_ollama(settings):
                logger.warning("Ollama init failed, falling back to mock LLM")
        elif provider == "openai":
            if not self._try_init_openai(settings):
                logger.warning("OpenAI init failed, falling back to mock LLM")
        else:
            logger.info("LLMClient: provider=%s, using mock LLM", provider)

    def _try_init_ollama(self, settings) -> bool:
        """Try to initialize Ollama client via OpenAI-compatible API."""
        try:
            from openai import OpenAI

            base_url = f"{settings.ollama_base_url}/v1"
            self._client = OpenAI(base_url=base_url, api_key="ollama")
            self._model = settings.ollama_model
            self._provider = "ollama"

            # Quick health check
            self._client.models.list()
            logger.info(
                "LLMClient: using Ollama provider (model=%s, url=%s)",
                self._model,
                base_url,
            )
            return True
        except ImportError:
            logger.warning("openai package not installed")
            return False
        except Exception as exc:
            logger.debug("Ollama not available: %s", exc)
            self._client = None
            return False

    def _try_init_openai(self, settings) -> bool:
        """Try to initialize OpenAI client."""
        if not settings.openai_api_key:
            return False
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.openai_api_key)
            self._model = settings.openai_model
            self._provider = "openai"
            logger.info("LLMClient: using OpenAI provider (model=%s)", self._model)
            return True
        except ImportError:
            logger.warning("openai package not installed")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Uses Ollama/OpenAI when available, otherwise returns a mock response.
        Retry, timeout, and token logging are handled here rather than inside agents.
        """

        if self._client is not None:
            return self._call_api(system_prompt, user_prompt)
        return self._mock_complete(system_prompt, user_prompt)

    def _call_api(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call the OpenAI-compatible Chat Completions API (works with Ollama too)."""

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None

        # Cost estimation: $0 for Ollama (self-hosted), estimated for OpenAI
        cost = None
        if self._provider == "openai" and input_tokens is not None and output_tokens is not None:
            cost = (input_tokens * _COST_PER_1M_INPUT + output_tokens * _COST_PER_1M_OUTPUT) / 1_000_000
        elif self._provider == "ollama":
            cost = 0.0  # Self-hosted = free

        logger.info(
            "LLM call: provider=%s model=%s input_tokens=%s output_tokens=%s cost=$%s",
            self._provider,
            self._model,
            input_tokens,
            output_tokens,
            f"{cost:.6f}" if cost is not None else "N/A",
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    def _mock_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a deterministic mock response for testing without any LLM provider."""

        mock_content = (
            f"[Mock LLM Response]\n\n"
            f"Based on the query: '{user_prompt[:100]}...'\n\n"
            f"This is a simulated response. Key points:\n"
            f"1. Multi-agent systems distribute tasks across specialized agents.\n"
            f"2. A supervisor/router decides which agent handles each subtask.\n"
            f"3. Shared state enables seamless handoff between agents.\n"
            f"4. Guardrails (max iterations, timeout, validation) prevent runaway execution.\n"
            f"5. Benchmarking compares single-agent vs multi-agent on latency, cost, and quality.\n\n"
            f"Sources: [1] Anthropic Building Effective Agents, [2] LangGraph Docs, "
            f"[3] OpenAI Agents SDK Documentation."
        )

        return LLMResponse(
            content=mock_content,
            input_tokens=len(system_prompt.split()) + len(user_prompt.split()),
            output_tokens=len(mock_content.split()),
            cost_usd=0.0001,
        )
