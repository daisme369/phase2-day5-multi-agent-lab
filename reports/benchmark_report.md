# Benchmark Report

_Generated: 2026-05-06 15:35 UTC_

## Results Summary

| Run | Latency (s) | Cost (USD) | Quality (/10) | Notes |
|---|---:|---:|---:|---|
| single-agent-baseline | 96.41 | — | 3.0 | iterations=0, agents=1, sources=0, tokens=51+869 |
| multi-agent-workflow | 277.61 | — | 10.0 | iterations=3, agents=7, sources=5, tokens=3950+2343 |

## Analysis

- **Highest quality**: multi-agent-workflow (10.0/10)
- **Fastest**: single-agent-baseline (96.41s)

### Single-Agent vs Multi-Agent

| Metric | Single-Agent | Multi-Agent | Delta |
|---|---:|---:|---:|
| Latency | 96.41s | 277.61s | 0.35x |
| Quality | 3.0 | 10.0 | +7.0 |

## Failure Modes & Mitigations

| Failure Mode | Mitigation |
|---|---|
| Agent loops indefinitely | `max_iterations` guard in Supervisor |
| LLM timeout or rate limit | `tenacity` retry with exponential backoff |
| Empty research results | Mock search fallback provides baseline data |
| Missing API keys | Graceful fallback to mock LLM/search |
| State corruption | Pydantic validation on all state mutations |

