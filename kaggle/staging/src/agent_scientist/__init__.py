"""agent_scientist — closed-loop multi-agent system for autonomous ML experimentation.

Agents: Hypothesizer, Executor, Analyst, Critic. Wired together by a minimal
state-graph engine (graph.py) with LangGraph-equivalent semantics
(nodes, edges, conditional routing over a shared state dict) but zero
third-party dependencies, so the whole loop runs anywhere Python runs.

The default LLM provider is deterministic and rule-based ($0 cost); an
Ollama-backed provider can be plugged in for free-form reasoning.
"""

from .graph import StateGraph, END
from .providers import LLMProvider, RuleBasedProvider, OllamaProvider
from .backends import ExperimentBackend, LocalBackend, KaggleBackend, RunSpec, RunResult
from .campaign import CampaignRunner

__all__ = [
    "StateGraph",
    "END",
    "LLMProvider",
    "RuleBasedProvider",
    "OllamaProvider",
    "ExperimentBackend",
    "LocalBackend",
    "KaggleBackend",
    "RunSpec",
    "RunResult",
    "CampaignRunner",
]

__version__ = "0.1.0"
