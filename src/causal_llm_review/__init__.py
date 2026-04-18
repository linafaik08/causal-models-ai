"""
causal_llm_review
=================
LLM-based review pipeline for causal graphs produced by causal-learn.

Supports Anthropic, OpenAI, and Gemini via a thin adapter layer.
Dataset-agnostic: all domain knowledge is passed by the caller at runtime.

Usage
-----
    from src.causal_llm_review import AnthropicAdapter, CausalGraphReviewer, decode_adj_matrix

    adapter  = AnthropicAdapter("claude-opus-4-5")
    reviewer = CausalGraphReviewer(adapter)

    edges     = decode_adj_matrix(G_fci.graph, node_names)
    decisions = reviewer.review(edges, node_names, dataset_context="...")
    adj_fixed = reviewer.apply_corrections(G_fci.graph, decisions, node_names)
"""

from .models import EdgeInput, EdgeDecision, EdgeReviewResponse
from .graph import decode_adj_matrix
from .adapters import LLMAdapter, AnthropicAdapter, OpenAIAdapter, GeminiAdapter
from .reviewer import CausalGraphReviewer
from .prompts import SYSTEM_TEMPLATE, USER_TEMPLATE

__all__ = [
    "EdgeInput",
    "EdgeDecision",
    "EdgeReviewResponse",
    "decode_adj_matrix",
    "LLMAdapter",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "GeminiAdapter",
    "CausalGraphReviewer",
    "SYSTEM_TEMPLATE",
    "USER_TEMPLATE",
]
