"""Pydantic models and type aliases for edge representation and LLM decisions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EdgeMark   = Literal["-->", "<--", "---", "<->", "o->", "<-o", "o-o", "none"]
Action     = Literal["keep", "remove", "reverse", "orient"]
Confidence = Literal["high", "medium", "low"]


class EdgeInput(BaseModel):
    """One edge extracted from the causal-learn adjacency matrix."""
    node_from: str
    node_to:   str
    mark:      EdgeMark
    idx_from:  int
    idx_to:    int


class EdgeDecision(BaseModel):
    """LLM review decision for a single edge."""
    node_from:      str      = Field(description="Source node name exactly as given in input")
    node_to:        str      = Field(description="Target node name exactly as given in input")
    original_mark:  EdgeMark = Field(description="The mark as received — do not change")
    action:         Action   = Field(description=(
        "keep: edge and direction are correct. "
        "remove: edge is spurious, delete it. "
        "reverse: flip the direction of a directed edge. "
        "orient: commit to a direction for an undirected or uncertain edge."
    ))
    corrected_mark: EdgeMark = Field(description=(
        "The mark after correction. "
        "Use 'none' when action is remove. "
        "For keep, repeat original_mark. "
        "For orient, choose --> or <--."
    ))
    confidence: Confidence = Field(
        description="Confidence in this decision based on domain knowledge."
    )
    reasoning: str = Field(
        description="1-3 sentence explanation grounded in domain knowledge. Be specific."
    )


class EdgeReviewResponse(BaseModel):
    """Top-level tool response: one decision per edge in input order."""
    decisions: list[EdgeDecision]
