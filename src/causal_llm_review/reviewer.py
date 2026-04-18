"""Main CausalGraphReviewer class — orchestrates the LLM review pipeline."""

from __future__ import annotations

import numpy as np

from .adapters import LLMAdapter
from .models import EdgeDecision, EdgeInput, EdgeReviewResponse
from .prompts import SYSTEM_TEMPLATE, USER_TEMPLATE


class CausalGraphReviewer:
    """
    Reviews a causal graph edge list with an LLM and applies corrections.

    Parameters
    ----------
    adapter : LLMAdapter
        Provider adapter (AnthropicAdapter, OpenAIAdapter, or GeminiAdapter).
    """

    TOOL_NAME = "submit_edge_review"
    TOOL_DESC = "Submit domain-knowledge review decisions for all causal edges."

    def __init__(self, adapter: LLMAdapter):
        self.adapter = adapter

    def review(
        self,
        edges: list[EdgeInput],
        node_names: list[str],
        dataset_context: str | None = None,
        domain_rules: str | None = None,
    ) -> list[EdgeDecision]:
        """
        Send the edge list to the LLM and return one decision per edge.

        Parameters
        ----------
        edges : list[EdgeInput]
            Output of decode_adj_matrix().
        node_names : list[str]
            Variable names in the same order as the adjacency matrix columns.
        dataset_context : str, optional
            Free-text description of the dataset variables and their domain meaning.
        domain_rules : str, optional
            Optional additional instructions (e.g. immutability constraints).
            If None, the LLM reasons purely from built-in knowledge.
        """
        system = SYSTEM_TEMPLATE.render(
            dataset_context=dataset_context,
            domain_rules=domain_rules,
        )
        user = USER_TEMPLATE.render(n=len(edges), edges=edges)
        schema = EdgeReviewResponse.model_json_schema()
        raw = self.adapter.complete_with_tool(
            system, user, schema, self.TOOL_NAME, self.TOOL_DESC
        )
        return EdgeReviewResponse.model_validate(raw).decisions

    def apply_corrections(
        self,
        adj: np.ndarray,
        decisions: list[EdgeDecision],
        node_names: list[str],
    ) -> np.ndarray:
        """
        Apply LLM decisions to the adjacency matrix.

        Returns a new array (does not mutate the input).

        Encoding used:
            remove  → (0, 0)
            orient --> → (-1, 1)   i.e. node_from → node_to
            orient <-- → (1, -1)   i.e. node_to → node_from
            reverse → swap adj[i,j] and adj[j,i]
            keep    → no change
        """
        result = adj.copy()
        idx = {name: i for i, name in enumerate(node_names)}

        for dec in decisions:
            i = idx[dec.node_from]
            j = idx[dec.node_to]

            if dec.action == "remove":
                result[i, j] = 0
                result[j, i] = 0

            elif dec.action == "reverse":
                result[i, j], result[j, i] = result[j, i], result[i, j]

            elif dec.action == "orient":
                if dec.corrected_mark == "-->":
                    result[i, j], result[j, i] = -1, 1
                elif dec.corrected_mark == "<--":
                    result[i, j], result[j, i] = 1, -1

            # keep → no change

        return result
