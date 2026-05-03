"""Graph-based resolution and reward feedback for ontology-backed query systems.

This module provides the ResolutionMixin that adds expression resolution
(walking from leaf expression nodes to bound concept nodes), structural
edge injection, and UCB1 reward feedback from external signals such as
successful or failed query executions.
"""

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ontogen.ontology import Ontology


logger = logging.getLogger("ontogen.ontology")


class ResolutionMixin:
    """Mixin for expression resolution, structural edge injection, and reward feedback."""

    # ------------------------------------------------------------------
    # Expression resolution
    # ------------------------------------------------------------------

    def _get_lexical_level_names(self: "Ontology") -> frozenset[str]:
        """Return schema levels that behave as lexical alias mappings."""
        lexical_levels = frozenset(
            level.name for level in self.level_schema if getattr(level, "is_lexical", False)
        )
        if lexical_levels:
            return lexical_levels
        return frozenset(
            level.name for level in self.level_schema if not level.expandable
        )

    def resolve_expression(
        self: "Ontology",
        token: str,
        *,
        match_mode: str = "contains",
        target_level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Resolve a natural-language token to concept nodes via the ontology graph.

        Searches lexical alias nodes whose ``term`` matches *token*, then walks
        up to the nearest expandable ancestor to return the bound concept with
        its full attributes.

        Args:
            token: The user-facing expression to look up (e.g. ``"margin"``).
            match_mode: ``"exact"`` for case-insensitive equality,
                ``"contains"`` for substring match (default).
            target_level: If given, only consider alias nodes at this level.

        Returns:
            A list of dicts, each with keys ``expression_node``,
            ``concept_node``, ``concept_attrs``, and ``path`` (list of nodes
            from expression to concept).
        """
        token_lower = token.strip().lower()
        if not token_lower:
            return []

        leaf_level_names = self._get_lexical_level_names()
        if target_level:
            leaf_level_names = frozenset([target_level])

        matches: List[Dict[str, Any]] = []
        for node_id, attrs in self.ontology_graph.nodes(data=True):
            if attrs.get("level") not in leaf_level_names:
                continue
            node_term = (attrs.get("term") or "").lower()
            if match_mode == "exact" and node_term != token_lower:
                continue
            if match_mode == "contains" and token_lower not in node_term and node_term not in token_lower:
                continue

            # Walk up to ALL possible expandable ancestors (multi-parent).
            # An expression node may be cross-linked under multiple concepts.
            for pred in self.ontology_graph.predecessors(node_id):
                path = [node_id]
                concept_node, concept_attrs = self._walk_to_ancestor(
                    pred, path
                )
                if concept_node is not None:
                    matches.append({
                        "expression_node": node_id,
                        "concept_node": concept_node,
                        "concept_attrs": dict(concept_attrs),
                        "path": list(path),
                    })

        return matches

    def _walk_to_ancestor(
        self: "Ontology",
        start: str,
        path: List[str],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Walk predecessors until an expandable node is found."""
        expandable_levels = frozenset(
            level.name for level in self.level_schema if level.expandable
        )
        current = start
        visited: set = set(path)  # avoid revisiting nodes already in path
        visited.add(start)
        path.append(start)
        # Check if start itself is expandable
        start_attrs = self.ontology_graph.nodes.get(start, {})
        if start_attrs.get("level") in expandable_levels:
            return start, start_attrs
        while current is not None:
            preds = list(self.ontology_graph.predecessors(current))
            if not preds:
                break
            parent = preds[0]
            if parent in visited:
                break
            visited.add(parent)
            path.append(parent)
            parent_attrs = self.ontology_graph.nodes[parent]
            if parent_attrs.get("level") in expandable_levels:
                return parent, parent_attrs
            current = parent
        return None, {}

    # ------------------------------------------------------------------
    # Connected-concept traversal
    # ------------------------------------------------------------------

    def get_connected_concepts(
        self: "Ontology",
        node: str,
        *,
        relation: Optional[str] = None,
        direction: str = "outgoing",
    ) -> List[Dict[str, Any]]:
        """Return concepts connected to *node* by structural edges.

        Args:
            node: Source node identifier.
            relation: If given, only follow edges with this ``relation``
                attribute.
            direction: ``"outgoing"`` (successors), ``"incoming"``
                (predecessors), or ``"both"``.

        Returns:
            List of dicts with ``node``, ``relation``, ``direction``, and
            ``attrs`` for each connected concept.
        """
        if node not in self.ontology_graph:
            return []

        results: List[Dict[str, Any]] = []

        if direction in ("outgoing", "both"):
            for _, target, edge_attrs in self.ontology_graph.out_edges(node, data=True):
                edge_rel = edge_attrs.get("relation")
                if relation and edge_rel != relation:
                    continue
                results.append({
                    "node": target,
                    "relation": edge_rel,
                    "direction": "outgoing",
                    "attrs": dict(self.ontology_graph.nodes[target]),
                })

        if direction in ("incoming", "both"):
            for source, _, edge_attrs in self.ontology_graph.in_edges(node, data=True):
                edge_rel = edge_attrs.get("relation")
                if relation and edge_rel != relation:
                    continue
                results.append({
                    "node": source,
                    "relation": edge_rel,
                    "direction": "incoming",
                    "attrs": dict(self.ontology_graph.nodes[source]),
                })

        return results

    # ------------------------------------------------------------------
    # Structural edge injection
    # ------------------------------------------------------------------

    def add_structural_edges(
        self: "Ontology",
        edges: List[Dict[str, str]],
    ) -> int:
        """Inject pre-validated structural edges into the ontology graph.

        Use this to insert relationships derived from authoritative metadata
        (e.g. Power BI TOM relationship graph) that should not go through
        LLM similarity validation.

        Each edge dict must contain ``source``, ``target``, and ``relation``.

        Args:
            edges: List of edge specifications.

        Returns:
            Number of edges successfully added.
        """
        added = 0
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            relation = edge.get("relation", "")
            if not source or not target or not relation:
                logger.warning("Skipping malformed structural edge: %s", edge)
                continue
            if source not in self.ontology_graph:
                logger.warning(
                    "Source node '%s' not in graph; skipping edge to '%s'",
                    source,
                    target,
                )
                continue
            if target not in self.ontology_graph:
                logger.warning(
                    "Target node '%s' not in graph; skipping edge from '%s'",
                    target,
                    source,
                )
                continue
            self.ontology_graph.add_edge(source, target, relation=relation)
            added += 1
            logger.info(
                "Injected structural edge: '%s' --%s--> '%s'",
                source,
                relation,
                target,
            )
        return added

    # ------------------------------------------------------------------
    # UCB1 reward feedback
    # ------------------------------------------------------------------

    def update_reward(
        self: "Ontology",
        node: str,
        reward: float,
    ) -> None:
        """Update UCB1 statistics for a node from an external signal.

        Call this after a query succeeds or fails to feed back into the
        bandit's exploitation/exploration trade-off.

        Args:
            node: The concept node whose resolution produced the outcome.
            reward: A value in ``[0, 1]`` — ``1.0`` for a successful query,
                ``0.0`` for a failure, fractional for partial success.
        """
        if node not in self.ontology_graph:
            raise ValueError(f"Node '{node}' not found in ontology graph")
        attrs = self.ontology_graph.nodes[node]
        attrs["n_visits"] = attrs.get("n_visits", 0) + 1
        attrs["total_reward"] = attrs.get("total_reward", 0.0) + reward
        logger.info(
            "Updated reward for '%s': n_visits=%d, total_reward=%.2f",
            node,
            attrs["n_visits"],
            attrs["total_reward"],
        )

    def get_ucb1_scores(
        self: "Ontology",
        exploration_constant: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute current UCB1 scores for all visited nodes.

        Args:
            exploration_constant: Override for the instance's
                ``exploration_constant``. Defaults to ``self.exploration_constant``.

        Returns:
            Dict mapping node identifiers to their UCB1 score.
        """
        c = exploration_constant if exploration_constant is not None else self.exploration_constant
        total_N = sum(
            d.get("n_visits", 0) for _, d in self.ontology_graph.nodes(data=True)
        )
        if total_N == 0:
            return {}

        scores: Dict[str, float] = {}
        for node_id, attrs in self.ontology_graph.nodes(data=True):
            n_i = attrs.get("n_visits", 0)
            if n_i == 0:
                continue
            x_bar = attrs.get("total_reward", 0.0) / n_i
            exploration = c * math.sqrt(math.log(total_N) / n_i)
            scores[node_id] = x_bar + exploration
        return scores

    # ------------------------------------------------------------------
    # UCB1 stats export / import
    # ------------------------------------------------------------------

    def export_ucb1_stats(self: "Ontology") -> Dict[str, Dict[str, float]]:
        """Export node-level UCB1 statistics as a portable dict.

        Returns:
            Dict mapping node identifiers to ``{"n_visits": int, "total_reward": float}``.
        """
        stats: Dict[str, Dict[str, float]] = {}
        for node_id, attrs in self.ontology_graph.nodes(data=True):
            n_visits = attrs.get("n_visits", 0)
            total_reward = attrs.get("total_reward", 0.0)
            if n_visits > 0 or total_reward > 0:
                stats[node_id] = {
                    "n_visits": n_visits,
                    "total_reward": total_reward,
                }
        return stats

    def import_ucb1_stats(
        self: "Ontology",
        stats: Dict[str, Dict[str, float]],
    ) -> int:
        """Overlay UCB1 stats onto existing graph nodes.

        Nodes present in *stats* but missing from the graph are silently
        skipped.  New nodes in the graph that are absent from *stats* keep
        their current values.

        Args:
            stats: Dict as returned by :meth:`export_ucb1_stats`.

        Returns:
            Number of nodes whose stats were updated.
        """
        updated = 0
        for node_id, node_stats in stats.items():
            if node_id not in self.ontology_graph:
                logger.debug(
                    "Skipping stats for absent node '%s'",
                    node_id,
                )
                continue
            attrs = self.ontology_graph.nodes[node_id]
            attrs["n_visits"] = node_stats.get("n_visits", 0)
            attrs["total_reward"] = node_stats.get("total_reward", 0.0)
            updated += 1
        logger.info("Imported UCB1 stats for %d nodes", updated)
        return updated
