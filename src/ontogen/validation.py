"""Similarity-cache and structural validation helpers for the ontology pipeline."""

import concurrent.futures
import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ontogen.ontology import Ontology


logger = logging.getLogger("ontogen.ontology")


class ValidationMixin:
    """Mixin containing similarity caching and Phase 3 validation logic."""

    def _describe_request_policy(self: "Ontology") -> str:
        """Return request pacing diagnostics when the agent supports them."""
        describe_method = getattr(
            type(self.agent), "describe_request_policy", None)
        if callable(describe_method):
            return self.agent.describe_request_policy()
        return "request_policy=unavailable"

    def _get_similarity_cached(
        self: "Ontology",
        term_a: str,
        term_b: str,
        description_a: Optional[str] = None,
        description_b: Optional[str] = None,
        context_label: str = "similarity",
    ) -> float:
        """Get similarity between two terms, using cache to avoid redundant LLM calls."""
        cache_key = tuple(sorted([term_a, term_b]))

        if cache_key in self.similarity_cache:
            cached_score = self.similarity_cache[cache_key]
            logger.debug(
                "[%s] Cache hit: similarity(%s, %s) = %f",
                context_label,
                term_a,
                term_b,
                cached_score,
            )
            return cached_score

        logger.info(
            "[%s] Cache miss: evaluating similarity(%s, %s)",
            context_label,
            term_a,
            term_b,
        )
        similarity_kwargs = {
            "term_x": term_a,
            "description_x": description_a,
            "term_y": term_b,
            "description_y": description_b,
        }
        try:
            response = self.agent.get_similarity_with_descriptions(
                **similarity_kwargs,
                request_label=context_label,
            )
        except TypeError as exc:
            if "request_label" not in str(exc):
                raise
            response = self.agent.get_similarity_with_descriptions(
                **similarity_kwargs,
            )

        score = response.get("similarity", 0.0)
        if score is None:
            logger.warning(
                "[%s] LLM returned None for similarity(%s, %s); defaulting to 0",
                context_label,
                term_a,
                term_b,
            )
            score = 0.0
        score = float(score)

        self.similarity_cache[cache_key] = score
        logger.info(
            "[%s] Similarity(%s, %s) = %f",
            context_label,
            term_a,
            term_b,
            score,
        )
        return score

    def _precompute_similarities_parallel(
        self: "Ontology",
        pairs: List[Dict[str, str]],
        term_a_key: str = "term_a",
        term_b_key: str = "term_b",
        desc_a_key: str = "description_a",
        desc_b_key: str = "description_b",
        context_label: str = "similarity-precompute",
    ) -> None:
        """Pre-fill the similarity cache for multiple pairs using concurrent threads."""
        batch_start = time.monotonic()
        uncached: Dict[Tuple[str, str], Dict[str, str]] = {}
        for pair in pairs:
            cache_key = tuple(sorted([pair[term_a_key], pair[term_b_key]]))
            if cache_key not in self.similarity_cache and cache_key not in uncached:
                uncached[cache_key] = pair

        if not uncached:
            logger.debug(
                "[%s] All pairs already cached; nothing to compute",
                context_label,
            )
            return

        logger.info(
            "[%s] Starting similarity batch: total_pairs=%d, uncached_pairs=%d, workers=%d, %s",
            context_label,
            len(pairs),
            len(uncached),
            self.max_workers,
            self._describe_request_policy(),
        )

        def _evaluate(pair: Dict[str, str]) -> Tuple[float, float]:
            request_start = time.monotonic()
            score = self._get_similarity_cached(
                term_a=pair[term_a_key],
                term_b=pair[term_b_key],
                description_a=pair.get(desc_a_key),
                description_b=pair.get(desc_b_key),
                context_label=context_label,
            )
            return score, time.monotonic() - request_start

        completed_count = 0
        failure_count = 0
        durations: List[float] = []
        slowest_key: Optional[Tuple[str, str]] = None
        slowest_duration = 0.0

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
        ) as executor:
            future_to_key = {
                executor.submit(_evaluate, pair): key
                for key, pair in uncached.items()
            }
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    _, duration = future.result()
                    completed_count += 1
                    durations.append(duration)
                    if duration > slowest_duration:
                        slowest_duration = duration
                        slowest_key = key
                except Exception as exc:
                    failure_count += 1
                    logger.error(
                        "[%s] Parallel similarity failed for %s after %.2fs "
                        "(workers=%d, %s): %s",
                        context_label,
                        key,
                        time.monotonic() - batch_start,
                        self.max_workers,
                        self._describe_request_policy(),
                        exc,
                    )
                    self.similarity_cache[key] = 0.0

        average_duration = sum(durations) / \
            len(durations) if durations else 0.0
        logger.info(
            "[%s] Similarity batch complete: uncached_pairs=%d, completed=%d, failed=%d, "
            "batch_duration=%.2fs, avg_call_duration=%.2fs, slowest_pair=%s, "
            "slowest_call_duration=%.2fs",
            context_label,
            len(uncached),
            completed_count,
            failure_count,
            time.monotonic() - batch_start,
            average_duration,
            slowest_key,
            slowest_duration,
        )

    def _generate_validation_pairs(self: "Ontology") -> List[Dict[str, str]]:
        """Generate parent-child validation pairs from the seeded DiGraph."""
        pairs = []
        for parent, child in self.ontology_graph.edges():
            parent_node = self.ontology_graph.nodes[parent]
            child_node = self.ontology_graph.nodes[child]
            pairs.append({
                "term_x": parent_node["term"],
                "desc_x": parent_node["description"],
                "term_y": child_node["term"],
                "desc_y": child_node["description"],
                "category": "parent-child",
            })

        logger.info("Generated %d parent-child validation pairs", len(pairs))
        return pairs

    def validate_structure(self: "Ontology") -> Dict[str, int]:
        """Validate and prune ontology structure based on parent-child similarity."""
        edges_pruned = 0
        pairs = self._generate_validation_pairs()
        context_label = "Phase 3 validation"

        parallel_pairs = [
            {
                "term_a": pair["term_x"],
                "term_b": pair["term_y"],
                "description_a": pair["desc_x"],
                "description_b": pair["desc_y"],
            }
            for pair in pairs
        ]
        self._precompute_similarities_parallel(
            parallel_pairs,
            context_label=context_label,
        )

        parent_child_threshold = self.similarity_threshold
        for pair in pairs:
            similarity = self._get_similarity_cached(
                term_a=pair["term_x"],
                description_a=pair["desc_x"],
                term_b=pair["term_y"],
                description_b=pair["desc_y"],
                context_label=context_label,
            )

            if similarity < parent_child_threshold:
                for parent, child in list(self.ontology_graph.edges()):
                    parent_node = self.ontology_graph.nodes[parent]
                    child_node = self.ontology_graph.nodes[child]

                    if (
                        parent_node["term"] == pair["term_x"]
                        and child_node["term"] == pair["term_y"]
                    ):
                        self.ontology_graph.remove_edge(parent, child)
                        edges_pruned += 1
                        parent_level = parent_node.get("level", "unknown")
                        child_level = child_node.get("level", "unknown")
                        logger.warning(
                            "Pruned weak parent-child edge %s (%s) → %s (%s): similarity=%.1f%%",
                            parent,
                            parent_level,
                            child,
                            child_level,
                            similarity,
                        )
                        break

        orphaned_nodes = 0
        for node_id in self.ontology_graph.nodes():
            if self.ontology_graph.degree(node_id) == 0:
                node_attrs = self.ontology_graph.nodes[node_id]
                term = node_attrs.get("term", node_id)
                level = node_attrs.get("level", "unknown")
                orphaned_nodes += 1
                logger.warning(
                    "Orphaned node detected: %s (term=%s, level=%s)",
                    node_id,
                    term,
                    level,
                )

        summary = {
            "edges_pruned": edges_pruned,
            "orphaned_nodes": orphaned_nodes,
        }
        logger.info("Validation summary: %s", summary)
        return summary
