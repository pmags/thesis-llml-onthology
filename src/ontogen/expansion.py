"""Expansion-phase implementation for the ontology pipeline.

This module groups the UCB1-guided expansion logic that previously lived
inside ``ontogen.ontology``. The public ``Ontology`` class keeps the same
method surface by inheriting from ``ExpansionMixin``.
"""

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ontogen.expansion_models import ExpansionRecord, PhaseRecord
from ontogen.progress import print_phase

if TYPE_CHECKING:
    from ontogen.ontology import Ontology


logger = logging.getLogger("ontogen.ontology")

_EXPANSION_PLATEAU_DELTA = 0.02
_EXPANSION_PLATEAU_LIMIT = 5
_EXPANSION_STAGNATION_LIMIT = 8


@dataclass
class _ExpansionPhaseState:
    """Mutable state tracked across the Phase 4 expansion loop."""

    prev_node_count: int
    productive_reward_history: List[float] = field(default_factory=list)
    plateau_count: int = 0
    stagnation_count: int = 0
    iteration_count: int = 0
    termination_reason: str = "max_iterations"


@dataclass
class _ExpansionIterationSnapshot:
    """Computed metrics for a single non-terminal expansion iteration."""

    node: str
    generated: int
    accepted: int
    reward: float
    acceptance_rate: float
    cumulative_nodes: int
    cumulative_edges: int


class ExpansionMixin:
    """Mixin holding the ontology expansion and convergence logic."""

    def _register_expansion_mode(self: "Ontology", mode: str) -> None:
        """Track whether the current ontology instance is being expanded manually or automatically.

        Manual expansion is a one-way choice for an Ontology instance. Once a
        user starts selecting nodes explicitly, the instance can no longer
        switch back to automatic UCB1-driven expansion.
        """
        if mode not in {"automatic", "manual"}:
            raise ValueError(f"Unknown expansion mode: {mode}")

        current_mode = getattr(self, "expansion_mode", None)
        if mode == "automatic" and current_mode == "manual":
            raise RuntimeError(
                "Automatic expansion is unavailable after manual expansion has started "
                "on this Ontology instance. Create a new Ontology instance to return "
                "to UCB1-driven expansion."
            )

        if mode == "manual" or current_mode is None:
            self.expansion_mode = mode

    def list_expandable_nodes(
        self: "Ontology",
        include_retired: bool = False,
    ) -> List[Any]:
        """Return graph nodes that can be expanded in the current schema.

        Args:
            include_retired: When True, include retired nodes that are still
                structurally expandable. This is useful for manual UI flows
                where users may intentionally override retirement.
        """
        expandable_level_names = frozenset(
            level.name for level in self.level_schema if level.expandable
        )
        return [
            node for node in self.ontology_graph.nodes()
            if self.ontology_graph.nodes[node].get("level") in expandable_level_names
            and (include_retired or not self.ontology_graph.nodes[node].get("retired", False))
        ]

    def _validate_manual_expansion_target(self: "Ontology", node: str) -> None:
        """Validate that a manually selected node exists and can produce children."""
        if node not in self.ontology_graph:
            raise ValueError(f"Node '{node}' not found in ontology graph")

        node_level = self.ontology_graph.nodes[node].get("level")
        level_def = self._get_level(node_level)
        if not level_def.expandable:
            raise ValueError(
                f"Node '{node}' at level '{node_level}' is not expandable"
            )

    def _compute_expansion_reward(
        self: "Ontology",
        accepted_similarities: List[float],
        candidates_generated: int,
    ) -> float:
        """Compute quality-weighted yield for one expansion step.

        The reward treats rejected candidates as zero contribution by dividing
        the sum of accepted similarities by the full generated candidate count.
        This makes the bandit prefer nodes that produce both good and frequent
        additions, rather than only a few high-similarity accepts.
        """
        if candidates_generated <= 0:
            return 0.0
        return sum(accepted_similarities) / candidates_generated

    def _generate_candidates(
        self: "Ontology",
        node: str,
    ) -> List[Dict[str, str]]:
        """Generate new child terms for a given parent node using LLM prompts."""
        if node not in self.ontology_graph:
            logger.warning(
                "Cannot generate candidates for unknown node: %s",
                node,
            )
            return []

        node_attrs = self.ontology_graph.nodes[node]
        parent_term = node_attrs.get("term", node)
        parent_description = node_attrs.get("description", "")
        parent_level = node_attrs.get("level")

        parent_level_def = None
        parent_idx = None
        for idx, level in enumerate(self.level_schema):
            if level.name == parent_level:
                parent_level_def = level
                parent_idx = idx
                break

        if parent_level_def is None:
            logger.warning(
                "Parent level '%s' not found in level_schema",
                parent_level,
            )
            return []

        if parent_idx is None or parent_idx + 1 >= len(self.level_schema):
            logger.info(
                "Node '%s' is at leaf level; no children can be generated",
                node,
            )
            return []

        child_level_def = self.level_schema[parent_idx + 1]

        existing_children = list(self.ontology_graph.successors(node))
        existing_terms = []
        if existing_children:
            for child_node in existing_children:
                child_attrs = self.ontology_graph.nodes[child_node]
                existing_terms.append(child_attrs.get("term", child_node))

        existing_terms_str = ", ".join(
            f'"{term}"' for term in existing_terms
        ) if existing_terms else "none yet"

        role_description = child_level_def.pluralized_name

        prompt = f"""Given the domain "{self.domain}" and scope "{self.scope_description}", generate {self.candidates_per_iteration} new {role_description} for the following {parent_level}:

            Parent {parent_level}: {parent_term}
            Description: {parent_description}

            Context:
            - Domain: {self.domain}
            - The parent {parent_level} already has these {role_description}: {existing_terms_str}
            - Do NOT repeat existing terms

            Generate {self.candidates_per_iteration} NEW {role_description} for "{parent_term}" that fit the {self.domain} domain.

            Return ONLY a valid JSON array with no additional text. Each element must have "term" and "description" keys:
            [
            {{"term": "...", "description": "..."}},
            {{"term": "...", "description": "..."}},
            ...
            ]
            """

        logger.debug(
            "Generating candidates for node '%s' at level '%s'",
            node,
            parent_level,
        )
        logger.debug("Prompt:\n%s", prompt)

        try:
            response = self.agent.chat(
                instructions=(
                    "You are an ontology engineer specialist. Generate accurate, "
                    "specific taxonomies for the given domain based on the requested hierarchy."
                ),
                input=prompt,
            )
        except Exception as exc:
            logger.error("LLM call failed for node '%s': %s", node, exc)
            return []

        logger.debug("LLM response: %s", response)

        candidates = []
        try:
            response_text = response.strip()

            try:
                candidates_raw = json.loads(response_text)
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    candidates_raw = json.loads(json_match.group(0))
                else:
                    raise json.JSONDecodeError(
                        "No JSON array found in response",
                        response_text,
                        0,
                    )

            if not isinstance(candidates_raw, list):
                logger.error(
                    "LLM response is not a list: %s",
                    type(candidates_raw),
                )
                return []

            for item in candidates_raw:
                if not isinstance(item, dict):
                    logger.warning("Candidate item is not a dict: %s", item)
                    continue

                term = item.get("term", "").strip()
                description = item.get("description", "").strip()

                if not term:
                    logger.warning(
                        "Candidate missing 'term' key or empty value")
                    continue

                candidates.append({
                    "term": term,
                    "description": description,
                })

            logger.info("Parsed %d candidates for node '%s'",
                        len(candidates), node)
            return candidates

        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse LLM response as JSON for node '%s': %s",
                node,
                exc,
            )
            logger.error("Raw response: %s", response)
            return []
        except Exception as exc:
            logger.error(
                "Unexpected error parsing candidates for node '%s': %s",
                node,
                exc,
            )
            return []

    def _validate_candidates(
        self: "Ontology",
        parent_node: str,
        candidates: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Filter candidates by similarity to their parent node."""
        if parent_node not in self.ontology_graph:
            logger.warning("Parent node '%s' not found in graph", parent_node)
            return []

        parent_attrs = self.ontology_graph.nodes[parent_node]
        parent_term = parent_attrs.get("term", parent_node)
        parent_description = parent_attrs.get("description", "")

        accepted = []
        threshold = self.similarity_threshold

        logger.info(
            "Validating %d candidates for parent '%s' (threshold=%.1f%%)",
            len(candidates),
            parent_term,
            threshold,
        )
        context_label = f"Phase 4 candidate validation for {parent_term}"

        parallel_pairs = [
            {
                "term_a": parent_term,
                "term_b": candidate.get("term", ""),
                "description_a": parent_description,
                "description_b": candidate.get("description", ""),
            }
            for candidate in candidates
            if candidate.get("term", "")
        ]
        self._precompute_similarities_parallel(
            parallel_pairs,
            context_label=context_label,
        )

        for candidate in candidates:
            candidate_term = candidate.get("term", "")
            candidate_desc = candidate.get("description", "")

            if not candidate_term:
                logger.warning("Candidate missing term; skipping")
                continue

            similarity = self._get_similarity_cached(
                term_a=parent_term,
                description_a=parent_description,
                term_b=candidate_term,
                description_b=candidate_desc,
                context_label=context_label,
            )

            if similarity >= threshold:
                accepted.append(candidate)
                logger.info(
                    "Accepted candidate '%s' for parent '%s': similarity=%.1f%%",
                    candidate_term,
                    parent_term,
                    similarity,
                )
            else:
                logger.info(
                    "Rejected candidate '%s' for parent '%s': similarity=%.1f%% < %.1f%%",
                    candidate_term,
                    parent_term,
                    similarity,
                    threshold,
                )

        logger.info(
            "Validated candidates: %d/%d accepted",
            len(accepted),
            len(candidates),
        )
        return accepted

    def _add_candidate_to_graph(
        self: "Ontology",
        parent: str,
        candidate: Dict[str, str],
    ) -> None:
        """Add an accepted candidate to the ontology graph as a new node and edge."""
        if parent not in self.ontology_graph:
            logger.warning(
                "Parent node '%s' not found in graph; cannot add candidate",
                parent,
            )
            raise ValueError(f"Parent node '{parent}' not found in graph")

        parent_attrs = self.ontology_graph.nodes[parent]
        parent_level = parent_attrs.get("level")

        if parent_level is None:
            logger.warning("Parent node '%s' has no level attribute", parent)
            raise ValueError(f"Parent node '{parent}' has no level attribute")

        parent_level_idx = None
        for idx, level in enumerate(self.level_schema):
            if level.name == parent_level:
                parent_level_idx = idx
                break

        if parent_level_idx is None:
            logger.warning(
                "Parent level '%s' not found in level_schema",
                parent_level,
            )
            raise ValueError(
                f"Parent level '{parent_level}' not found in level_schema"
            )

        if parent_level_idx + 1 >= len(self.level_schema):
            logger.warning(
                "No child level exists for parent level '%s'",
                parent_level,
            )
            raise ValueError(
                f"No child level exists for parent level '{parent_level}'"
            )

        child_level_def = self.level_schema[parent_level_idx + 1]
        child_level_name = child_level_def.name
        child_relation = child_level_def.relation_to_parent

        candidate_term = candidate.get("term", "").strip()
        candidate_desc = candidate.get("description", "").strip()

        if not candidate_term:
            logger.warning("Candidate term is empty; cannot add to graph")
            raise ValueError("Candidate term is empty")

        if candidate_term in self.ontology_graph:
            logger.warning(
                "Candidate '%s' already exists in graph; skipping duplicate",
                candidate_term,
            )
            return

        self.ontology_graph.add_node(
            candidate_term,
            term=candidate_term,
            description=candidate_desc,
            level=child_level_name,
            n_visits=0,
            total_reward=0.0,
        )
        self.ontology_graph.add_edge(
            parent,
            candidate_term,
            relation=child_relation,
        )

        logger.info(
            "Added node '%s' (level=%s) as child of '%s' with relation=%s",
            candidate_term,
            child_level_name,
            parent,
            child_relation,
        )

    def _check_cross_branch_links_batch(
        self: "Ontology",
        candidates: List[Dict[str, str]],
    ) -> None:
        """Batch cross-branch linking for multiple candidates using parallel pre-computation."""
        if not candidates:
            return

        root_level_name = self.level_schema[0].name
        class_nodes = [
            node_id for node_id in self.ontology_graph.nodes()
            if self.ontology_graph.nodes[node_id].get("level") == root_level_name
        ]

        if len(class_nodes) == 0:
            logger.debug("No root-level nodes found for cross-branch linking")
            return

        class_representatives: Dict[str, Tuple[str, str]] = {}
        for class_node in class_nodes:
            representative = class_node
            children = list(self.ontology_graph.successors(class_node))
            if children:
                representative = children[0]
            rep_attrs = self.ontology_graph.nodes[representative]
            class_representatives[class_node] = (
                rep_attrs.get("term", representative),
                rep_attrs.get("description", ""),
            )

        parallel_pairs: List[Dict[str, str]] = []
        evaluation_plan: List[Tuple[str, str, str]] = []

        for candidate in candidates:
            candidate_term = candidate.get("term", "")
            candidate_desc = candidate.get("description", "")

            if not candidate_term or candidate_term not in self.ontology_graph:
                continue

            ancestor_class = self._find_root_ancestor(
                candidate_term,
                root_level_name,
            )

            for class_node in class_nodes:
                if class_node == ancestor_class:
                    continue

                rep_term, rep_desc = class_representatives[class_node]
                parallel_pairs.append({
                    "term_a": candidate_term,
                    "term_b": rep_term,
                    "description_a": candidate_desc,
                    "description_b": rep_desc,
                })
                evaluation_plan.append((candidate_term, class_node, rep_term))

        self._precompute_similarities_parallel(
            parallel_pairs,
            context_label="Phase 4 cross-branch linking",
        )

        cross_link_threshold = self.cross_link_threshold / 100.0
        for candidate_term, class_node, rep_term in evaluation_plan:
            cache_key = tuple(sorted([candidate_term, rep_term]))
            score = self.similarity_cache.get(cache_key, 0.0) / 100.0

            if score > cross_link_threshold:
                self.ontology_graph.add_edge(
                    candidate_term,
                    class_node,
                    relation="type",
                )
                logger.info(
                    "Added cross-branch link: '%s' → '%s' (similarity=%.3f > threshold=%.3f)",
                    candidate_term,
                    class_node,
                    score,
                    cross_link_threshold,
                )

    def _find_root_ancestor(
        self: "Ontology",
        node: str,
        root_level_name: str,
    ) -> Optional[str]:
        """Walk up the graph from a node to find its root-level ancestor."""
        current = node
        visited: set = set()

        while current is not None and current not in visited:
            visited.add(current)
            current_level = self.ontology_graph.nodes[current].get("level")
            if current_level == root_level_name:
                return current

            predecessors = list(self.ontology_graph.predecessors(current))
            current = predecessors[0] if predecessors else None

        return None

    def _check_cross_branch_links(
        self: "Ontology",
        candidate_term: str,
        candidate_desc: str,
    ) -> None:
        """Check if a single candidate should be typed under additional classes."""
        self._check_cross_branch_links_batch([
            {"term": candidate_term, "description": candidate_desc},
        ])

    def _discover_new_classes(
        self: "Ontology",
        num_classes: int = 2,
    ) -> List[str]:
        """Discover new top-level classes from the domain that are not yet in the ontology."""
        root_level = self.level_schema[0]
        root_level_plural = root_level.pluralized_name
        existing_classes = [
            self.ontology_graph.nodes[node].get("term", node)
            for node in self.ontology_graph.nodes()
            if self.ontology_graph.nodes[node].get("level") == root_level.name
        ]

        existing_str = ", ".join(
            f'"{class_name}"' for class_name in existing_classes
        ) if existing_classes else "none yet"

        prompt = f"""Given the domain "{self.domain}" and the its scope "{self.scope_description}", suggest {num_classes} NEW top-level {root_level_plural} that are NOT already in the ontology.

            Existing {root_level_plural}: {existing_str}

            Requirements:
            - Each new {root_level.name} must be a distinct, broad category within "{self.domain}" and its scope "{self.scope_description}"
            - Do NOT repeat any existing {root_level_plural} listed above
            - Each {root_level.name} should open up a new area of the domain not yet covered

            Return ONLY a valid JSON array with no additional text. Each element must have "term" and "description" keys:
            [
            {{"term": "...", "description": "..."}},
            {{"term": "...", "description": "..."}}
            ]"""

        try:
            raw = self.agent.chat(
                instructions="You are an ontology engineer. Suggest new top-level categories.",
                input=prompt,
            )
        except Exception as exc:
            logger.error("Class discovery LLM call failed: %s", exc)
            return []

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "taxonomy" in parsed:
                parsed = parsed["taxonomy"]
            if not isinstance(parsed, list):
                logger.warning(
                    "Class discovery response is not a list: %s",
                    type(parsed),
                )
                return []
        except json.JSONDecodeError as exc:
            logger.error("Class discovery JSON parse error: %s", exc)
            return []

        added: List[str] = []
        for item in parsed:
            term = item.get("term", "").strip(
            ) if isinstance(item, dict) else ""
            desc = item.get("description", "").strip(
            ) if isinstance(item, dict) else ""

            if not term:
                continue
            if term in self.ontology_graph:
                logger.info("Class discovery: skipping duplicate '%s'", term)
                continue

            self.ontology_graph.add_node(
                term,
                term=term,
                description=desc,
                level=root_level.name,
                n_visits=0,
                total_reward=0.0,
            )
            added.append(term)
            logger.info("Discovered new %s: '%s'", root_level.name, term)

        return added

    def _select_node_ucb1(self: "Ontology") -> Optional[str]:
        """Select the next node to expand using the UCB1 algorithm."""
        expandable_nodes = []
        for node_id in self.ontology_graph.nodes():
            node_attrs = self.ontology_graph.nodes[node_id]
            if node_attrs.get("retired", False):
                continue
            node_level = node_attrs.get("level")
            try:
                level_def = self._get_level(node_level)
                if level_def.expandable:
                    expandable_nodes.append(node_id)
            except ValueError:
                logger.warning(
                    "Node %s has unknown level '%s'",
                    node_id,
                    node_level,
                )
                continue

        if not expandable_nodes:
            logger.info("No expandable nodes remaining in the graph")
            return None

        unvisited = [
            node_id for node_id in expandable_nodes
            if self.ontology_graph.nodes[node_id].get("n_visits", 0) == 0
        ]
        if unvisited:
            selected = unvisited[0]
            logger.info("Selected unvisited expandable node: %s", selected)
            return selected

        total_visits = sum(
            self.ontology_graph.nodes[node_id].get("n_visits", 0)
            for node_id in expandable_nodes
        )
        if total_visits == 0:
            logger.warning(
                "All nodes have zero visits; selecting first expandable node")
            return expandable_nodes[0]

        best_node = None
        best_score = -float("inf")
        for node_id in expandable_nodes:
            n_visits = self.ontology_graph.nodes[node_id].get("n_visits", 0)
            total_reward = self.ontology_graph.nodes[node_id].get(
                "total_reward", 0.0)

            if n_visits == 0:
                continue

            mean_reward = total_reward / n_visits
            ln_n = math.log(total_visits)
            exploration_term = self.exploration_constant * \
                math.sqrt(ln_n / n_visits)
            ucb_score = mean_reward + exploration_term

            logger.debug(
                "Node %s: n_visits=%d, mean_reward=%.3f, exploration_term=%.3f, ucb_score=%.3f",
                node_id,
                n_visits,
                mean_reward,
                exploration_term,
                ucb_score,
            )

            if ucb_score > best_score:
                best_score = ucb_score
                best_node = node_id

        if best_node is None:
            logger.warning(
                "Could not select a node via UCB1 (this shouldn't happen)")
            return None

        logger.info("Selected node via UCB1: %s (score=%.3f)",
                    best_node, best_score)
        return best_node

    def _update_bandit(
        self: "Ontology",
        node: str,
        reward: float,
        candidates_accepted: int = 0,
    ) -> None:
        """Update the bandit reward tracking for a node after expansion."""
        if node not in self.ontology_graph:
            raise ValueError(f"Node '{node}' not found in ontology graph")

        attrs = self.ontology_graph.nodes[node]
        new_visits = attrs.get("n_visits", 0) + 1
        attrs["n_visits"] = new_visits

        new_total_reward = attrs.get("total_reward", 0.0) + reward
        attrs["total_reward"] = new_total_reward

        if candidates_accepted > 0:
            attrs["consecutive_low_yield"] = 0
        else:
            low_yield = attrs.get("consecutive_low_yield", 0) + 1
            attrs["consecutive_low_yield"] = low_yield
            if self.retirement_limit > 0 and low_yield >= self.retirement_limit:
                attrs["retired"] = True
                logger.info(
                    "Node '%s' retired after %d consecutive zero-acceptance visits",
                    node,
                    low_yield,
                )

        logger.info(
            "Updated bandit for node '%s': n_visits=%d, total_reward=%.3f, mean_reward=%.3f",
            node,
            new_visits,
            new_total_reward,
            new_total_reward / new_visits,
        )

    def _empty_expansion_result(self: "Ontology") -> Dict[str, Any]:
        """Return the canonical empty expansion payload."""
        return {
            "node": None,
            "candidates_generated": 0,
            "candidates_accepted": 0,
            "reward": 0.0,
        }

    def _expand_selected_node(
        self: "Ontology",
        selected_node: str,
    ) -> Dict[str, Any]:
        """Run one expansion iteration for a specific already-selected node."""
        logger.info("Expanding node '%s'", selected_node)

        candidates = self._generate_candidates(selected_node)
        candidates_generated = len(candidates)
        logger.info(
            "Generated %d candidates for '%s'",
            candidates_generated,
            selected_node,
        )

        accepted_candidates = self._validate_candidates(
            selected_node, candidates)
        candidates_accepted = len(accepted_candidates)
        logger.info(
            "Validated %d candidates for '%s'",
            candidates_accepted,
            selected_node,
        )

        accepted_similarities = []
        added_candidates: List[Dict[str, str]] = []

        for candidate in accepted_candidates:
            candidate_term = candidate.get("term", "")
            candidate_desc = candidate.get("description", "")

            try:
                self._add_candidate_to_graph(selected_node, candidate)
                similarity = self._get_similarity_cached(
                    term_a=self.ontology_graph.nodes[selected_node].get(
                        "term", selected_node),
                    description_a=self.ontology_graph.nodes[selected_node].get(
                        "description", ""),
                    term_b=candidate_term,
                    description_b=candidate_desc,
                    context_label=f"Phase 4 accepted candidate scoring for {selected_node}",
                ) / 100.0
                accepted_similarities.append(similarity)
                added_candidates.append(candidate)
            except Exception as exc:
                logger.error(
                    "Error adding candidate '%s' to graph: %s",
                    candidate_term,
                    exc,
                )
                continue

        self._check_cross_branch_links_batch(added_candidates)

        reward = self._compute_expansion_reward(
            accepted_similarities=accepted_similarities,
            candidates_generated=candidates_generated,
        )
        logger.info("Expansion reward for '%s': %.3f", selected_node, reward)

        self._update_bandit(selected_node, reward, candidates_accepted)

        result = {
            "node": selected_node,
            "candidates_generated": candidates_generated,
            "candidates_accepted": candidates_accepted,
            "reward": reward,
        }
        logger.info(
            "Expansion iteration complete: %d/%d candidates added to graph. Reward: %.3f",
            candidates_accepted,
            candidates_generated,
            reward,
        )
        return result

    def expand_node(
        self: "Ontology",
        node: str,
    ) -> Dict[str, Any]:
        """Expand a specific node selected manually by the caller.

        Manual expansion reuses the same generation, validation, graph update,
        cross-linking, and reward bookkeeping as automatic UCB1 expansion. Once
        manual expansion starts on an Ontology instance, that instance can no
        longer switch back to automatic expansion.
        """
        self._validate_manual_expansion_target(node)
        self._register_expansion_mode("manual")
        return self._expand_selected_node(node)

    def expand_ontology(self: "Ontology") -> Dict[str, Any]:
        """Run one iteration of automatic UCB1-guided ontology expansion."""
        self._register_expansion_mode("automatic")

        selected_node = self._select_node_ucb1()
        if selected_node is None:
            logger.info("Expansion complete: no expandable nodes remaining")
            return self._empty_expansion_result()

        return self._expand_selected_node(selected_node)

    def _run_expansion_phase(
        self: "Ontology",
        pipeline_start: float,
    ) -> _ExpansionPhaseState:
        """Run Phase 4 and record its timing, progress output, and history."""
        print_phase(4, "UCB1 Iterative Expansion")
        phase_start = time.monotonic()
        logger.info("Phase 4: Running expansion loop with UCB1 selection")

        self._print_expansion_table_header()
        state = _ExpansionPhaseState(
            prev_node_count=self.ontology_graph.number_of_nodes(),
        )

        for iteration in range(self.max_iterations):
            should_stop = self._run_expansion_iteration(
                iteration_count=iteration + 1,
                state=state,
                pipeline_start=pipeline_start,
            )
            if should_stop:
                break

        phase_duration = time.monotonic() - phase_start
        self.history.phases.append(PhaseRecord(
            phase=4,
            name="UCB1 expansion",
            duration_seconds=phase_duration,
            details={
                "iterations": state.iteration_count,
                "final_plateau_count": state.plateau_count,
                "termination_reason": state.termination_reason,
            },
        ))
        print(
            f"\n  Expansion finished: {state.iteration_count} iterations in {phase_duration:.1f}s ({state.termination_reason})"
        )
        logger.info(
            "Expansion loop complete: %d iterations, graph now has %d nodes",
            state.iteration_count,
            self.ontology_graph.number_of_nodes(),
        )
        return state

    def _run_expansion_iteration(
        self: "Ontology",
        iteration_count: int,
        state: _ExpansionPhaseState,
        pipeline_start: float,
    ) -> bool:
        """Run one Phase 4 iteration and return True when expansion should stop."""
        state.iteration_count = iteration_count
        logger.info("Expansion iteration %d/%d",
                    iteration_count, self.max_iterations)

        if self._run_periodic_class_discovery(iteration_count):
            state.stagnation_count = 0

        stats = self.expand_ontology()

        if stats["node"] is None:
            self._handle_no_expandable_nodes(
                iteration_count=iteration_count,
                state=state,
                pipeline_start=pipeline_start,
            )
            return True

        snapshot = self._build_expansion_iteration_snapshot(stats)
        self._update_plateau_tracking(
            state=state,
            reward=snapshot.reward,
            candidates_accepted=snapshot.accepted,
        )
        self._update_stagnation_tracking(
            state=state,
            current_node_count=snapshot.cumulative_nodes,
        )
        self._record_expansion_iteration(
            iteration_count=iteration_count,
            state=state,
            snapshot=snapshot,
            pipeline_start=pipeline_start,
        )
        self._log_retired_expansion_node(snapshot.node)

        status = self._build_expansion_status(snapshot.node, state)
        status = self._resolve_expansion_termination(
            status=status,
            state=state,
            current_node_count=snapshot.cumulative_nodes,
        )
        self._print_expansion_iteration_row(
            iteration_count=iteration_count,
            snapshot=snapshot,
            status=status,
        )

        logger.info(
            "Iteration %d: generated=%d, accepted=%d, reward=%.3f",
            iteration_count,
            snapshot.generated,
            snapshot.accepted,
            snapshot.reward,
        )
        return "CONVERGED" in status

    def _print_expansion_table_header(self: "Ontology") -> None:
        """Print the Phase 4 progress table header."""
        print(
            f"\n  {'Iter':>4s}  {'Node':<20s}  {'Gen':>4s}  {'Acc':>4s}  {'Rate':>6s}  {'Reward':>7s}  {'Nodes':>5s}  {'Edges':>5s}  {'Status'}"
        )
        print(
            f"  {'─' * 4}  {'─' * 20}  {'─' * 4}  {'─' * 4}  {'─' * 6}  {'─' * 7}  {'─' * 5}  {'─' * 5}  {'─' * 12}"
        )

    def _run_periodic_class_discovery(
        self: "Ontology",
        iteration_count: int,
    ) -> bool:
        """Discover new top-level classes when the configured interval is reached."""
        if not (
            self.class_discovery_interval > 0
            and iteration_count > 1
            and (iteration_count - 1) % self.class_discovery_interval == 0
        ):
            return False

        new_classes = self._discover_new_classes(num_classes=2)
        if not new_classes:
            return False

        print(
            f"  {'':>4s}  {'[discovery]':<20s}  {'':>4s}  {len(new_classes):>4d}  {'':>6s}  {'':>7s}  {self.ontology_graph.number_of_nodes():>5d}  {self.ontology_graph.number_of_edges():>5d}  new classes: {', '.join(new_classes)}"
        )
        return True

    def _handle_no_expandable_nodes(
        self: "Ontology",
        iteration_count: int,
        state: _ExpansionPhaseState,
        pipeline_start: float,
    ) -> None:
        """Record and print the terminal iteration when no nodes remain."""
        state.termination_reason = "no_expandable_nodes"
        logger.info("No expandable nodes remaining; terminating expansion")

        self.history.expansion_records.append(ExpansionRecord(
            iteration=iteration_count,
            node_expanded=None,
            candidates_generated=0,
            candidates_accepted=0,
            reward=0.0,
            cumulative_nodes=self.ontology_graph.number_of_nodes(),
            cumulative_edges=self.ontology_graph.number_of_edges(),
            acceptance_rate=0.0,
            plateau_count=state.plateau_count,
            stagnation_count=state.stagnation_count,
            elapsed_seconds=time.monotonic() - pipeline_start,
        ))
        print(
            f"  {iteration_count:>4d}  {'—':<20s}  {'—':>4s}  {'—':>4s}  {'—':>6s}  {'—':>7s}  {self.ontology_graph.number_of_nodes():>5d}  {self.ontology_graph.number_of_edges():>5d}  no nodes left"
        )

    def _build_expansion_iteration_snapshot(
        self: "Ontology",
        stats: Dict[str, Any],
    ) -> _ExpansionIterationSnapshot:
        """Normalize per-iteration stats into a structured snapshot."""
        generated = stats["candidates_generated"]
        accepted = stats["candidates_accepted"]
        acceptance_rate = accepted / generated if generated > 0 else 0.0
        return _ExpansionIterationSnapshot(
            node=stats["node"],
            generated=generated,
            accepted=accepted,
            reward=stats["reward"],
            acceptance_rate=acceptance_rate,
            cumulative_nodes=self.ontology_graph.number_of_nodes(),
            cumulative_edges=self.ontology_graph.number_of_edges(),
        )

    def _update_plateau_tracking(
        self: "Ontology",
        state: _ExpansionPhaseState,
        reward: float,
        candidates_accepted: int,
    ) -> None:
        """Update the productive-iteration plateau counter."""
        if candidates_accepted <= 0:
            return

        if state.productive_reward_history:
            delta = abs(reward - state.productive_reward_history[-1])
            if delta < _EXPANSION_PLATEAU_DELTA:
                state.plateau_count += 1
                logger.debug(
                    "Plateau detected (delta=%.4f); count=%d",
                    delta,
                    state.plateau_count,
                )
            else:
                state.plateau_count = 0
                logger.debug(
                    "Reward changed (delta=%.4f); plateau reset", delta)

        state.productive_reward_history.append(reward)

    def _update_stagnation_tracking(
        self: "Ontology",
        state: _ExpansionPhaseState,
        current_node_count: int,
    ) -> None:
        """Track consecutive iterations where the graph does not grow."""
        if current_node_count > state.prev_node_count:
            state.stagnation_count = 0
        else:
            state.stagnation_count += 1
            logger.debug(
                "Stagnation detected (nodes unchanged at %d); count=%d",
                current_node_count,
                state.stagnation_count,
            )

        state.prev_node_count = current_node_count

    def _record_expansion_iteration(
        self: "Ontology",
        iteration_count: int,
        state: _ExpansionPhaseState,
        snapshot: _ExpansionIterationSnapshot,
        pipeline_start: float,
    ) -> None:
        """Append a structured ExpansionRecord for one iteration."""
        self.history.expansion_records.append(ExpansionRecord(
            iteration=iteration_count,
            node_expanded=snapshot.node,
            candidates_generated=snapshot.generated,
            candidates_accepted=snapshot.accepted,
            reward=snapshot.reward,
            cumulative_nodes=snapshot.cumulative_nodes,
            cumulative_edges=snapshot.cumulative_edges,
            acceptance_rate=snapshot.acceptance_rate,
            plateau_count=state.plateau_count,
            stagnation_count=state.stagnation_count,
            elapsed_seconds=time.monotonic() - pipeline_start,
        ))

    def _log_retired_expansion_node(self: "Ontology", node: str) -> None:
        """Log when the just-expanded node was retired by the bandit update."""
        if not self._is_retired_node(node):
            return

        retired_level = self.ontology_graph.nodes[node].get("level", "?")
        logger.info("Node '%s' (%s) retired", node, retired_level)

    def _is_retired_node(self: "Ontology", node: Optional[str]) -> bool:
        """Return True when a graph node exists and is marked as retired."""
        return (
            node is not None
            and node in self.ontology_graph
            and self.ontology_graph.nodes[node].get("retired", False)
        )

    def _build_expansion_status(
        self: "Ontology",
        node: str,
        state: _ExpansionPhaseState,
    ) -> str:
        """Build the status label printed for an expansion iteration."""
        status_parts: List[str] = []
        if self._is_retired_node(node):
            status_parts.append("RETIRED")
        if state.plateau_count > 0:
            status_parts.append(f"plateau({state.plateau_count})")
        if state.stagnation_count > 0:
            status_parts.append(f"stagnant({state.stagnation_count})")
        return " ".join(status_parts)

    def _resolve_expansion_termination(
        self: "Ontology",
        status: str,
        state: _ExpansionPhaseState,
        current_node_count: int,
    ) -> str:
        """Apply early-termination rules and return the final status label."""
        plateau_reason = self._get_plateau_termination_reason(state)
        if plateau_reason is not None:
            state.termination_reason = plateau_reason
            logger.info("Early termination: %s", state.termination_reason)
            return "CONVERGED (plateau)"

        if state.stagnation_count >= _EXPANSION_STAGNATION_LIMIT:
            state.termination_reason = (
                f"graph stagnant for {state.stagnation_count} consecutive iterations "
                f"(stuck at {current_node_count} nodes)"
            )
            logger.info("Early termination: %s", state.termination_reason)
            return "CONVERGED (stagnant)"

        return status

    def _get_plateau_termination_reason(
        self: "Ontology",
        state: _ExpansionPhaseState,
    ) -> Optional[str]:
        """Return a plateau termination reason after the cold-start phase has finished."""
        if state.plateau_count < _EXPANSION_PLATEAU_LIMIT:
            return None

        current_expandable = self._get_current_expandable_nodes()
        all_expandable_visited = all(
            self.ontology_graph.nodes[node].get("n_visits", 0) >= 1
            for node in current_expandable
        )
        if not all_expandable_visited:
            return None

        if not self._has_revisited_expandable_node(current_expandable):
            logger.debug(
                "Plateau detected but convergence deferred until an expandable node is revisited"
            )
            return None

        return (
            f"reward plateau for {state.plateau_count} productive iterations "
            f"after at least one revisit and all {len(current_expandable)} expandable nodes visited"
        )

    def _has_revisited_expandable_node(
        self: "Ontology",
        current_expandable: List[Any],
    ) -> bool:
        """Return True when some current expandable node has been expanded more than once."""
        return any(
            self.ontology_graph.nodes[node].get("n_visits", 0) > 1
            for node in current_expandable
        )

    def _get_current_expandable_nodes(self: "Ontology") -> List[Any]:
        """Return all currently expandable, non-retired graph nodes."""
        expandable_level_names = frozenset(
            level.name for level in self.level_schema if level.expandable
        )
        return [
            node for node in self.ontology_graph.nodes()
            if self.ontology_graph.nodes[node].get("level") in expandable_level_names
            and not self.ontology_graph.nodes[node].get("retired", False)
        ]

    def _print_expansion_iteration_row(
        self: "Ontology",
        iteration_count: int,
        snapshot: _ExpansionIterationSnapshot,
        status: str,
    ) -> None:
        """Print a single Phase 4 progress row."""
        node_display = self._format_expansion_node_display(snapshot.node)
        print(
            f"  {iteration_count:>4d}  {node_display:<20s}  {snapshot.generated:>4d}  {snapshot.accepted:>4d}  {snapshot.acceptance_rate:>5.0%}   {snapshot.reward:>6.3f}  {snapshot.cumulative_nodes:>5d}  {snapshot.cumulative_edges:>5d}  {status}"
        )

    def _format_expansion_node_display(self: "Ontology", node: str) -> str:
        """Truncate long node names so the expansion table stays aligned."""
        if len(node) > 20:
            return node[:18] + ".."
        return node
