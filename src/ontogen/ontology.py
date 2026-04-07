"""
Ontology construction, expansion, and RDF serialization.

This module contains the Ontology class which orchestrates the full
ontology generation pipeline: seed generation, iterative expansion,
graph construction, and RDF serialization.
"""

import concurrent.futures
import io
import itertools
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple, Any

import pandas as pd
import networkx as nx
import pydotplus
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS
from rdflib.tools.rdf2dot import rdf2dot
from IPython.display import display, Image

from ontogen.llm_client import ChatGpt

logger = logging.getLogger(__name__)


@dataclass
class OntologyLevel:
    """Defines one tier in the ontology hierarchy.

    Attributes:
        name: Level identifier used in DiGraph node attributes (e.g., "class").
        relation_to_parent: Edge relation label for edges FROM parent TO this level
            (e.g., "subClassOf"). None for the root level.
        rdf_predicate: The RDF predicate string to use when serializing edges
            from parent to this level (e.g., "rdfs:subClassOf", "rdf:type").
            None for the root level.
        is_rdf_class: Whether nodes at this level should get `rdf:type rdfs:Class`.
        expandable: Whether nodes at this level can be expanded (generate children).
            Leaf levels (e.g., instances) are not expandable.
        seed_key: The JSON key used in the seed taxonomy for this level's term name
            (e.g., "class" for classes/subclasses, "term" for instances).
        children_key: The JSON key used in the seed taxonomy to find children of this
            level (e.g., "subclasses", "instances"). None for leaf levels.
    """

    name: str
    relation_to_parent: Optional[str] = None
    rdf_predicate: Optional[str] = None
    is_rdf_class: bool = True
    expandable: bool = True
    seed_key: str = "class"
    children_key: Optional[str] = None


DEFAULT_LEVEL_SCHEMA: List[OntologyLevel] = [
    OntologyLevel(
        name="class",
        relation_to_parent=None,
        rdf_predicate=None,
        is_rdf_class=True,
        expandable=True,
        seed_key="class",
        children_key="subclasses",
    ),
    OntologyLevel(
        name="subclass",
        relation_to_parent="subClassOf",
        rdf_predicate="rdfs:subClassOf",
        is_rdf_class=True,
        expandable=True,
        seed_key="class",
        children_key="instances",
    ),
    OntologyLevel(
        name="instance",
        relation_to_parent="type",
        rdf_predicate="rdf:type",
        is_rdf_class=False,
        expandable=False,
        seed_key="term",
        children_key=None,
    ),
]


@dataclass
class ExpansionRecord:
    """Metrics captured for a single expansion iteration.

    Tracks what happened during one UCB1-guided expansion step, including
    which node was expanded, how many candidates were generated/accepted,
    the reward signal, and the cumulative graph state at that point.

    Attributes:
        iteration: 1-based iteration number.
        node_expanded: The node ID selected for expansion, or None if none available.
        candidates_generated: Number of candidate terms returned by the LLM.
        candidates_accepted: Number of candidates passing the similarity threshold.
        reward: Mean similarity of accepted candidates (0–1 scale).
        cumulative_nodes: Total graph nodes after this iteration.
        cumulative_edges: Total graph edges after this iteration.
        acceptance_rate: Fraction of generated candidates that were accepted (0–1).
        plateau_count: Current consecutive plateau counter at this iteration.
        stagnation_count: Consecutive iterations where graph node count did not grow.
        elapsed_seconds: Wall-clock seconds since the pipeline started.
    """

    iteration: int
    node_expanded: Optional[str]
    candidates_generated: int
    candidates_accepted: int
    reward: float
    cumulative_nodes: int
    cumulative_edges: int
    acceptance_rate: float
    plateau_count: int
    stagnation_count: int
    elapsed_seconds: float


@dataclass
class PhaseRecord:
    """Timing and summary for a single pipeline phase.

    Attributes:
        phase: 1-based phase number.
        name: Human-readable phase name (e.g., "Seed generation").
        duration_seconds: Wall-clock duration of this phase.
        details: Arbitrary key-value details about what happened in the phase.
    """

    phase: int
    name: str
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationHistory:
    """Complete audit trail of an ontology generation run.

    Stores configuration, per-phase timing, per-iteration expansion metrics,
    and final summary statistics. Designed for post-hoc analysis, convergence
    plotting, and reproducibility auditing.

    Attributes:
        domain: The domain that was used for generation.
        started_at: ISO 8601 timestamp when the pipeline started.
        completed_at: ISO 8601 timestamp when the pipeline finished (None if still running).
        config: Snapshot of generation parameters (thresholds, max_iterations, etc.).
        phases: Ordered list of phase records with timing.
        expansion_records: Ordered list of per-iteration expansion metrics.
        total_iterations: Number of expansion iterations that ran.
        final_nodes: Final node count in the ontology graph.
        final_edges: Final edge count in the ontology graph.
        final_triples: Number of RDF triples in the serialized output.
        early_terminated: Whether the expansion loop stopped before max_iterations.
        termination_reason: Human-readable reason for termination.
    """

    domain: str
    started_at: str
    completed_at: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    phases: List[PhaseRecord] = field(default_factory=list)
    expansion_records: List[ExpansionRecord] = field(default_factory=list)
    total_iterations: int = 0
    final_nodes: int = 0
    final_edges: int = 0
    final_triples: int = 0
    early_terminated: bool = False
    termination_reason: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert expansion records to a pandas DataFrame for tabular analysis.

        Returns:
            DataFrame with one row per expansion iteration, columns matching
            ExpansionRecord fields.
        """
        if not self.expansion_records:
            return pd.DataFrame()
        from dataclasses import asdict

        return pd.DataFrame([asdict(r) for r in self.expansion_records])

    def summary(self) -> str:
        """Return a human-readable summary of the generation run.

        Returns:
            Multi-line string summarizing domain, timing, iterations,
            graph size, and termination status.
        """
        lines = [
            f"{'=' * 60}",
            f"  Ontology Generation Summary",
            f"{'=' * 60}",
            f"  Domain:            {self.domain}",
            f"  Started:           {self.started_at}",
            f"  Completed:         {self.completed_at or 'N/A'}",
            f"  Iterations:        {self.total_iterations}",
            f"  Final nodes:       {self.final_nodes}",
            f"  Final edges:       {self.final_edges}",
            f"  RDF triples:       {self.final_triples}",
            f"  Early terminated:  {self.early_terminated}",
        ]
        if self.termination_reason:
            lines.append(f"  Reason:            {self.termination_reason}")

        # Phase timing breakdown
        if self.phases:
            lines.append(f"{'─' * 60}")
            lines.append("  Phase Timing:")
            for phase in self.phases:
                lines.append(f"    {phase.phase}. {phase.name:<30s} {phase.duration_seconds:>7.1f}s")

        # Expansion summary statistics
        if self.expansion_records:
            rewards = [r.reward for r in self.expansion_records]
            accepted = [r.candidates_accepted for r in self.expansion_records]
            generated = [r.candidates_generated for r in self.expansion_records]
            lines.append(f"{'─' * 60}")
            lines.append("  Expansion Stats:")
            lines.append(f"    Avg reward:        {sum(rewards) / len(rewards):.3f}")
            lines.append(f"    Total generated:   {sum(generated)}")
            lines.append(f"    Total accepted:    {sum(accepted)}")
            total_gen = sum(generated)
            overall_rate = sum(accepted) / total_gen if total_gen > 0 else 0
            lines.append(f"    Overall accept %:  {overall_rate:.1%}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


class Ontology:
    """
    Orchestrates ontology generation from a domain using LLM-driven expansion.

    Manages the full pipeline: structured seed generation, pairwise similarity
    validation, UCB1-guided iterative expansion, and RDF/OWL serialization.

    Attributes:
        domain: The target domain for ontology generation (e.g., "Star Trek").
        agent: ChatGpt instance for LLM interactions.
        seed: Initial seed terms to bootstrap the ontology.
        exploration_constant: UCB1 exploration parameter. Defaults to 2.
        max_iterations: Maximum expansion iterations. Defaults to 10.
        similarity_threshold: Minimum similarity to keep an edge. Defaults to 0.5.
        confidence_threshold: Minimum confidence for relationship classification.
        candidates_per_iteration: Number of new terms to generate per expansion.
        ontology_graph: Internal directed graph (networkx DiGraph).
        rdf: RDFLib Graph for RDF/OWL serialization.
    """

    def __init__(
        self,
        domain: str,
        agent: ChatGpt,
        seed: Optional[Dict[str, Any]] = None,
        exploration_constant: float = 2,
        max_iterations: int = 10,
        similarity_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
        candidates_per_iteration: int = 20,
        level_schema: Optional[List[OntologyLevel]] = None,
        cross_link_threshold: float = 70,
        max_workers: int = 5,
        class_discovery_interval: int = 0,
        retirement_limit: int = 3,
    ) -> None:
        # Configuration
        self.domain = domain
        self.max_workers = max_workers
        self.agent = agent
        self.seed = seed
        self.exploration_constant = exploration_constant
        self.max_iterations = max_iterations
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.candidates_per_iteration = candidates_per_iteration
        self.level_schema = level_schema or DEFAULT_LEVEL_SCHEMA
        self.cross_link_threshold = cross_link_threshold
        self.class_discovery_interval = class_discovery_interval
        self.retirement_limit = retirement_limit

        # Internal graph representation
        self.ontology_graph = nx.DiGraph()

        # Similarity cache: maps sorted term pairs to similarity scores
        self.similarity_cache: Dict[Tuple[str, str], float] = {}

        # Term and community tracking
        self.terms: Optional[list] = None
        self.communities: Optional[dict] = None

        # RDF serialization
        self.rdf = Graph()
        self.turtle: Optional[str] = None
        self.base_namespace = Namespace("http://example.org/ontology/")

        # Generation history — populated by generate_ontology()
        self.history: Optional[GenerationHistory] = None

    def _get_level(self, name: str) -> OntologyLevel:
        """Retrieve a level definition by name.

        Args:
            name: The level name (e.g., "class", "subclass", "instance").

        Returns:
            The OntologyLevel definition matching the name.

        Raises:
            ValueError: If the level name is not found in the schema.
        """
        for level in self.level_schema:
            if level.name == name:
                return level
        raise ValueError(f"Level '{name}' not found in level_schema")

    def _get_child_level(self, parent_level_name: str) -> Optional[OntologyLevel]:
        """Get the next level down in the hierarchy.

        Args:
            parent_level_name: The name of the parent level.

        Returns:
            The child OntologyLevel, or None if the parent is a leaf level.

        Raises:
            ValueError: If the parent level name is not found.
        """
        parent_index = None
        for i, level in enumerate(self.level_schema):
            if level.name == parent_level_name:
                parent_index = i
                break

        if parent_index is None:
            raise ValueError(f"Parent level '{parent_level_name}' not found in level_schema")

        if parent_index + 1 < len(self.level_schema):
            return self.level_schema[parent_index + 1]
        return None

    def _get_similarity_cached(
        self,
        term_a: str,
        term_b: str,
        description_a: Optional[str] = None,
        description_b: Optional[str] = None,
    ) -> float:
        """Get similarity between two terms, using cache to avoid redundant LLM calls.

        Checks self.similarity_cache first. If the pair has not been evaluated,
        calls the LLM (choosing method based on description availability) and stores
        the result in the cache before returning.

        The cache key is order-agnostic: _get_similarity_cached("A", "B") and
        _get_similarity_cached("B", "A") will hit the same cache entry.

        Args:
            term_a: First term (e.g., "Vulcans").
            term_b: Second term (e.g., "Spock").
            description_a: Optional description of term_a for semantic context.
            description_b: Optional description of term_b for semantic context.

        Returns:
            Similarity score (0–100) as a float. Score is always cached after
            the first evaluation.

        Implementation:
            1. Create cache key: tuple(sorted([term_a, term_b]))
            2. Check cache; if hit, log and return
            3. If cache miss:
               - Choose LLM method (with or without descriptions)
               - Call the method
               - Extract score from response dict
               - Store in cache
               - Return score
        """
        # Step 1: Create order-agnostic cache key
        cache_key = tuple(sorted([term_a, term_b]))

        # Step 2: Check cache first
        if cache_key in self.similarity_cache:
            cached_score = self.similarity_cache[cache_key]
            logger.debug(f"Cache hit: similarity({term_a}, {term_b}) = {cached_score}")
            return cached_score

        # Step 3: Cache miss — evaluate similarity
        logger.info(f"Cache miss: evaluating similarity({term_a}, {term_b})")
        response = self.agent.get_similarity_with_descriptions(
            term_x=term_a,
            description_x=description_a,
            term_y=term_b,
            description_y=description_b,
        )

        # Step 4: Parse response defensively
        score = response.get("similarity", 0.0)
        if score is None:
            logger.warning(f"LLM returned None for similarity({term_a}, {term_b}); defaulting to 0")
            score = 0.0
        score = float(score)

        # Step 5: Store and return
        self.similarity_cache[cache_key] = score
        logger.info(f"Similarity({term_a}, {term_b}) = {score}")
        return score

    def generate_initial_terms(self, num_classes: int = 5) -> Optional[Dict[str, Any]]:
        """Generate a structured multi-level taxonomic skeleton from the domain using the LLM.

        Uses self.level_schema to dynamically build a prompt requesting the hierarchy.
        The LLM response must be valid JSON conforming to the seed_keys defined in 
        self.level_schema.

        For the default 3-level schema (class → subclass → instance), the prompt requests
        classes, subclasses, and instances for the given domain. The method dynamically
        adapts the prompt for any custom level_schema.

        Args:
            num_classes: Number of top-level entries to generate. Defaults to 5.

        Returns:
            A dict with keys:
            - "domain": The domain name (str)
            - "taxonomy": A list of top-level entries (whose structure depends on level_schema)
            
            Returns None if JSON parsing fails or the LLM response is invalid.
            
        Example:
            >>> ontology = Ontology(domain="Star Trek", agent=mock_agent)
            >>> seed = ontology.generate_initial_terms(num_classes=2)
            >>> seed["domain"]
            'Star Trek'
            >>> len(seed["taxonomy"])
            2
        """
        # Build dynamic description of the hierarchy from level_schema
        level_descriptions = []
        for i, level in enumerate(self.level_schema):
            level_num = i + 1
            if i == 0:
                level_descriptions.append(
                    f"LEVEL {level_num}: {level.name.upper()} - Top-level abstract categories in {self.domain}"
                )
            else:
                parent_level = self.level_schema[i - 1]
                level_descriptions.append(
                    f"LEVEL {level_num}: {level.name.upper()} - Specific items within each {parent_level.name}"
                )
        
        hierarchy_description = "\n".join(level_descriptions)

        # Build a JSON schema example dynamically from level_schema
        # Start with the simplest case: build the innermost structure first
        example_schema = self._build_example_schema()

        # Construct the LLM prompt
        instructions = (
            "You are an ontology engineer specialist. Generate accurate, specific taxonomies "
            "for the given domain based on the requested hierarchy."
        )

        prompt = f"""You are an ontology engineer specialist. Generate a detailed taxonomic hierarchy 
for the given domain.

DOMAIN: {self.domain}

TASK:
Generate a structured taxonomy with {len(self.level_schema)} levels:
{hierarchy_description}

{self._build_count_instructions(num_classes)}

EXPECTED JSON STRUCTURE:
{example_schema}

CRITICAL REQUIREMENTS:
- Return ONLY valid JSON (no markdown code blocks or preamble)
- Each entry must have a unique name and description
- All descriptions must be concise (1-2 sentences)
- Ensure entries are concrete, recognizable examples for the domain "{self.domain}"
- Do NOT encapsulate the output in code blocks

START YOUR RESPONSE WITH {{ CHARACTER. OUTPUT ONLY VALID JSON."""

        # Call the LLM
        try:
            raw_response = self.agent.chat(instructions=instructions, input=prompt)
            logger.debug(f"Raw seed response (first 500 chars): {raw_response[:500]}")
        except Exception as e:
            logger.error(f"LLM chat call failed: {e}")
            return None

        # Parse JSON defensively
        try:
            seed_dict = json.loads(raw_response)
            
            # Validate the structure
            if not isinstance(seed_dict, dict):
                logger.error(f"Seed response is not a dict: {type(seed_dict)}")
                return None
            
            if "domain" not in seed_dict or "taxonomy" not in seed_dict:
                logger.error("Seed response missing 'domain' or 'taxonomy' keys")
                return None
            
            if not isinstance(seed_dict.get("taxonomy"), list):
                logger.error("'taxonomy' value is not a list")
                return None
            
            if len(seed_dict.get("taxonomy", [])) == 0:
                logger.error("'taxonomy' list is empty")
                return None

            self.seed = seed_dict
            logger.info(
                f"Parsed seed successfully: domain={seed_dict['domain']}, "
                f"taxonomy count={len(seed_dict['taxonomy'])}"
            )
            return seed_dict

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse seed JSON: {e}. Raw response (first 300 chars): {raw_response[:300]}"
            )
            return None

    def _build_count_instructions(self, num_classes: int) -> str:
        """Build count instruction lines for the seed prompt, safe for any schema depth.

        Dynamically generates instructions like:
        - "Generate exactly 5 top-level classs."
        - "For each class, include 2-4 subclasss."
        - "For each subclass, include 2-3 instances (if applicable)."

        Args:
            num_classes: Number of top-level entries to generate.

        Returns:
            A multi-line string with generation count instructions.
        """
        lines = [f"Generate exactly {num_classes} top-level {self.level_schema[0].name.lower()}s."]
        counts = [(2, 4), (2, 3), (1, 3)]  # Default child count ranges per level depth
        for i in range(len(self.level_schema) - 1):
            parent_name = self.level_schema[i].name.lower()
            child_name = self.level_schema[i + 1].name.lower()
            lo, hi = counts[min(i, len(counts) - 1)]
            suffix = " (if applicable)" if i > 0 else ""
            lines.append(f"For each {parent_name}, include {lo}-{hi} {child_name}s{suffix}.")
        return "\n".join(lines)

    def _build_example_schema(self) -> str:
        """Build a JSON schema example string from self.level_schema.
        
        Dynamically generates a nested example structure that matches the
        expected format for the LLM to return. Programmatically builds a nested
        dict and serializes it using json.dumps() to ensure valid JSON for any
        schema depth.
        
        Returns:
            A formatted JSON string showing the expected structure.
        """
        # Start with domain and empty taxonomy
        result = {
            "domain": self.domain,
            "taxonomy": []
        }
        
        # Build one example row in the taxonomy
        root_level = self.level_schema[0]
        current = {
            root_level.seed_key: f"Example {root_level.name.capitalize()}",
            "description": "Brief description"
        }
        
        # Iterate through remaining levels, nesting them
        # We maintain a reference to the innermost item to add children to
        innermost = current
        for level_idx in range(1, len(self.level_schema)):
            level = self.level_schema[level_idx]
            parent_level = self.level_schema[level_idx - 1]
            
            # Add children key if parent has children
            if parent_level.children_key is not None:
                child_item = {
                    level.seed_key: f"Example {level.name.capitalize()}",
                    "description": "Brief description"
                }
                
                # Check if this child level has its own children
                if level.children_key is not None:
                    # Add example children
                    child_item[level.children_key] = [
                        {"term": "Example instance", "description": "..."},
                        {"term": "Another instance", "description": "..."}
                    ]
                
                innermost[parent_level.children_key] = [child_item]
                innermost = child_item
        
        result["taxonomy"].append(current)
        
        return json.dumps(result, indent=2)

    def create_seed_ontology(self) -> None:
        """Create the initial ontology skeleton from the structured seed.
        
        Parses self.seed (obtained from generate_initial_terms()) and recursively
        traverses the taxonomy tree using self.level_schema to determine:
        - Which JSON key holds the term name (level.seed_key)
        - Which JSON key holds children (level.children_key)
        - What level name to assign (level.name)
        - What relation to assign to edges (level.relation_to_parent)
        
        Stores result in self.ontology_graph.
        
        Raises:
            ValueError: If self.seed is None.
        """
        if self.seed is None:
            logger.error("seed is None - call generate_initial_terms() first")
            return

        # Clear the graph to ensure idempotency
        self.ontology_graph.clear()

        # Get the root level (first level in the schema)
        root_level = self.level_schema[0]
        taxonomy = self.seed.get("taxonomy", [])

        # Process each root item in the taxonomy
        for root_item in taxonomy:
            self._process_taxonomy_item(root_item, root_level, parent_id=None)

        logger.info(f"Created seed ontology with {len(self.ontology_graph)} nodes")

    def _process_taxonomy_item(
        self,
        item: Dict[str, Any],
        level: OntologyLevel,
        parent_id: Optional[str],
    ) -> Optional[str]:
        """Recursively process a single taxonomy item from the seed.
        
        Adds a node and edge to the graph, then recursively processes children.
        
        Args:
            item: A dict containing the term and metadata (from seed JSON).
            level: The OntologyLevel definition for this item's level.
            parent_id: The term string of the parent node (None if root).
        
        Returns:
            The term string (node ID) if successfully added, None otherwise.
        """
        # Extract the term name using the level's seed_key
        term = item.get(level.seed_key)
        if not term:
            logger.warning(f"Item missing '{level.seed_key}': {item}")
            return None

        # Extract description
        description = item.get("description", "")

        # Check for duplicate term
        if term in self.ontology_graph:
            logger.warning(
                f"Duplicate term '{term}' at level '{level.name}' - skipping"
            )
            return term

        # Add node to the graph with all required attributes
        self.ontology_graph.add_node(
            term,
            term=term,
            description=description,
            level=level.name,
            n_visits=0,
            total_reward=0.0,
        )

        # Add edge from parent if this is not the root level
        if parent_id is not None and level.relation_to_parent:
            self.ontology_graph.add_edge(
                parent_id,
                term,
                relation=level.relation_to_parent,
            )

        # Process children recursively if this level has children
        child_level = self._get_child_level(level.name)
        if child_level and level.children_key:
            children = item.get(level.children_key, [])
            for child_item in children:
                self._process_taxonomy_item(child_item, child_level, parent_id=term)

        return term

    def _precompute_similarities_parallel(
        self,
        pairs: List[Dict[str, str]],
        term_a_key: str = "term_a",
        term_b_key: str = "term_b",
        desc_a_key: str = "description_a",
        desc_b_key: str = "description_b",
    ) -> None:
        """Pre-fill the similarity cache for multiple pairs using concurrent threads.

        Filters out pairs already in the cache, then evaluates the remaining
        pairs in parallel via ThreadPoolExecutor.  After this call, every
        pair is guaranteed to be in ``self.similarity_cache`` so subsequent
        calls to ``_get_similarity_cached()`` will be instant cache hits.

        Thread-safety note: each uncached pair maps to a unique sorted cache
        key, so concurrent dict writes do not collide.  CPython's GIL also
        makes simple dict assignments atomic.

        Args:
            pairs: List of dicts whose keys are given by the ``*_key`` args.
            term_a_key: Dict key for the first term.
            term_b_key: Dict key for the second term.
            desc_a_key: Dict key for the first description.
            desc_b_key: Dict key for the second description.
        """
        # Deduplicate & filter out cached pairs
        uncached: Dict[Tuple[str, str], Dict[str, str]] = {}
        for pair in pairs:
            cache_key = tuple(sorted([pair[term_a_key], pair[term_b_key]]))
            if cache_key not in self.similarity_cache and cache_key not in uncached:
                uncached[cache_key] = pair

        if not uncached:
            logger.debug("All pairs already cached; nothing to compute")
            return

        logger.info(
            f"Pre-computing {len(uncached)} uncached similarity pairs "
            f"with {self.max_workers} workers"
        )

        def _evaluate(pair: Dict[str, str]) -> float:
            return self._get_similarity_cached(
                term_a=pair[term_a_key],
                term_b=pair[term_b_key],
                description_a=pair.get(desc_a_key),
                description_b=pair.get(desc_b_key),
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_key = {
                executor.submit(_evaluate, pair): key
                for key, pair in uncached.items()
            }
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    future.result()  # result is cached inside _get_similarity_cached
                except Exception as e:
                    logger.error(f"Parallel similarity failed for {key}: {e}")
                    self.similarity_cache[key] = 0.0

        logger.info(f"Parallel pre-computation complete for {len(uncached)} pairs")

    def _generate_validation_pairs(self) -> List[Dict[str, str]]:
        """Generate validation pairs (parent-child, sibling, cross-branch) from the seeded DiGraph.

        Selects ~3n pairs across three categories to enable structural validation via
        pairwise similarity checks with linear-scale LLM calls (~3n instead of n²).

        Pair categories:
        - parent-child: Validates direct hierarchical relationships (edges in graph)
        - sibling: Validates homogeneity within a parent's children
        - cross-branch: Detects possible cross-class links (same semantic domain)

        Returns:
            List of pair dicts, each with keys:
            - "term_x": First term name (str)
            - "desc_x": First term description (str)
            - "term_y": Second term name (str)
            - "desc_y": Second term description (str)
            - "category": Pair category: "parent-child", "sibling", or "cross-branch" (str)

            Rows are unordered; counts approximately: ~1n parent-child + ~1n sibling +
            ~1n cross-branch = ~3n pairs total, where n = number of nodes.
        """
        pairs = []

        # === CATEGORY 1: Parent-child pairs ===
        # For every edge in the graph, create a pair dict
        for parent, child in self.ontology_graph.edges():
            parent_node = self.ontology_graph.nodes[parent]
            child_node = self.ontology_graph.nodes[child]

            pair = {
                "term_x": parent_node["term"],
                "desc_x": parent_node["description"],
                "term_y": child_node["term"],
                "desc_y": child_node["description"],
                "category": "parent-child",
            }
            pairs.append(pair)

        # === CATEGORY 2: Sibling pairs ===
        # For each parent node, get all children and form pairwise combinations
        for parent in self.ontology_graph.nodes():
            children = list(self.ontology_graph.successors(parent))
            if len(children) < 2:
                # Need at least 2 children to form a sibling pair
                continue

            # Generate all combinations of children (unordered pairs)
            for child_a, child_b in itertools.combinations(children, 2):
                child_a_node = self.ontology_graph.nodes[child_a]
                child_b_node = self.ontology_graph.nodes[child_b]

                pair = {
                    "term_x": child_a_node["term"],
                    "desc_x": child_a_node["description"],
                    "term_y": child_b_node["term"],
                    "desc_y": child_b_node["description"],
                    "category": "sibling",
                }
                pairs.append(pair)

        # === CATEGORY 3: Cross-branch pairs ===
        # Identify all top-level (root) nodes using the first level in the schema
        root_level_name = self.level_schema[0].name
        class_nodes = [
            node for node in self.ontology_graph.nodes()
            if self.ontology_graph.nodes[node]["level"] == root_level_name
        ]

        if len(class_nodes) >= 2:
            # For each class node, pick one representative child (first successor if available)
            representatives = {}
            for class_node in class_nodes:
                children = list(self.ontology_graph.successors(class_node))
                if children:
                    # Use the first child as the representative
                    representatives[class_node] = children[0]

            # Form cross-branch pairs from representatives
            rep_list = list(representatives.values())
            for rep_a, rep_b in itertools.combinations(rep_list, 2):
                rep_a_node = self.ontology_graph.nodes[rep_a]
                rep_b_node = self.ontology_graph.nodes[rep_b]

                pair = {
                    "term_x": rep_a_node["term"],
                    "desc_x": rep_a_node["description"],
                    "term_y": rep_b_node["term"],
                    "desc_y": rep_b_node["description"],
                    "category": "cross-branch",
                }
                pairs.append(pair)

        logger.info(
            f"Generated {len(pairs)} validation pairs: "
            f"{sum(1 for p in pairs if p['category'] == 'parent-child')} parent-child, "
            f"{sum(1 for p in pairs if p['category'] == 'sibling')} sibling, "
            f"{sum(1 for p in pairs if p['category'] == 'cross-branch')} cross-branch"
        )

        return pairs

    def validate_structure(self) -> Dict[str, int]:
        """Validate and prune ontology structure based on pairwise similarity.

        Evaluates all validation pairs (parent-child, sibling, cross-branch) via
        cached similarity calls. Applies category-specific thresholds to prune weak
        edges, detects orphaned nodes, and returns summary statistics.

        Algorithm:
        1. Generate validation pairs (parent-child, sibling, cross-branch)
        2. For each pair, compute similarity via cached lookup
        3. Prune weak parent-child edges (similarity < 50%)
        4. Flag sibling dissimilarities (similarity < 30%)
        5. Flag cross-branch link candidates (similarity > 70%)
        6. Detect orphaned nodes (degree 0 after pruning)
        7. Return summary with counts

        Thresholds:
        - parent-child: 50% (remove edge if below)
        - sibling: 30% (flag if below, no graph change)
        - cross-branch: 70% (flag if above, no graph change)

        Returns:
            Dict with keys:
            - "edges_pruned": Count of edges removed from graph
            - "siblings_flagged": Count of sibling pairs below threshold
            - "cross_branch_candidates": Count of pairs above cross-branch threshold
            - "orphaned_nodes": Count of nodes with zero degree after pruning
        """
        # Initialize counters
        edges_pruned = 0
        siblings_flagged = 0
        cross_branch_candidates = 0

        # Step 1: Generate validation pairs
        pairs = self._generate_validation_pairs()

        # Step 1b: Pre-compute all similarities in parallel
        parallel_pairs = [
            {
                "term_a": p["term_x"],
                "term_b": p["term_y"],
                "description_a": p["desc_x"],
                "description_b": p["desc_y"],
            }
            for p in pairs
        ]
        self._precompute_similarities_parallel(parallel_pairs)

        # Step 2: Evaluate pairs and apply pruning rules
        # Thresholds (as percentages, 0-100 scale matching LLM output)
        # similarity_threshold is already on the 0-100 scale
        parent_child_threshold = self.similarity_threshold
        sibling_threshold = self.similarity_threshold * 0.6
        cross_branch_threshold = self.similarity_threshold * 1.4

        for pair in pairs:
            # Evaluate similarity via cached lookup
            similarity = self._get_similarity_cached(
                term_a=pair["term_x"],
                description_a=pair["desc_x"],
                term_b=pair["term_y"],
                description_b=pair["desc_y"],
            )

            category = pair["category"]

            if category == "parent-child":
                # Prune weak parent-child edges (similarity < 50%)
                if similarity < parent_child_threshold:
                    # Find and remove the edge by matching term pairs
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
                                f"Pruned weak parent-child edge {parent} ({parent_level}) "
                                f"→ {child} ({child_level}): similarity={similarity:.1f}%"
                            )
                            break

            elif category == "sibling":
                # Flag sibling dissimilarities (similarity < 30%)
                if similarity < sibling_threshold:
                    siblings_flagged += 1
                    logger.info(
                        f"Sibling dissimilarity flagged: {pair['term_x']} vs {pair['term_y']}: "
                        f"similarity={similarity:.1f}%"
                    )

            elif category == "cross-branch":
                # Flag cross-branch link candidates (similarity > 70%)
                if similarity > cross_branch_threshold:
                    cross_branch_candidates += 1
                    logger.info(
                        f"Cross-branch link candidate: {pair['term_x']} and {pair['term_y']}: "
                        f"similarity={similarity:.1f}%"
                    )

        # Step 3: Detect orphaned nodes (degree = 0 after pruning)
        orphaned_nodes = 0
        for node_id in self.ontology_graph.nodes():
            if self.ontology_graph.degree(node_id) == 0:
                node_attrs = self.ontology_graph.nodes[node_id]
                term = node_attrs.get("term", node_id)
                level = node_attrs.get("level", "unknown")
                orphaned_nodes += 1
                logger.warning(f"Orphaned node detected: {node_id} (term={term}, level={level})")

        # Step 4: Return summary
        summary = {
            "edges_pruned": edges_pruned,
            "siblings_flagged": siblings_flagged,
            "cross_branch_candidates": cross_branch_candidates,
            "orphaned_nodes": orphaned_nodes,
        }

        logger.info(f"Validation summary: {summary}")
        return summary

    def _sanitize_uri(self, term: str) -> URIRef:
        """Convert a term string to a valid RDF URI.

        Sanitizes term strings by:
        1. Replacing spaces with underscores
        2. Removing special characters (keep alphanumeric, underscores, hyphens)
        3. Prepending the base namespace

        Examples:
        - "Star Trek Officer" → "http://example.org/ontology/Star_Trek_Officer"
        - "Spock (TOS)" → "http://example.org/ontology/Spock_TOS"
        - "test_99-alpha!" → "http://example.org/ontology/test_99-alpha"

        Args:
            term: The term string to sanitize.

        Returns:
            rdflib.URIRef: A valid RDF URI in the base namespace.
        """
        # Step 1: Replace spaces with underscores
        sanitized = term.replace(" ", "_")

        # Step 2: Remove special characters (keep alphanumeric, underscores, hyphens)
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', sanitized)

        # Step 3: Ensure we have something left (fallback to 'unknown' if empty)
        if not sanitized:
            logger.warning(f"Term '{term}' sanitized to empty string; using 'unknown'")
            sanitized = "unknown"

        # Step 4: Prepend base namespace and return URIRef
        uri = self.base_namespace[sanitized]
        logger.debug(f"Sanitized term '{term}' → {uri}")
        return uri

    def _generate_candidates(self, node: str) -> List[Dict[str, str]]:
        """Generate new child terms for a given parent node using LLM prompts.

        This method is role-aware: it generates subclasses for class/subclass nodes,
        and instances for instance-parent nodes. The prompt includes context about
        existing children to avoid duplicating terms already in the graph.

        Algorithm:
        1. Read parent node attributes (term, description, level)
        2. Determine child level using level_schema
        3. If no child level exists (leaf node), return empty list
        4. Gather existing children to exclude from generation
        5. Build role-specific prompt with context
        6. Call LLM to generate candidates
        7. Parse JSON response defensively
        8. Return list of candidate dicts or empty list on parse error

        Args:
            node: The node ID (term string) to expand.

        Returns:
            List of candidate dicts with keys "term" and "description", or empty
            list if the node is at a leaf level or if LLM response parsing fails.
        """
        # Step 1: Read node attributes
        if node not in self.ontology_graph:
            logger.warning(f"Cannot generate candidates for unknown node: {node}")
            return []

        node_attrs = self.ontology_graph.nodes[node]
        parent_term = node_attrs.get("term", node)
        parent_description = node_attrs.get("description", "")
        parent_level = node_attrs.get("level")

        # Step 2: Determine child level from schema
        parent_level_def = None
        child_level_def = None
        parent_idx = None

        for idx, level in enumerate(self.level_schema):
            if level.name == parent_level:
                parent_level_def = level
                parent_idx = idx
                break

        if parent_level_def is None:
            logger.warning(f"Parent level '{parent_level}' not found in level_schema")
            return []

        # Step 3: Find child level
        if parent_idx is None or parent_idx + 1 >= len(self.level_schema):
            # No child level exists (leaf node)
            logger.info(f"Node '{node}' is at leaf level; no children can be generated")
            return []

        child_level_def = self.level_schema[parent_idx + 1]
        child_level_name = child_level_def.name

        # Step 4: Gather existing children to avoid duplication
        existing_children = list(self.ontology_graph.successors(node))
        existing_terms = []
        if existing_children:
            for child_node in existing_children:
                child_attrs = self.ontology_graph.nodes[child_node]
                existing_terms.append(child_attrs.get("term", child_node))

        # Step 5: Build role-aware prompt
        # The prompt engineering rationale:
        # - Include domain context to ground the LLM in the right namespace
        # - Specify the parent's level and role to make the relationship clear
        # - Enumerate existing children to prevent duplicate generation
        # - Explicitly request JSON array format with required keys
        # - Use clear role descriptions (subclasses vs instances) to disambiguate the task

        existing_terms_str = ", ".join(f'"{term}"' for term in existing_terms) if existing_terms else "none yet"

        # Determine generation role from child level name
        if child_level_name == "subclass":
            role_description = "subclasses"
        elif child_level_name == "instance":
            role_description = "instances"
        else:
            role_description = f"{child_level_name} terms"

        prompt = f"""Given the domain "{self.domain}", generate {self.candidates_per_iteration} new {role_description} for the following {parent_level}:

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

        logger.debug(f"Generating candidates for node '{node}' at level '{parent_level}'")
        logger.debug(f"Prompt:\n{prompt}")

        # Step 6: Call LLM
        try:
            response = self.agent.chat(
                instructions="You are an ontology engineer specialist. Generate accurate, specific taxonomies for the given domain based on the requested hierarchy.",
                input=prompt
            )
        except Exception as e:
            logger.error(f"LLM call failed for node '{node}': {e}")
            return []

        logger.debug(f"LLM response: {response}")

        # Step 7: Parse JSON defensively
        candidates = []
        try:
            # Try to extract JSON array from the response
            # Handle cases where LLM returns extra text before/after JSON
            response_text = response.strip()

            # Try direct parsing first
            try:
                candidates_raw = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON array if there's extra text
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    candidates_raw = json.loads(json_match.group(0))
                else:
                    raise json.JSONDecodeError("No JSON array found in response", response_text, 0)

            # Validate structure: must be a list of dicts with "term" and "description"
            if not isinstance(candidates_raw, list):
                logger.error(f"LLM response is not a list: {type(candidates_raw)}")
                return []

            for item in candidates_raw:
                if not isinstance(item, dict):
                    logger.warning(f"Candidate item is not a dict: {item}")
                    continue

                term = item.get("term", "").strip()
                description = item.get("description", "").strip()

                if not term:
                    logger.warning(f"Candidate missing 'term' key or empty value")
                    continue

                candidates.append({
                    "term": term,
                    "description": description
                })

            logger.info(f"Parsed {len(candidates)} candidates for node '{node}'")
            return candidates

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON for node '{node}': {e}")
            logger.error(f"Raw response: {response}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing candidates for node '{node}': {e}")
            return []

    def _validate_candidates(self, parent_node: str, candidates: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter candidates by similarity to their parent node.

        Each candidate is evaluated against the parent node using cached similarity calls.
        Only candidates with similarity >= self.similarity_threshold are accepted.

        Algorithm:
        1. Read parent node's term and description from graph attributes
        2. For each candidate:
           - Call _get_similarity_cached() between parent and candidate
           - Compare similarity to threshold (similarity_threshold, 0-100 scale)
           - Accept if >= threshold, reject and log if below
        3. Return list of accepted candidates

        Args:
            parent_node: The node ID (term string) of the parent.
            candidates: List of candidate dicts with "term" and "description" keys.

        Returns:
            List of candidate dicts that passed the similarity threshold.
        """
        if parent_node not in self.ontology_graph:
            logger.warning(f"Parent node '{parent_node}' not found in graph")
            return []

        parent_attrs = self.ontology_graph.nodes[parent_node]
        parent_term = parent_attrs.get("term", parent_node)
        parent_description = parent_attrs.get("description", "")

        accepted = []
        # similarity_threshold is already on the 0-100 scale
        threshold = self.similarity_threshold

        logger.info(
            f"Validating {len(candidates)} candidates for parent '{parent_term}' "
            f"(threshold={threshold:.1f}%)"
        )

        # Pre-compute all candidate similarities in parallel
        parallel_pairs = [
            {
                "term_a": parent_term,
                "term_b": c.get("term", ""),
                "description_a": parent_description,
                "description_b": c.get("description", ""),
            }
            for c in candidates
            if c.get("term", "")
        ]
        self._precompute_similarities_parallel(parallel_pairs)

        for candidate in candidates:
            candidate_term = candidate.get("term", "")
            candidate_desc = candidate.get("description", "")

            if not candidate_term:
                logger.warning("Candidate missing term; skipping")
                continue

            # Get cached similarity score
            similarity = self._get_similarity_cached(
                term_a=parent_term,
                description_a=parent_description,
                term_b=candidate_term,
                description_b=candidate_desc,
            )

            if similarity >= threshold:
                accepted.append(candidate)
                logger.info(
                    f"Accepted candidate '{candidate_term}' for parent '{parent_term}': "
                    f"similarity={similarity:.1f}%"
                )
            else:
                logger.info(
                    f"Rejected candidate '{candidate_term}' for parent '{parent_term}': "
                    f"similarity={similarity:.1f}% < {threshold:.1f}%"
                )

        logger.info(f"Validated candidates: {len(accepted)}/{len(candidates)} accepted")
        return accepted

    def _add_candidate_to_graph(self, parent: str, candidate: Dict[str, str]) -> None:
        """Add an accepted candidate to the ontology graph as a new node and edge.

        Inserts the candidate as a new node with the appropriate level (determined from
        the schema based on the parent's level), and adds an edge from parent to candidate
        with the correct relation label.

        If the candidate term already exists in the graph (duplicate), logs a warning and
        skips insertion.

        Algorithm:
        1. Read parent's level from graph attributes
        2. Find parent's OntologyLevel in level_schema
        3. Find the next (child) level in level_schema
        4. Extract candidate's term and description
        5. Check if term already exists in graph; skip if duplicate
        6. Add node with attributes: term, description, level, n_visits=0, total_reward=0.0
        7. Add edge (parent → candidate) with relation attribute

        Args:
            parent: The parent node ID (term string).
            candidate: Candidate dict with "term" and "description".

        Raises:
            ValueError: If parent node doesn't exist or has unknown level.
        """
        if parent not in self.ontology_graph:
            logger.warning(f"Parent node '{parent}' not found in graph; cannot add candidate")
            raise ValueError(f"Parent node '{parent}' not found in graph")

        parent_attrs = self.ontology_graph.nodes[parent]
        parent_level = parent_attrs.get("level")

        if parent_level is None:
            logger.warning(f"Parent node '{parent}' has no level attribute")
            raise ValueError(f"Parent node '{parent}' has no level attribute")

        # Find parent's level definition
        parent_level_idx = None
        for idx, level in enumerate(self.level_schema):
            if level.name == parent_level:
                parent_level_idx = idx
                break

        if parent_level_idx is None:
            logger.warning(f"Parent level '{parent_level}' not found in level_schema")
            raise ValueError(f"Parent level '{parent_level}' not found in level_schema")

        # Find child level
        if parent_level_idx + 1 >= len(self.level_schema):
            logger.warning(f"No child level exists for parent level '{parent_level}'")
            raise ValueError(f"No child level exists for parent level '{parent_level}'")

        child_level_def = self.level_schema[parent_level_idx + 1]
        child_level_name = child_level_def.name
        child_relation = child_level_def.relation_to_parent

        # Extract candidate information
        candidate_term = candidate.get("term", "").strip()
        candidate_desc = candidate.get("description", "").strip()

        if not candidate_term:
            logger.warning("Candidate term is empty; cannot add to graph")
            raise ValueError("Candidate term is empty")

        # Check for duplicate term
        if candidate_term in self.ontology_graph:
            logger.warning(
                f"Candidate '{candidate_term}' already exists in graph; skipping duplicate"
            )
            return

        # Add node to graph
        self.ontology_graph.add_node(
            candidate_term,
            term=candidate_term,
            description=candidate_desc,
            level=child_level_name,
            n_visits=0,
            total_reward=0.0,
        )

        # Add edge from parent to candidate
        self.ontology_graph.add_edge(parent, candidate_term, relation=child_relation)

        logger.info(
            f"Added node '{candidate_term}' (level={child_level_name}) as child of "
            f"'{parent}' with relation={child_relation}"
        )

    def _check_cross_branch_links_batch(self, candidates: List[Dict[str, str]]) -> None:
        """Batch cross-branch linking for multiple candidates using parallel pre-computation.

        Collects all (candidate, class-representative) similarity pairs across all
        candidates, pre-computes them in parallel via _precompute_similarities_parallel(),
        then applies the cross-branch threshold using only cached lookups.

        This replaces per-candidate sequential calls with a single parallel batch,
        reducing wall-clock time from O(candidates * classes) sequential LLM calls
        to O(candidates * classes / max_workers) parallel batches.

        Args:
            candidates: List of candidate dicts with "term" and "description" keys.
                        Each candidate must already exist in self.ontology_graph.
        """
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

        # Build a mapping of class → representative (computed once for all candidates)
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

        # For each candidate, find its ancestor class and collect similarity pairs
        parallel_pairs: List[Dict[str, str]] = []
        # Track which (candidate_term, class_node) pairs to evaluate after pre-computation
        evaluation_plan: List[Tuple[str, str, str]] = []  # (candidate_term, class_node, rep_term)

        for candidate in candidates:
            candidate_term = candidate.get("term", "")
            candidate_desc = candidate.get("description", "")

            if not candidate_term or candidate_term not in self.ontology_graph:
                continue

            # Walk up the graph to find the candidate's root-level ancestor
            ancestor_class = self._find_root_ancestor(candidate_term, root_level_name)

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

        # Pre-compute all cross-branch similarities in parallel
        self._precompute_similarities_parallel(parallel_pairs)

        # Apply cross-branch threshold using cached results
        cross_link_threshold = self.cross_link_threshold / 100.0

        for candidate_term, class_node, rep_term in evaluation_plan:
            cache_key = tuple(sorted([candidate_term, rep_term]))
            score = self.similarity_cache.get(cache_key, 0.0) / 100.0

            if score > cross_link_threshold:
                self.ontology_graph.add_edge(candidate_term, class_node, relation="type")
                logger.info(
                    f"Added cross-branch link: '{candidate_term}' → '{class_node}' "
                    f"(similarity={score:.3f} > threshold={cross_link_threshold:.3f})"
                )

    def _find_root_ancestor(self, node: str, root_level_name: str) -> Optional[str]:
        """Walk up the graph from a node to find its root-level ancestor.

        Args:
            node: The starting node ID.
            root_level_name: The level name of root nodes (e.g., "class").

        Returns:
            The root-level ancestor node ID, or None if not found.
        """
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

    def _check_cross_branch_links(self, candidate_term: str, candidate_desc: str) -> None:
        """Check if a single candidate should be typed under additional classes.

        Delegates to _check_cross_branch_links_batch for a single candidate.
        Kept for backward compatibility and tests that call this method directly.

        Args:
            candidate_term: The term string of the newly added candidate.
            candidate_desc: The description of the candidate.
        """
        self._check_cross_branch_links_batch([{"term": candidate_term, "description": candidate_desc}])

    def _discover_new_classes(self, num_classes: int = 2) -> List[str]:
        """Discover new top-level classes from the domain that are not yet in the ontology.

        Prompts the LLM to suggest new top-level classes for the domain, excluding
        classes that already exist. Accepted classes are added to the graph as new
        root-level nodes with n_visits=0, making them eligible for UCB1 expansion.

        Args:
            num_classes: Number of new classes to request from the LLM. Defaults to 2.

        Returns:
            List of newly added class term strings. May be shorter than num_classes
            if duplicates are found or if the LLM response is invalid.
        """
        root_level = self.level_schema[0]
        existing_classes = [
            self.ontology_graph.nodes[n].get("term", n)
            for n in self.ontology_graph.nodes()
            if self.ontology_graph.nodes[n].get("level") == root_level.name
        ]

        existing_str = ", ".join(f'"{c}"' for c in existing_classes) if existing_classes else "none yet"

        prompt = f"""Given the domain "{self.domain}", suggest {num_classes} NEW top-level {root_level.name}s that are NOT already in the ontology.

Existing {root_level.name}s: {existing_str}

Requirements:
- Each new {root_level.name} must be a distinct, broad category within "{self.domain}"
- Do NOT repeat any existing {root_level.name}s listed above
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
        except Exception as e:
            logger.error(f"Class discovery LLM call failed: {e}")
            return []

        # Parse response
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "taxonomy" in parsed:
                parsed = parsed["taxonomy"]
            if not isinstance(parsed, list):
                logger.warning(f"Class discovery response is not a list: {type(parsed)}")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Class discovery JSON parse error: {e}")
            return []

        # Add valid, non-duplicate classes to the graph
        added: List[str] = []
        for item in parsed:
            term = item.get("term", "").strip() if isinstance(item, dict) else ""
            desc = item.get("description", "").strip() if isinstance(item, dict) else ""

            if not term:
                continue
            if term in self.ontology_graph:
                logger.info(f"Class discovery: skipping duplicate '{term}'")
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
            logger.info(f"Discovered new {root_level.name}: '{term}'")

        return added

    def _select_node_ucb1(self) -> Optional[str]:
        """Select the next node to expand using the UCB1 (Upper Confidence Bound) algorithm.

        UCB1 is a multi-armed bandit strategy that balances exploration (trying less-visited nodes)
        and exploitation (expanding nodes that historically produce high-quality children).

        Algorithm:
        1. Identify all expandable nodes: nodes whose level has expandable=True in level_schema
        2. Prioritize unvisited nodes: if any arm has n_visits=0, return one uniformly at random
        3. Compute UCB1 score for each visited node:
           score_i = mean_reward_i + c * sqrt(ln(N) / n_visits_i)
           where c = self.exploration_constant, N = sum of all n_visits
        4. Return the node with the highest UCB1 score

        The mean_reward is computed as total_reward / n_visits (avoiding division by zero for
        unvisited nodes, which are handled in step 2).

        Returns:
            The node ID (term string) of the selected node to expand, or None if no
            expandable nodes remain (all nodes are leaf-level instances).

        Raises:
            ValueError: If expandable nodes exist but all have zero visits (this shouldn't
                happen in the normal flow, but guards against inconsistent state).
        """
        # Step 1: Collect all expandable nodes (skip retired ones)
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
                # Level not found in schema; skip this node
                logger.warning(f"Node {node_id} has unknown level '{node_level}'")
                continue

        # If no expandable nodes, return None
        if not expandable_nodes:
            logger.info("No expandable nodes remaining in the graph")
            return None

        # Step 2: Prioritize unvisited nodes
        # Return the first unvisited node (or pick randomly if multiple exist)
        unvisited = [
            node_id for node_id in expandable_nodes
            if self.ontology_graph.nodes[node_id].get("n_visits", 0) == 0
        ]
        if unvisited:
            selected = unvisited[0]
            logger.info(f"Selected unvisited expandable node: {selected}")
            return selected

        # Step 3: Compute total visits across all arms
        total_visits = sum(
            self.ontology_graph.nodes[node_id].get("n_visits", 0)
            for node_id in expandable_nodes
        )

        if total_visits == 0:
            # This shouldn't happen if we got past step 2, but guard against it
            logger.warning("All nodes have zero visits; selecting first expandable node")
            return expandable_nodes[0]

        # Step 4: Compute UCB1 score for each visited node and select the best
        best_node = None
        best_score = -float("inf")

        for node_id in expandable_nodes:
            n_visits = self.ontology_graph.nodes[node_id].get("n_visits", 0)
            total_reward = self.ontology_graph.nodes[node_id].get("total_reward", 0.0)

            if n_visits == 0:
                # Should have been caught in step 2; skip if any slipped through
                continue

            # Compute mean reward
            mean_reward = total_reward / n_visits

            # Compute UCB1 score
            ln_N = math.log(total_visits)
            exploration_term = self.exploration_constant * math.sqrt(ln_N / n_visits)
            ucb_score = mean_reward + exploration_term

            logger.debug(
                f"Node {node_id}: n_visits={n_visits}, mean_reward={mean_reward:.3f}, "
                f"exploration_term={exploration_term:.3f}, ucb_score={ucb_score:.3f}"
            )

            if ucb_score > best_score:
                best_score = ucb_score
                best_node = node_id

        if best_node is None:
            logger.warning("Could not select a node via UCB1 (this shouldn't happen)")
            return None

        logger.info(f"Selected node via UCB1: {best_node} (score={best_score:.3f})")
        return best_node

    def _update_bandit(
        self, node: str, reward: float, candidates_accepted: int = 0
    ) -> None:
        """Update the bandit reward tracking for a node after expansion.

        Increments the node's visit count and accumulates the reward (similarity score).
        These values are used by _select_node_ucb1() to balance exploration vs. exploitation.

        Also tracks consecutive zero-acceptance visits for retirement logic.
        When a node accumulates ``self.retirement_limit`` consecutive visits
        with zero accepted candidates, it is marked ``retired=True`` and
        excluded from future UCB1 selection.

        Args:
            node: The node ID (term string) to update.
            reward: The reward score (0-1) from this expansion iteration.
            candidates_accepted: Number of candidates accepted this iteration.

        Raises:
            ValueError: If the node does not exist in the graph.
        """
        if node not in self.ontology_graph:
            raise ValueError(f"Node '{node}' not found in ontology graph")

        attrs = self.ontology_graph.nodes[node]

        # Increment visit count
        new_visits = attrs.get("n_visits", 0) + 1
        attrs["n_visits"] = new_visits

        # Accumulate reward
        new_total_reward = attrs.get("total_reward", 0.0) + reward
        attrs["total_reward"] = new_total_reward

        # Track consecutive low-yield visits for retirement
        if candidates_accepted > 0:
            attrs["consecutive_low_yield"] = 0
        else:
            low_yield = attrs.get("consecutive_low_yield", 0) + 1
            attrs["consecutive_low_yield"] = low_yield
            if self.retirement_limit > 0 and low_yield >= self.retirement_limit:
                attrs["retired"] = True
                logger.info(
                    f"Node '{node}' retired after {low_yield} consecutive "
                    f"zero-acceptance visits"
                )

        logger.info(
            f"Updated bandit for node '{node}': n_visits={new_visits}, "
            f"total_reward={new_total_reward:.3f}, mean_reward={new_total_reward / new_visits:.3f}"
        )

    def expand_ontology(self) -> Dict[str, Any]:
        """Run one iteration of UCB1-guided ontology expansion.

        This method orchestrates a single expansion iteration of the bottom-up population phase:
        1. Select the next node to expand using UCB1 bandit algorithm
        2. Generate new candidate children for that node
        3. Validate candidates by similarity to their parent
        4. Add accepted candidates to the graph
        5. Check for cross-branch links (multi-class membership)
        6. Update the bandit's reward tracking
        7. Return iteration statistics

        The expansion loop in generate_ontology() calls this method repeatedly until
        termination conditions are met (max iterations, no expandable nodes, or plateau).

        Returns:
            Dict[str, Any] with keys:
            - "node": The node ID that was expanded (str), or None if no expandable nodes
            - "candidates_generated": Number of candidates returned by LLM (int)
            - "candidates_accepted": Number of candidates passing similarity threshold (int)
            - "reward": Mean similarity of accepted candidates (0-1 scale), or 0.0 if none accepted
        """
        # Step 1: Select node to expand
        selected_node = self._select_node_ucb1()

        if selected_node is None:
            logger.info("Expansion complete: no expandable nodes remaining")
            return {
                "node": None,
                "candidates_generated": 0,
                "candidates_accepted": 0,
                "reward": 0.0,
            }

        logger.info(f"Expanding node '{selected_node}'")

        # Step 2: Generate candidates
        candidates = self._generate_candidates(selected_node)
        candidates_generated = len(candidates)
        logger.info(f"Generated {candidates_generated} candidates for '{selected_node}'")

        # Step 3: Validate candidates
        accepted_candidates = self._validate_candidates(selected_node, candidates)
        candidates_accepted = len(accepted_candidates)
        logger.info(f"Validated {candidates_accepted} candidates for '{selected_node}'")

        # Step 4: Add accepted candidates to graph
        accepted_similarities = []
        added_candidates: List[Dict[str, str]] = []

        for candidate in accepted_candidates:
            candidate_term = candidate.get("term", "")
            candidate_desc = candidate.get("description", "")

            try:
                # Add to graph
                self._add_candidate_to_graph(selected_node, candidate)
                
                # Retrieve cached similarity (already computed by _validate_candidates)
                similarity = self._get_similarity_cached(
                    term_a=self.ontology_graph.nodes[selected_node].get("term", selected_node),
                    description_a=self.ontology_graph.nodes[selected_node].get("description", ""),
                    term_b=candidate_term,
                    description_b=candidate_desc,
                ) / 100.0  # Convert 0-100 scale to 0-1

                accepted_similarities.append(similarity)
                added_candidates.append(candidate)

            except Exception as e:
                logger.error(f"Error adding candidate '{candidate_term}' to graph: {e}")
                continue

        # Step 4b: Batch cross-branch linking for all added candidates in parallel
        self._check_cross_branch_links_batch(added_candidates)

        # Step 5: Compute reward (mean similarity of accepted candidates)
        if accepted_similarities:
            reward = sum(accepted_similarities) / len(accepted_similarities)
        else:
            reward = 0.0

        logger.info(f"Expansion reward for '{selected_node}': {reward:.3f}")

        # Step 6: Update bandit (with retirement tracking)
        self._update_bandit(selected_node, reward, candidates_accepted)

        # Step 7: Return iteration stats
        result = {
            "node": selected_node,
            "candidates_generated": candidates_generated,
            "candidates_accepted": candidates_accepted,
            "reward": reward,
        }

        logger.info(
            f"Expansion iteration complete: {candidates_accepted}/{candidates_generated} "
            f"candidates added to graph. Reward: {reward:.3f}"
        )

        return result

    def generate_ontology(self) -> Graph:
        """Run the full ontology generation pipeline with progress tracking.

        This is the main entry point for users. It orchestrates the complete
        pipeline:
        1. Seed generation: LLM-generated structured 3-level taxonomy
        2. Seed-to-DiGraph conversion
        3. Structural validation: Pairwise similarity pruning of weak edges
        4. Iterative expansion: UCB1-guided bottom-up population of ontology
        5. RDF serialization: Convert DiGraph to RDF/OWL triples
        6. Turtle output: Write serialized ontology to file

        The expansion loop uses early termination via two independent signals:

        - **Reward plateau**: When reward (from iterations that accepted ≥1 candidate)
          stays within delta < 0.01 for 3+ consecutive productive iterations, AND
          all non-instance nodes that existed before expansion have been visited.
        - **Growth stagnation**: When the graph node count has not increased for
          5+ consecutive iterations (the LLM cannot find new valid candidates).

        Progress is printed to stdout during execution. After completion, the full
        generation history is available via ``self.history`` (a GenerationHistory
        instance) for auditing, tabular analysis (``self.history.to_dataframe()``),
        and convergence plotting (``self.plot_convergence()``).

        Returns:
            rdflib.Graph: The RDF graph representing the final ontology.

        Raises:
            ValueError: If seed generation fails (LLM returns None or invalid structure).
        """
        import pathlib

        pipeline_start = time.monotonic()

        # Initialize history object for this run
        self.history = GenerationHistory(
            domain=self.domain,
            started_at=datetime.now(timezone.utc).isoformat(),
            config={
                "exploration_constant": self.exploration_constant,
                "max_iterations": self.max_iterations,
                "similarity_threshold": self.similarity_threshold,
                "confidence_threshold": self.confidence_threshold,
                "candidates_per_iteration": self.candidates_per_iteration,
                "cross_link_threshold": self.cross_link_threshold,
                "retirement_limit": self.retirement_limit,
                "level_schema": [level.name for level in self.level_schema],
            },
        )

        # ── Helper: print a progress banner ──────────────────────────
        def _print_phase(phase_num: int, name: str) -> None:
            """Print a formatted phase header to stdout."""
            print(f"\n{'━' * 60}")
            print(f"  Phase {phase_num}: {name}")
            print(f"{'━' * 60}")

        # ── Phase 1: Seed generation ─────────────────────────────────
        _print_phase(1, "Seed Generation")
        phase_start = time.monotonic()
        logger.info(f"Phase 1: Generating seed from domain '{self.domain}'")

        seed = self.generate_initial_terms()
        if seed is None:
            raise ValueError("Seed generation failed: LLM returned None or invalid structure")

        num_top_classes = len(seed["taxonomy"])
        phase_duration = time.monotonic() - phase_start
        self.history.phases.append(PhaseRecord(
            phase=1, name="Seed generation", duration_seconds=phase_duration,
            details={"top_level_classes": num_top_classes},
        ))
        logger.info(f"Seed generated: {num_top_classes} top-level classes")
        print(f"  ✓ Generated {num_top_classes} top-level classes ({phase_duration:.1f}s)")

        # ── Phase 2: Seed → DiGraph ──────────────────────────────────
        _print_phase(2, "Graph Construction")
        phase_start = time.monotonic()
        logger.info("Phase 2: Converting seed to ontology graph")

        self.create_seed_ontology()
        num_nodes = self.ontology_graph.number_of_nodes()
        num_edges = self.ontology_graph.number_of_edges()
        phase_duration = time.monotonic() - phase_start
        self.history.phases.append(PhaseRecord(
            phase=2, name="Graph construction", duration_seconds=phase_duration,
            details={"nodes": num_nodes, "edges": num_edges},
        ))
        logger.info(f"Ontology graph created: {num_nodes} nodes, {num_edges} edges")
        print(f"  ✓ Created graph: {num_nodes} nodes, {num_edges} edges ({phase_duration:.1f}s)")

        # ── Phase 3: Structural validation ────────────────────────────
        _print_phase(3, "Structural Validation")
        phase_start = time.monotonic()
        logger.info("Phase 3: Validating structure and pruning weak edges")

        validation_summary = self.validate_structure()
        phase_duration = time.monotonic() - phase_start
        self.history.phases.append(PhaseRecord(
            phase=3, name="Structural validation", duration_seconds=phase_duration,
            details=validation_summary,
        ))
        logger.info(
            f"Validation complete: {validation_summary['edges_pruned']} edges pruned, "
            f"{validation_summary['orphaned_nodes']} orphaned nodes"
        )
        print(
            f"  ✓ Pruned {validation_summary['edges_pruned']} edges, "
            f"{validation_summary['orphaned_nodes']} orphaned nodes ({phase_duration:.1f}s)"
        )

        # ── Phase 4: UCB1 expansion loop ─────────────────────────────
        _print_phase(4, "UCB1 Iterative Expansion")
        phase_start = time.monotonic()
        logger.info("Phase 4: Running expansion loop with UCB1 selection")

        # Print expansion table header
        print(f"\n  {'Iter':>4s}  {'Node':<20s}  {'Gen':>4s}  {'Acc':>4s}  "
              f"{'Rate':>6s}  {'Reward':>7s}  {'Nodes':>5s}  {'Edges':>5s}  {'Status'}")
        print(f"  {'─' * 4}  {'─' * 20}  {'─' * 4}  {'─' * 4}  "
              f"{'─' * 6}  {'─' * 7}  {'─' * 5}  {'─' * 5}  {'─' * 12}")

        # Productive reward history: only rewards from iterations that accepted
        # at least 1 candidate. Zero-acceptance iterations yield reward=0.0 which
        # is meaningless for plateau detection (it just means the LLM couldn't
        # find valid candidates, not that the ontology converged).
        productive_reward_history: List[float] = []
        plateau_count = 0
        stagnation_count = 0
        iteration_count = 0
        termination_reason = "max_iterations"
        prev_node_count = self.ontology_graph.number_of_nodes()

        # NOTE: convergence requires ALL current expandable nodes (classes
        # and subclasses) to have been visited — including nodes added during
        # expansion.  This is checked dynamically in the loop below.

        # Convergence thresholds
        PLATEAU_LIMIT = 3   # consecutive productive plateaus to trigger convergence
        STAGNATION_LIMIT = 5  # consecutive no-growth iterations to trigger convergence

        for iteration in range(self.max_iterations):
            iteration_count = iteration + 1
            logger.info(f"Expansion iteration {iteration_count}/{self.max_iterations}")

            # ── Periodic class discovery ──────────────────────────────
            # When enabled, ask the LLM for new top-level classes at a
            # fixed interval to broaden the ontology beyond the seed.
            if (
                self.class_discovery_interval > 0
                and iteration_count > 1
                and (iteration_count - 1) % self.class_discovery_interval == 0
            ):
                new_classes = self._discover_new_classes(num_classes=2)
                if new_classes:
                    print(f"  {'':>4s}  {'[discovery]':<20s}  {'':>4s}  {len(new_classes):>4d}  "
                          f"{'':>6s}  {'':>7s}  "
                          f"{self.ontology_graph.number_of_nodes():>5d}  "
                          f"{self.ontology_graph.number_of_edges():>5d}  "
                          f"new classes: {', '.join(new_classes)}")
                    # Reset stagnation since the graph just grew
                    stagnation_count = 0

            # Run one expansion iteration
            stats = self.expand_ontology()

            # Check if no expandable nodes remain
            if stats["node"] is None:
                termination_reason = "no_expandable_nodes"
                logger.info("No expandable nodes remaining; terminating expansion")

                # Record a terminal expansion record
                self.history.expansion_records.append(ExpansionRecord(
                    iteration=iteration_count,
                    node_expanded=None,
                    candidates_generated=0,
                    candidates_accepted=0,
                    reward=0.0,
                    cumulative_nodes=self.ontology_graph.number_of_nodes(),
                    cumulative_edges=self.ontology_graph.number_of_edges(),
                    acceptance_rate=0.0,
                    plateau_count=plateau_count,
                    stagnation_count=stagnation_count,
                    elapsed_seconds=time.monotonic() - pipeline_start,
                ))
                print(f"  {iteration_count:>4d}  {'—':<20s}  {'—':>4s}  {'—':>4s}  "
                      f"{'—':>6s}  {'—':>7s}  "
                      f"{self.ontology_graph.number_of_nodes():>5d}  "
                      f"{self.ontology_graph.number_of_edges():>5d}  "
                      f"no nodes left")
                break

            # Track reward and convergence signals
            current_reward = stats["reward"]
            gen_count = stats["candidates_generated"]
            acc_count = stats["candidates_accepted"]
            acc_rate = acc_count / gen_count if gen_count > 0 else 0.0
            cur_nodes = self.ontology_graph.number_of_nodes()
            cur_edges = self.ontology_graph.number_of_edges()

            # ── Plateau detection (productive iterations only) ────────
            # Only compare rewards when candidates were actually accepted.
            # Zero-acceptance iterations (reward=0.0) are skipped to avoid
            # artificial oscillation between 0.85 and 0.0.
            if acc_count > 0:
                if productive_reward_history:
                    delta = abs(current_reward - productive_reward_history[-1])
                    if delta < 0.01:
                        plateau_count += 1
                        logger.debug(
                            f"Plateau detected (delta={delta:.4f}); count={plateau_count}"
                        )
                    else:
                        plateau_count = 0
                        logger.debug(
                            f"Reward changed (delta={delta:.4f}); plateau reset"
                        )
                productive_reward_history.append(current_reward)
            # else: zero-acceptance iteration — plateau_count unchanged

            # ── Growth stagnation detection ───────────────────────────
            # If the graph didn't grow (no new nodes added), increment
            # the stagnation counter. Any growth resets it.
            if cur_nodes > prev_node_count:
                stagnation_count = 0
            else:
                stagnation_count += 1
                logger.debug(
                    f"Stagnation detected (nodes unchanged at {cur_nodes}); "
                    f"count={stagnation_count}"
                )
            prev_node_count = cur_nodes

            # Record expansion record
            self.history.expansion_records.append(ExpansionRecord(
                iteration=iteration_count,
                node_expanded=stats["node"],
                candidates_generated=gen_count,
                candidates_accepted=acc_count,
                reward=current_reward,
                cumulative_nodes=cur_nodes,
                cumulative_edges=cur_edges,
                acceptance_rate=acc_rate,
                plateau_count=plateau_count,
                stagnation_count=stagnation_count,
                elapsed_seconds=time.monotonic() - pipeline_start,
            ))

            # ── Check for retirement event ─────────────────────────
            expanded_node = stats["node"]
            if (
                expanded_node is not None
                and expanded_node in self.ontology_graph
                and self.ontology_graph.nodes[expanded_node].get("retired", False)
            ):
                retired_level = self.ontology_graph.nodes[expanded_node].get("level", "?")
                logger.info(f"Node '{expanded_node}' ({retired_level}) retired")

            # ── Build status label ────────────────────────────────────
            status_parts: List[str] = []
            if (
                expanded_node is not None
                and expanded_node in self.ontology_graph
                and self.ontology_graph.nodes[expanded_node].get("retired", False)
            ):
                status_parts.append("RETIRED")
            if plateau_count > 0:
                status_parts.append(f"plateau({plateau_count})")
            if stagnation_count > 0:
                status_parts.append(f"stagnant({stagnation_count})")
            status = " ".join(status_parts)

            # ── Check early termination conditions ────────────────────
            # Condition A: Reward plateau — reward from productive iterations
            # hasn't changed for PLATEAU_LIMIT consecutive productive iterations
            # AND all non-instance nodes from the ORIGINAL seed have been visited.
            if plateau_count >= PLATEAU_LIMIT:
                # Dynamically collect all current expandable nodes using
                # the level_schema, including nodes added during expansion.
                expandable_level_names = frozenset(
                    level.name for level in self.level_schema if level.expandable
                )
                current_expandable = [
                    n for n in self.ontology_graph.nodes()
                    if self.ontology_graph.nodes[n].get("level") in expandable_level_names
                    and not self.ontology_graph.nodes[n].get("retired", False)
                ]
                all_expandable_visited = all(
                    self.ontology_graph.nodes[n].get("n_visits", 0) >= 1
                    for n in current_expandable
                )

                if all_expandable_visited:
                    termination_reason = (
                        f"reward plateau for {plateau_count} productive iterations "
                        f"and all {len(current_expandable)} expandable nodes visited"
                    )
                    status = "CONVERGED (plateau)"
                    logger.info(f"Early termination: {termination_reason}")

            # Condition B: Growth stagnation — graph hasn't grown for
            # STAGNATION_LIMIT consecutive iterations. The LLM is unable
            # to produce new valid candidates regardless of which node
            # is selected.
            if stagnation_count >= STAGNATION_LIMIT and status != "CONVERGED (plateau)":
                termination_reason = (
                    f"graph stagnant for {stagnation_count} consecutive iterations "
                    f"(stuck at {cur_nodes} nodes)"
                )
                status = "CONVERGED (stagnant)"
                logger.info(f"Early termination: {termination_reason}")

            # Truncate node name for display
            node_display = (stats["node"][:18] + "..") if len(stats["node"]) > 20 else stats["node"]

            # Print iteration row
            print(f"  {iteration_count:>4d}  {node_display:<20s}  {gen_count:>4d}  {acc_count:>4d}  "
                  f"{acc_rate:>5.0%}   {current_reward:>6.3f}  {cur_nodes:>5d}  {cur_edges:>5d}  "
                  f"{status}")

            logger.info(
                f"Iteration {iteration_count}: "
                f"generated={gen_count}, accepted={acc_count}, reward={current_reward:.3f}"
            )

            # Break after printing if converged
            if "CONVERGED" in status:
                break

        phase_duration = time.monotonic() - phase_start
        self.history.phases.append(PhaseRecord(
            phase=4, name="UCB1 expansion", duration_seconds=phase_duration,
            details={
                "iterations": iteration_count,
                "final_plateau_count": plateau_count,
                "termination_reason": termination_reason,
            },
        ))
        print(f"\n  Expansion finished: {iteration_count} iterations in {phase_duration:.1f}s "
              f"({termination_reason})")
        logger.info(
            f"Expansion loop complete: {iteration_count} iterations, "
            f"graph now has {self.ontology_graph.number_of_nodes()} nodes"
        )

        # ── Phase 5: RDF serialization ────────────────────────────────
        _print_phase(5, "RDF Serialization")
        phase_start = time.monotonic()
        logger.info("Phase 5: Building RDF graph from DiGraph")

        self.build_ontology()
        turtle_output = self.serialize_ontology()

        phase_duration = time.monotonic() - phase_start
        num_triples = len(self.rdf)
        self.history.phases.append(PhaseRecord(
            phase=5, name="RDF serialization", duration_seconds=phase_duration,
            details={"triples": num_triples, "format": "turtle"},
        ))
        logger.info("Serializing to Turtle format")
        print(f"  ✓ Built {num_triples} RDF triples ({phase_duration:.1f}s)")

        # ── Phase 6: File output ──────────────────────────────────────
        _print_phase(6, "File Output")
        phase_start = time.monotonic()
        logger.info("Phase 6: Writing output to file")

        output_path = pathlib.Path("output") / "ontology.ttl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(turtle_output)

        phase_duration = time.monotonic() - phase_start
        self.history.phases.append(PhaseRecord(
            phase=6, name="File output", duration_seconds=phase_duration,
            details={"path": str(output_path.absolute())},
        ))
        logger.info(f"Ontology written to {output_path.absolute()}")
        print(f"  ✓ Written to {output_path.absolute()} ({phase_duration:.1f}s)")

        # ── Finalize history ──────────────────────────────────────────
        total_duration = time.monotonic() - pipeline_start
        self.history.completed_at = datetime.now(timezone.utc).isoformat()
        self.history.total_iterations = iteration_count
        self.history.final_nodes = self.ontology_graph.number_of_nodes()
        self.history.final_edges = self.ontology_graph.number_of_edges()
        self.history.final_triples = num_triples
        self.history.early_terminated = termination_reason != "max_iterations"
        self.history.termination_reason = termination_reason

        # Print final summary
        print(self.history.summary())
        logger.info(
            f"Pipeline complete: {self.history.final_nodes} nodes, {num_triples} RDF triples "
            f"in {total_duration:.1f}s"
        )

        return self.rdf

    def _calculate_similarity(self, pairs_table: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pairwise similarity for a table of term pairs.

        Args:
            pairs_table: DataFrame with term pairs to evaluate.

        Returns:
            The input DataFrame with similarity scores added.
        """
        return pairs_table

    def build_ontology(self) -> Graph:
        """
        Build an RDF ontology graph from the internal DiGraph representation.

        Converts the hierarchical ontology stored in self.ontology_graph to RDF triples
        following W3C RDF Schema conventions:
        - Nodes with is_rdf_class=True get rdf:type rdfs:Class
        - Hierarchical edges are mapped to rdfs:subClassOf or rdf:type predicates
          based on the child node's level definition (OntologyLevel.rdf_predicate)
        - All nodes get rdfs:label literals

        The method works with any level schema defined in self.level_schema,
        mapping edge relations to RDF predicates dynamically.

        Returns:
            An RDFLib Graph representing the ontology with hierarchical structure.
        """
        g = Graph()
        g.bind("ex", self.base_namespace)

        # Helper to resolve predicate strings to RDF namespace URIs
        def resolve_predicate(predicate_str: Optional[str]) -> Optional[URIRef]:
            """Convert a predicate string like 'rdfs:subClassOf' to actual RDF URIRef."""
            if not predicate_str:
                return None
            if predicate_str.startswith("rdfs:"):
                local_name = predicate_str.split(":", 1)[1]
                return getattr(RDFS, local_name)
            elif predicate_str.startswith("rdf:"):
                local_name = predicate_str.split(":", 1)[1]
                return getattr(RDF, local_name)
            else:
                # Custom namespace predicate — use as-is (could extend this for other namespaces)
                logger.warning(f"Unknown predicate namespace in '{predicate_str}'; using as custom URI")
                return URIRef(predicate_str)

        # Add all nodes to the graph
        logger.debug(f"Building RDF ontology from DiGraph with {self.ontology_graph.number_of_nodes()} nodes")
        for node_id in self.ontology_graph.nodes():
            node_attrs = self.ontology_graph.nodes[node_id]
            term = node_attrs.get("term", node_id)
            level_name = node_attrs.get("level")

            # Create sanitized URI for this node
            node_uri = self._sanitize_uri(term)

            # Add rdfs:label for all nodes
            g.add((node_uri, RDFS.label, Literal(term)))

            # Add rdf:type rdfs:Class for class nodes
            if level_name:
                try:
                    level = self._get_level(level_name)
                    if level.is_rdf_class:
                        g.add((node_uri, RDF.type, RDFS.Class))
                        logger.debug(f"Added rdf:type rdfs:Class for {term} (level: {level_name})")
                except ValueError:
                    logger.warning(f"Level '{level_name}' not found in schema; skipping rdf:type rdfs:Class")

        # Add all edges to the graph
        logger.debug(f"Building RDF edges from DiGraph with {self.ontology_graph.number_of_edges()} edges")
        for parent_id, child_id in self.ontology_graph.edges():
            edge_attrs = self.ontology_graph.edges[parent_id, child_id]
            relation = edge_attrs.get("relation")
            child_attrs = self.ontology_graph.nodes[child_id]
            child_level_name = child_attrs.get("level")

            parent_uri = self._sanitize_uri(self.ontology_graph.nodes[parent_id].get("term", parent_id))
            child_uri = self._sanitize_uri(self.ontology_graph.nodes[child_id].get("term", child_id))

            # Map edge relation to RDF predicate using child level's rdf_predicate
            predicate = None
            if child_level_name:
                try:
                    child_level = self._get_level(child_level_name)
                    predicate = resolve_predicate(child_level.rdf_predicate)
                except ValueError:
                    logger.warning(f"Could not resolve predicate for child level '{child_level_name}'")

            # If no predicate found, fall back to relation string (if available)
            if not predicate and relation:
                logger.debug(f"No RDF predicate for relation '{relation}'; attempting fallback")
                # Map common relation names to RDF predicates
                relation_to_predicate = {
                    "subClassOf": "rdfs:subClassOf",
                    "type": "rdf:type",
                }
                predicate_str = relation_to_predicate.get(relation, f"rdfs:{relation}")
                predicate = resolve_predicate(predicate_str)

            if predicate:
                g.add((child_uri, predicate, parent_uri))
                logger.debug(f"Added edge: {child_uri} {predicate} {parent_uri}")
            else:
                logger.warning(
                    f"Could not determine RDF predicate for edge {parent_id} → {child_id}; skipping"
                )

        self.rdf = g
        logger.info(f"RDF ontology built successfully: {len(g)} triples")
        return self.rdf

    def serialize_ontology(self, format: str = "turtle") -> str:
        """
        Serialize the ontology graph into the specified format.

        Supports multiple serialization formats for RDF output.

        Args:
            format: The serialization format. Must be one of "turtle", "xml", "json-ld"
                (default: "turtle").

        Returns:
            The serialized ontology as a string.

        Raises:
            ValueError: If format is not one of the supported formats.
        """
        # Validate format parameter
        allowed_formats = {"turtle", "xml", "json-ld"}
        if format not in allowed_formats:
            raise ValueError(
                f"Unsupported format '{format}'. Must be one of: {', '.join(sorted(allowed_formats))}"
            )

        self.turtle = self.rdf.serialize(format=format)
        return self.turtle

    def visualize(self) -> None:
        """
        Visualize the RDF ontology graph as a PNG image.

        Renders the hierarchical RDF graph (post-build_ontology) with relationship
        types shown as edge labels (subClassOf, type, etc). Displays inline in a
        notebook using pydotplus.

        Note: This renders the RDF graph, not the internal DiGraph used during
        expansion. For debugging the internal structure, use visualize_graph().
        """
        stream = io.StringIO()
        # opts dict allows rdf2dot option control; empty dict uses defaults
        rdf2dot(self.rdf, stream, opts={})
        graph = pydotplus.graph_from_dot_data(stream.getvalue())
        png_data = graph.create_png()
        display(Image(png_data))

    def visualize_graph(self) -> None:
        """
        Visualize the internal ontology graph with nodes colored by level.

        Renders the internal networkx.DiGraph (self.ontology_graph) with nodes
        colored by their level (class=blue, subclass=green, instance=orange).
        Edge labels show the relationship type (subClassOf, type, etc).

        This is a debugging tool for inspecting the hierarchical structure before
        RDF serialization. For the RDF visualization, use visualize().

        Returns:
            None. Displays the graph inline in a notebook using matplotlib.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        # Color map for ontology levels
        color_map = {
            "class": "#4A90D9",      # Blue
            "subclass": "#50C878",   # Green
            "instance": "#FF8C42"    # Orange
        }
        
        # Compute node colors based on level
        node_colors = [
            color_map.get(self.ontology_graph.nodes[n].get("level", "instance"), "#FF8C42")
            for n in self.ontology_graph.nodes
        ]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Compute layout
        pos = nx.spring_layout(self.ontology_graph, k=2, seed=42, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.ontology_graph,
            pos,
            node_color=node_colors,
            node_size=2000,
            ax=ax
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            self.ontology_graph,
            pos,
            labels={n: n for n in self.ontology_graph.nodes},
            font_size=9,
            font_color="black",
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.ontology_graph,
            pos,
            edge_color="gray",
            arrows=True,
            arrowsize=15,
            ax=ax
        )
        
        # Draw edge labels
        edge_labels = {
            (u, v): d.get("relation", "")
            for u, v, d in self.ontology_graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            self.ontology_graph,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            ax=ax
        )
        
        # Add legend
        legend_patches = [
            mpatches.Patch(color="#4A90D9", label="Class"),
            mpatches.Patch(color="#50C878", label="Subclass"),
            mpatches.Patch(color="#FF8C42", label="Instance")
        ]
        ax.legend(handles=legend_patches, loc="upper left", fontsize=10)
        
        # Set title and labels
        ax.set_title(f"Ontology Graph: {self.domain}", fontsize=14, fontweight="bold")
        ax.axis("off")
        
        # Tight layout and display
        plt.tight_layout()
        plt.show()

    def plot_convergence(self) -> None:
        """Plot epoch-style convergence charts from the generation history.

        Produces a multi-panel matplotlib figure similar to neural network training
        dashboards, showing how key metrics evolved across expansion iterations:

        1. **Reward** — per-iteration mean similarity reward with running average
        2. **Acceptance Rate** — fraction of generated candidates accepted
        3. **Graph Growth** — cumulative node and edge counts
        4. **Plateau Indicator** — consecutive plateau counter over time
        5. **Visit Distribution** — histogram of n_visits across expandable nodes
        6. **Top-10 Visited Nodes** — horizontal bar chart of most-visited nodes

        Requires that ``generate_ontology()`` has been run (populates ``self.history``).

        Raises:
            RuntimeError: If no generation history is available.
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        if self.history is None or not self.history.expansion_records:
            raise RuntimeError(
                "No generation history available. Run generate_ontology() first."
            )

        df = self.history.to_dataframe()

        # Filter out terminal records with no node expanded (keep for completeness)
        plot_df = df[df["node_expanded"].notna()].copy()
        if plot_df.empty:
            print("No expansion iterations to plot.")
            return

        iterations = plot_df["iteration"]

        # Compute running average of reward (window=3, or fewer if not enough data)
        window = min(3, len(plot_df))
        plot_df["reward_ma"] = plot_df["reward"].rolling(window=window, min_periods=1).mean()

        # Create figure with 6 subplots (3 rows × 2 cols)
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        fig.suptitle(
            f"Ontology Generation Convergence — {self.domain}",
            fontsize=16, fontweight="bold", y=0.98,
        )

        # ── Panel 1: Reward curve ─────────────────────────────────
        ax1 = axes[0, 0]
        ax1.plot(iterations, plot_df["reward"], "o-", color="#4A90D9",
                 alpha=0.5, markersize=5, label="Per-iteration reward")
        ax1.plot(iterations, plot_df["reward_ma"], "-", color="#2C5F8A",
                 linewidth=2.5, label=f"Moving avg (w={window})")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Reward (mean similarity)")
        ax1.set_title("Reward per Iteration")
        ax1.legend(loc="lower right", fontsize=9)
        ax1.set_ylim(bottom=0)
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax1.grid(True, alpha=0.3)

        # ── Panel 2: Acceptance rate ──────────────────────────────
        ax2 = axes[0, 1]
        ax2.bar(iterations, plot_df["acceptance_rate"], color="#50C878", alpha=0.7,
                edgecolor="#3A9D5C", label="Acceptance rate")
        # Overlay trend line
        acc_ma = plot_df["acceptance_rate"].rolling(window=window, min_periods=1).mean()
        ax2.plot(iterations, acc_ma, "-", color="#2D7A47", linewidth=2,
                 label=f"Moving avg (w={window})")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Acceptance Rate")
        ax2.set_title("Candidate Acceptance Rate")
        ax2.set_ylim(0, 1.05)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax2.legend(loc="lower right", fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ── Panel 3: Graph growth ─────────────────────────────────
        ax3 = axes[1, 0]
        ax3.plot(iterations, plot_df["cumulative_nodes"], "s-", color="#FF8C42",
                 linewidth=2, markersize=5, label="Nodes")
        ax3.plot(iterations, plot_df["cumulative_edges"], "^-", color="#D9534F",
                 linewidth=2, markersize=5, label="Edges")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Count")
        ax3.set_title("Cumulative Graph Growth")
        ax3.legend(loc="upper left", fontsize=9)
        ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax3.grid(True, alpha=0.3)

        # ── Panel 4: Plateau indicator ────────────────────────────
        ax4 = axes[1, 1]
        colors = ["#D9534F" if p >= 3 else "#FFB347" if p > 0 else "#50C878"
                  for p in plot_df["plateau_count"]]
        ax4.bar(iterations, plot_df["plateau_count"], color=colors, alpha=0.8,
                edgecolor="gray", linewidth=0.5)
        ax4.axhline(y=3, color="#D9534F", linestyle="--", linewidth=1.5,
                    alpha=0.7, label="Early-stop threshold")
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Consecutive Plateaus")
        ax4.set_title("Plateau Counter (convergence signal)")
        ax4.legend(loc="upper left", fontsize=9)
        ax4.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax4.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax4.grid(True, alpha=0.3)

        # ── Panel 5: Visit distribution histogram ─────────────
        ax5 = axes[2, 0]
        expandable_level_names = frozenset(
            level.name for level in self.level_schema if level.expandable
        )
        visit_counts = [
            self.ontology_graph.nodes[n].get("n_visits", 0)
            for n in self.ontology_graph.nodes()
            if self.ontology_graph.nodes[n].get("level") in expandable_level_names
        ]
        if visit_counts:
            max_visits = max(visit_counts)
            bins = range(0, max_visits + 2)
            ax5.hist(visit_counts, bins=bins, color="#7B68EE", alpha=0.8,
                     edgecolor="#5B4ACE", align="left")
        ax5.set_xlabel("Number of Visits")
        ax5.set_ylabel("Number of Nodes")
        ax5.set_title("Visit Distribution (expandable nodes)")
        ax5.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax5.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax5.grid(True, alpha=0.3)

        # ── Panel 6: Top-10 most visited nodes ────────────────────
        ax6 = axes[2, 1]
        node_visits = [
            (n, self.ontology_graph.nodes[n].get("n_visits", 0))
            for n in self.ontology_graph.nodes()
            if self.ontology_graph.nodes[n].get("n_visits", 0) > 0
        ]
        node_visits.sort(key=lambda x: x[1], reverse=True)
        top_10 = node_visits[:10]
        if top_10:
            names = [nv[0][:20] for nv in reversed(top_10)]
            visits = [nv[1] for nv in reversed(top_10)]
            level_colors = {
                self.level_schema[0].name: "#4A90D9",
            }
            if len(self.level_schema) > 1:
                level_colors[self.level_schema[1].name] = "#50C878"
            if len(self.level_schema) > 2:
                level_colors[self.level_schema[2].name] = "#FF8C42"
            bar_colors = [
                level_colors.get(
                    self.ontology_graph.nodes[nv[0]].get("level", ""), "#999999"
                )
                for nv in reversed(top_10)
            ]
            bars = ax6.barh(names, visits, color=bar_colors, alpha=0.85,
                            edgecolor="gray", linewidth=0.5)
            for bar, v in zip(bars, visits):
                ax6.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                         str(v), va="center", fontsize=9)
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor=color, label=level_name)
                for level_name, color in level_colors.items()
            ]
            ax6.legend(handles=legend_handles, loc="lower right", fontsize=9)
        else:
            ax6.text(0.5, 0.5, "No visited nodes", ha="center", va="center",
                     transform=ax6.transAxes, fontsize=12, color="gray")
        ax6.set_xlabel("Number of Visits")
        ax6.set_title("Top-10 Most Visited Nodes")
        ax6.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax6.grid(True, alpha=0.3, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def visualize_interactive(self, output_path: str = "ontology.html") -> str:
        """
        Render the RDF ontology as a full-page interactive HTML graph using pyvis.

        Only **structural** RDF relationships are visualised (``rdfs:subClassOf``
        and ``rdf:type`` between domain resources).  Metadata triples are folded
        into each node's tooltip instead of being drawn as separate edges/nodes:

        - ``rdfs:label`` → used as the node's display name.
        - ``rdf:type rdfs:Class`` → noted in the tooltip, not drawn as an edge
          to a meta ``rdfs:Class`` node.

        The domain itself is rendered as the root node (pink) and all top-level
        classes connect to it via ``rdfs:subClassOf``.

        Node colours:
        - **Pink** — domain root
        - **Blue** — classes / subclasses (``rdf:type rdfs:Class``)
        - **Orange** — instances

        Edge colours:
        - **Blue** — ``rdfs:subClassOf``
        - **Green** — ``rdf:type`` (instance → class)

        Args:
            output_path: File path for the generated HTML.  Defaults to
                ``"ontology.html"`` in the current working directory.

        Returns:
            The absolute path to the generated HTML file.

        Raises:
            ImportError: If ``pyvis`` is not installed.
            RuntimeError: If the RDF graph is empty.  Call ``build_ontology()`` first.
        """
        try:
            from pyvis.network import Network
        except ImportError as exc:
            raise ImportError(
                "pyvis is required for interactive visualization. "
                "Install it with: pip install pyvis"
            ) from exc

        if self.rdf is None or len(self.rdf) == 0:
            raise RuntimeError(
                "RDF graph is empty. Run build_ontology() first."
            )

        # Full-page interactive directed graph
        net = Network(
            notebook=True,
            cdn_resources="in_line",
            height="100vh",
            width="100%",
            directed=True,
        )

        # ── Colour palette ───────────────────────────────────────
        DOMAIN_ROOT_COLOR = "#E91E63"   # Pink — domain root
        CLASS_COLOR = "#4A90D9"         # Blue — class / subclass
        INSTANCE_COLOR = "#FF8C42"      # Orange — instance
        EDGE_SUBCLASS = "#2196F3"       # Blue
        EDGE_TYPE = "#4CAF50"           # Green

        # ── Build look-ups from the RDF graph ────────────────────
        # Which resources carry rdf:type rdfs:Class?
        class_resources: set = set()
        for s, _p, _o in self.rdf.triples((None, RDF.type, RDFS.Class)):
            class_resources.add(str(s))

        # Collect rdfs:label values (used as node display names)
        labels: Dict[str, str] = {}
        for s, _p, o in self.rdf.triples((None, RDFS.label, None)):
            labels[str(s)] = str(o)

        # Enrich tooltips with internal DiGraph metadata
        level_map: Dict[str, str] = {}
        desc_map: Dict[str, str] = {}
        visit_map: Dict[str, int] = {}
        reward_map: Dict[str, float] = {}
        for node_id, data in self.ontology_graph.nodes(data=True):
            uri = str(self._sanitize_uri(data.get("term", node_id)))
            level_map[uri] = data.get("level", "")
            desc_map[uri] = data.get("description", "")
            visit_map[uri] = data.get("n_visits", 0)
            reward_map[uri] = data.get("total_reward", 0.0)

        # ── Identify top-level classes (no rdfs:subClassOf subject) ──
        subclass_subjects = {
            str(s)
            for s, _p, _o in self.rdf.triples((None, RDFS.subClassOf, None))
        }
        top_level_classes = class_resources - subclass_subjects

        # Synthetic domain root node
        domain_uri = str(self._sanitize_uri(self.domain))
        domain_label = self.domain

        # ── Helper: compact URI display ──────────────────────────
        ns_prefix = str(self.base_namespace)

        def short_uri(uri_str: str) -> str:
            if uri_str.startswith(ns_prefix):
                return uri_str[len(ns_prefix):]
            return uri_str

        # ── Collect nodes ────────────────────────────────────────
        all_nodes: Dict[str, dict] = {}

        def ensure_node(uri_str: str) -> None:
            """Register a resource node if not already tracked."""
            if uri_str in all_nodes:
                return

            label = labels.get(uri_str, short_uri(uri_str))
            level = level_map.get(uri_str, "")
            desc = desc_map.get(uri_str, "")
            visits = visit_map.get(uri_str, 0)
            reward = reward_map.get(uri_str, 0.0)
            is_class = uri_str in class_resources

            # Colour & size by role
            if uri_str == domain_uri:
                color = DOMAIN_ROOT_COLOR
                size = 35
            elif is_class:
                color = CLASS_COLOR
                size = 22
            else:
                color = INSTANCE_COLOR
                size = 16

            title_parts = [
                f"<b>{label}</b>",
                f"URI: {uri_str}",
                f"rdf:type: rdfs:Class" if is_class else "rdf:type: instance",
                f"Level: {level}" if level else None,
                f"Description: {desc}" if desc else None,
                f"UCB1 visits: {visits}, reward: {reward:.2f}" if visits else None,
            ]
            title = "<br>".join(p for p in title_parts if p)

            all_nodes[uri_str] = {
                "label": label,
                "color": color,
                "size": size,
                "title": title,
                "shape": "dot",
            }

        # Domain root (always present)
        all_nodes[domain_uri] = {
            "label": domain_label,
            "color": DOMAIN_ROOT_COLOR,
            "size": 40,
            "title": (
                f"<b>{domain_label}</b><br>"
                f"URI: {domain_uri}<br>"
                f"Domain root class"
            ),
            "shape": "dot",
        }

        # ── Walk RDF triples — keep only structural edges ────────
        edges: list = []
        # URI of the rdfs:Class meta-node — excluded from the graph entirely
        rdfs_class_uri = str(RDFS.Class)

        for s, p, o in self.rdf:
            # Skip literal triples (rdfs:label) — already folded into node label
            if isinstance(o, Literal):
                continue

            # Skip any triple that involves rdfs:Class as subject or object
            # (rdf:type rdfs:Class is shown in the tooltip, not as an edge)
            if str(s) == rdfs_class_uri or str(o) == rdfs_class_uri:
                continue

            s_str = str(s)
            o_str = str(o)

            if p == RDFS.subClassOf:
                pred_label = "rdfs:subClassOf"
                edge_color = EDGE_SUBCLASS
            elif p == RDF.type:
                # Instance → class typing edge
                pred_label = "rdf:type"
                edge_color = EDGE_TYPE
            else:
                # Other predicates — include but style generically
                pred_label = short_uri(str(p))
                edge_color = "#999999"

            ensure_node(s_str)
            ensure_node(o_str)
            edges.append((s_str, o_str, pred_label, edge_color))

        # ── Connect top-level classes to domain root ─────────────
        for cls_uri in top_level_classes:
            ensure_node(cls_uri)
            edges.append((cls_uri, domain_uri, "rdfs:subClassOf", EDGE_SUBCLASS))

        # ── Populate pyvis network ───────────────────────────────
        for node_id, props in all_nodes.items():
            net.add_node(
                node_id,
                label=props["label"],
                title=props["title"],
                color=props["color"],
                size=props["size"],
                shape=props["shape"],
            )

        for src, dst, pred_label, edge_color in edges:
            net.add_edge(
                src,
                dst,
                label=pred_label,
                title=pred_label,
                arrows="to",
                color=edge_color,
            )

        # ── Physics / layout ─────────────────────────────────────
        net.repulsion(
            node_distance=250,
            central_gravity=0.15,
            spring_length=180,
            spring_strength=0.04,
            damping=0.09,
        )

        # ── Render ───────────────────────────────────────────────
        net.show(output_path)
        abs_path = os.path.abspath(output_path)
        logger.info(f"Interactive RDF ontology graph saved to {abs_path}")
        return abs_path
