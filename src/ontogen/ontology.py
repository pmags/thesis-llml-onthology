"""
Ontology construction, expansion, and RDF serialization.

This module contains the Ontology class which orchestrates the full
ontology generation pipeline: seed generation, iterative expansion,
graph construction, and RDF serialization.
"""

import io
import itertools
import json
import logging
import math
import re
from dataclasses import dataclass, field
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
    ) -> None:
        # Configuration
        self.domain = domain
        self.agent = agent
        self.seed = seed
        self.exploration_constant = exploration_constant
        self.max_iterations = max_iterations
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.candidates_per_iteration = candidates_per_iteration
        self.level_schema = level_schema or DEFAULT_LEVEL_SCHEMA
        self.cross_link_threshold = cross_link_threshold

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

        # Step 3: Cache miss — choose LLM method
        if description_a is not None or description_b is not None:
            logger.info(f"Cache miss: evaluating similarity({term_a}, {term_b}) with descriptions")
            response = self.agent.get_similarity_with_descriptions(
                term_a=term_a,
                description_a=description_a,
                term_b=term_b,
                description_b=description_b,
            )
        else:
            logger.info(f"Cache miss: evaluating similarity({term_a}, {term_b})")
            response = self.agent.get_similarity(term_a, term_b)

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

Generate exactly {num_classes} top-level {self.level_schema[0].name.lower()}s. 
For each {self.level_schema[0].name.lower()}, include 2-4 {self.level_schema[1].name.lower()}s.
For each {self.level_schema[1].name.lower()}, include 2-3 {self.level_schema[2].name.lower()}s (if applicable).

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

    def _build_example_schema(self) -> str:
        """Build a JSON schema example string from self.level_schema.
        
        Dynamically generates a nested example structure that matches the
        expected format for the LLM to return.
        
        Returns:
            A formatted JSON string showing the expected structure.
        """
        # Build the innermost (deepest) level first
        example_lines = []
        example_lines.append("{")
        example_lines.append(f'  "domain": "{self.domain}",')
        example_lines.append(f'  "taxonomy": [')

        # Build nested structure for the first (top-level) category
        root_level = self.level_schema[0]
        example_lines.append(f'    {{')
        example_lines.append(f'      "{root_level.seed_key}": "Example {root_level.name.capitalize()}",')
        example_lines.append(f'      "description": "Brief description",')

        # Now nest the children levels
        indent = "      "
        for level_idx in range(1, len(self.level_schema)):
            level = self.level_schema[level_idx]
            parent_level = self.level_schema[level_idx - 1]
            
            if level.children_key is not None:
                example_lines.append(f'{indent}"{parent_level.children_key}": [')
                example_lines.append(f'{indent}  {{')
                indent += "  "
                example_lines.append(f'{indent}"{level.seed_key}": "Example {level.name.capitalize()}",')
                example_lines.append(f'{indent}"description": "Brief description"')
                
                # Check if this level has children
                if level.children_key is not None:
                    example_lines.append(f'{indent}"{level.children_key}": [')
                    example_lines.append(f'{indent}  {{"term": "Example instance", "description": "..."}},')
                    example_lines.append(f'{indent}  {{"term": "Another instance", "description": "..."}}')
                    example_lines.append(f'{indent}]')
            else:
                # Leaf level (no children_key)
                example_lines.append(f'{indent}"{level.seed_key}": "Example {level.name.capitalize()}",')
                example_lines.append(f'{indent}"description": "Brief description"')
        
        # Close all brackets
        example_lines.append(f'{indent}')
        example_lines.append(f'{indent[:-4]}}},')  # Remove 2 spaces from indent
        example_lines.append(f'    ]')
        indent = indent[:-4]  # Remove "  "
        if indent:
            example_lines.append(f'{indent}}}')
        example_lines.append('  ]')
        example_lines.append('}')

        return "\n".join(example_lines)

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
        # Identify all top-level class nodes (nodes with level == "class")
        class_nodes = [
            node for node in self.ontology_graph.nodes()
            if self.ontology_graph.nodes[node]["level"] == "class"
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

        # Step 2: Evaluate pairs and apply pruning rules
        # Thresholds (as percentages, 0-100 scale matching LLM output)
        parent_child_threshold = 50.0
        sibling_threshold = 30.0
        cross_branch_threshold = 70.0

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
            response = self.agent.chat(prompt)
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
                import re
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
           - Compare similarity to threshold (similarity_threshold * 100)
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
        threshold = self.similarity_threshold * 100.0

        logger.info(
            f"Validating {len(candidates)} candidates for parent '{parent_term}' "
            f"(threshold={threshold:.1f}%)"
        )

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

    def _check_cross_branch_links(self, candidate_term: str, candidate_desc: str) -> None:
        """Check if a candidate should be typed under additional classes (cross-branch linking).

        Cross-branch linking enables rich ontologies where an instance can belong to multiple
        conceptual classes. For example, "Spock" might be both a "Vulcan" (species) and a
        "StarfleetOfficer" (role).

        Algorithm:
        1. Get all top-level class nodes (nodes with level == "class")
        2. Determine candidate's ancestor class (walk up the graph to find the top-level class)
        3. For each top-level class NOT in the candidate's ancestry:
           - Pick a representative node (the class itself or one of its children)
           - Compute similarity between candidate and representative
           - If similarity > self.cross_link_threshold: add rdf:type edge and log
        4. Leave the original parent→child edge intact

        Args:
            candidate_term: The term string of the newly added candidate.
            candidate_desc: The description of the candidate.
        """
        # Verify candidate exists in graph
        if candidate_term not in self.ontology_graph:
            logger.warning(f"Candidate '{candidate_term}' not found in graph for cross-branch check")
            return

        candidate_attrs = self.ontology_graph.nodes[candidate_term]

        # Step 1: Get all top-level class nodes
        class_nodes = [
            node_id for node_id in self.ontology_graph.nodes()
            if self.ontology_graph.nodes[node_id].get("level") == "class"
        ]

        if len(class_nodes) == 0:
            logger.debug("No class nodes found for cross-branch linking")
            return

        # Step 2: Determine candidate's ancestor class (walk up the graph)
        candidate_ancestor_class = None
        current = candidate_term
        visited = set()

        while current is not None and current not in visited:
            visited.add(current)
            current_level = self.ontology_graph.nodes[current].get("level")

            if current_level == "class":
                candidate_ancestor_class = current
                break

            # Get parent (predecessor in graph)
            predecessors = list(self.ontology_graph.predecessors(current))
            if predecessors:
                current = predecessors[0]  # Follow first parent up the hierarchy
            else:
                current = None

        logger.debug(
            f"Candidate '{candidate_term}' ancestor class: {candidate_ancestor_class or 'not found'}"
        )

        # Step 3: For each class not in the candidate's ancestry, check for cross-branch link
        cross_link_threshold = self.cross_link_threshold / 100.0  # Convert to 0-1 scale for comparison

        for class_node in class_nodes:
            # Skip if this is the candidate's own ancestor class
            if class_node == candidate_ancestor_class:
                logger.debug(f"Skipping class '{class_node}' (candidate's ancestor)")
                continue

            # Pick a representative node for this class (prefer a child if available)
            representative = class_node
            children = list(self.ontology_graph.successors(class_node))
            if children:
                representative = children[0]  # Use first child as representative

            representative_attrs = self.ontology_graph.nodes[representative]
            representative_term = representative_attrs.get("term", representative)
            representative_desc = representative_attrs.get("description", "")

            # Compute similarity
            similarity = self._get_similarity_cached(
                term_a=candidate_term,
                description_a=candidate_desc,
                term_b=representative_term,
                description_b=representative_desc,
            ) / 100.0  # Convert LLM scale (0-100) to 0-1

            logger.debug(
                f"Cross-branch check: '{candidate_term}' vs class '{class_node}' "
                f"(representative: '{representative_term}'): similarity={similarity:.3f}"
            )

            # Add cross-branch link if similarity exceeds threshold
            if similarity > cross_link_threshold:
                # Add edge to the top-level class (not the representative)
                # Edge label is "type" to indicate rdf:type relationship
                self.ontology_graph.add_edge(candidate_term, class_node, relation="type")
                logger.info(
                    f"Added cross-branch link: '{candidate_term}' → '{class_node}' "
                    f"(similarity={similarity:.3f} > threshold={cross_link_threshold:.3f})"
                )

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
        # Step 1: Collect all expandable nodes
        expandable_nodes = []
        for node_id in self.ontology_graph.nodes():
            node_attrs = self.ontology_graph.nodes[node_id]
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

    def _update_bandit(self, node: str, reward: float) -> None:
        """Update the bandit reward tracking for a node after expansion.

        Increments the node's visit count and accumulates the reward (similarity score).
        These values are used by _select_node_ucb1() to balance exploration vs. exploitation.

        The reward is expected to be in the range [0, 1], representing the average
        similarity of accepted candidates (0-100 LLM scale divided by 100).

        Args:
            node: The node ID (term string) to update.
            reward: The reward score (0-1) from this expansion iteration. Typically
                the mean similarity of accepted candidates.

        Raises:
            ValueError: If the node does not exist in the graph.
        """
        if node not in self.ontology_graph:
            raise ValueError(f"Node '{node}' not found in ontology graph")

        # Increment visit count
        current_visits = self.ontology_graph.nodes[node].get("n_visits", 0)
        new_visits = current_visits + 1
        self.ontology_graph.nodes[node]["n_visits"] = new_visits

        # Accumulate reward
        current_total_reward = self.ontology_graph.nodes[node].get("total_reward", 0.0)
        new_total_reward = current_total_reward + reward
        self.ontology_graph.nodes[node]["total_reward"] = new_total_reward

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

        # Step 4: Add accepted candidates to graph and check cross-branch links
        accepted_similarities = []

        for candidate in accepted_candidates:
            candidate_term = candidate.get("term", "")
            candidate_desc = candidate.get("description", "")

            try:
                # Add to graph
                self._add_candidate_to_graph(selected_node, candidate)
                
                # Note: Best practice would be to capture similarity here, but we need to
                # recompute it since _validate_candidates already consumed it. For efficiency,
                # we'll recompute once for reward tracking.
                similarity = self._get_similarity_cached(
                    term_a=self.ontology_graph.nodes[selected_node].get("term", selected_node),
                    description_a=self.ontology_graph.nodes[selected_node].get("description", ""),
                    term_b=candidate_term,
                    description_b=candidate_desc,
                ) / 100.0  # Convert 0-100 scale to 0-1

                accepted_similarities.append(similarity)

                # Check for cross-branch links
                self._check_cross_branch_links(candidate_term, candidate_desc)

            except Exception as e:
                logger.error(f"Error adding candidate '{candidate_term}' to graph: {e}")
                continue

        # Step 5: Compute reward (mean similarity of accepted candidates)
        if accepted_similarities:
            reward = sum(accepted_similarities) / len(accepted_similarities)
        else:
            reward = 0.0

        logger.info(f"Expansion reward for '{selected_node}': {reward:.3f}")

        # Step 6: Update bandit
        self._update_bandit(selected_node, reward)

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
        """Run the full ontology generation pipeline.

        This is the main entry point for users. It orchestrates the complete
        pipeline:
        1. Seed generation: LLM-generated structured 3-level taxonomy
        2. Structural validation: Pairwise similarity pruning of weak edges
        3. Iterative expansion: UCB1-guided bottom-up population of ontology
        4. RDF serialization: Convert DiGraph to RDF/OWL triples
        5. Turtle output: Write serialized ontology to file

        The expansion loop uses early termination: if reward plateaus (3 consecutive
        iterations with < 0.01 delta) AND all non-leaf nodes have been visited at
        least once, the loop terminates early.

        Returns:
            rdflib.Graph: The RDF graph representing the final ontology.

        Raises:
            ValueError: If seed generation fails (LLM returns None or invalid structure).
        """
        import pathlib

        # Phase 1: Seed generation
        logger.info(f"Phase 1: Generating seed from domain '{self.domain}'")
        seed = self.generate_initial_terms()
        if seed is None:
            raise ValueError("Seed generation failed: LLM returned None or invalid structure")
        
        self.seed = seed
        logger.info(f"Seed generated successfully: {len(seed)} top-level classes")

        # Phase 2: Seed-to-DiGraph conversion
        logger.info("Phase 2: Converting seed to ontology graph")
        self.create_seed_ontology()
        num_nodes = self.ontology_graph.number_of_nodes()
        num_edges = self.ontology_graph.number_of_edges()
        logger.info(f"Ontology graph created: {num_nodes} nodes, {num_edges} edges")

        # Phase 3: Structural validation and pruning
        logger.info("Phase 3: Validating structure and pruning weak edges")
        validation_summary = self.validate_structure()
        logger.info(
            f"Validation complete: {validation_summary['edges_pruned']} edges pruned, "
            f"{validation_summary['orphaned_nodes']} orphaned nodes"
        )

        # Phase 4: UCB1-guided iterative expansion
        logger.info("Phase 4: Running expansion loop with UCB1 selection")
        reward_history = []
        plateau_count = 0
        iteration_count = 0

        for iteration in range(self.max_iterations):
            iteration_count = iteration + 1
            logger.info(f"Expansion iteration {iteration_count}/{self.max_iterations}")

            # Run one expansion iteration
            stats = self.expand_ontology()

            # Check if no expandable nodes remain
            if stats["node"] is None:
                logger.info("No expandable nodes remaining; terminating expansion")
                break

            # Track reward
            current_reward = stats["reward"]

            # Check for plateau (reward delta < 0.01)
            if reward_history:
                delta = abs(current_reward - reward_history[-1])
                if delta < 0.01:
                    plateau_count += 1
                    logger.debug(f"Plateau detected (delta={delta:.4f}); count={plateau_count}")
                else:
                    plateau_count = 0
                    logger.debug(f"Reward improved (delta={delta:.4f}); plateau reset")
            
            reward_history.append(current_reward)

            # Check early termination condition:
            # Plateau for 3+ iterations AND all non-leaf nodes visited at least once
            if plateau_count >= 3:
                # Check if all non-instance nodes have n_visits >= 1
                non_instance_nodes = [
                    node_id for node_id in self.ontology_graph.nodes()
                    if self.ontology_graph.nodes[node_id].get("level") != "instance"
                ]
                all_visited = all(
                    self.ontology_graph.nodes[node].get("n_visits", 0) >= 1
                    for node in non_instance_nodes
                )

                if all_visited:
                    logger.info(
                        f"Early termination: plateau for {plateau_count} iterations "
                        "and all non-instance nodes visited"
                    )
                    break

            logger.info(
                f"Iteration {iteration_count}: "
                f"generated={stats['candidates_generated']}, "
                f"accepted={stats['candidates_accepted']}, "
                f"reward={stats['reward']:.3f}"
            )

        logger.info(
            f"Expansion loop complete: {iteration_count} iterations, "
            f"graph now has {self.ontology_graph.number_of_nodes()} nodes"
        )

        # Phase 5: RDF serialization
        logger.info("Phase 5: Building RDF graph from DiGraph")
        self.build_ontology()
        logger.info("Serializing to Turtle format")
        turtle_output = self.serialize_ontology()

        # Phase 6: File output
        logger.info("Phase 6: Writing output to file")
        output_path = pathlib.Path("output") / "ontology.ttl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(turtle_output)
        logger.info(f"Ontology written to {output_path.absolute()}")

        # Phase 7: Return RDF graph
        logger.info(f"Pipeline complete: {self.ontology_graph.number_of_nodes()} nodes, "
                   f"{self.rdf.__len__()} RDF triples")
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
