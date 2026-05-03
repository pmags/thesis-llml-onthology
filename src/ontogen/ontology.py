"""
Ontology construction, expansion, and RDF serialization.

This module contains the Ontology class which orchestrates the full
ontology generation pipeline: seed generation, iterative expansion,
graph construction, and RDF serialization.
"""

import logging
import time
import pathlib
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple, Any

import pandas as pd
import networkx as nx
from rdflib import Graph, Namespace

from ontogen.expansion import ExpansionMixin
from ontogen.llm_client import ChatGpt
from ontogen.models import OntologyLevel, DEFAULT_LEVEL_SCHEMA
from ontogen.progress import print_phase
from ontogen.resolution import ResolutionMixin
from ontogen.serialization import SerializationMixin
from ontogen.seed import SeedMixin
from ontogen.validation import ValidationMixin
from ontogen.visualization import VisualizationMixin
from ontogen.expansion_models import (
    ExpansionRecord,
    GenerationHistory,
    PhaseRecord,
)

logger = logging.getLogger(__name__)


class Ontology(
    ExpansionMixin,
    ResolutionMixin,
    ValidationMixin,
    SeedMixin,
    SerializationMixin,
    VisualizationMixin,
):
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
        scope_description: Optional[str] = None,
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
        initial_seed_terms: int = 5
    ) -> None:
        # Configuration
        self.domain = domain
        self.scope_description = scope_description
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
        self.initial_seed_terms = initial_seed_terms

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

        # Expansion mode is locked to manual once a manual node expansion happens.
        # Users who want to return to automatic UCB1 selection must create a new
        # Ontology instance for a fresh session.
        self.expansion_mode: Optional[str] = None

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
            RuntimeError: If manual expansion already started on this Ontology instance.
        """

        if self.expansion_mode == "manual":
            raise RuntimeError(
                "Automatic ontology generation is unavailable after manual expansion has started "
                "on this Ontology instance. Create a new Ontology instance to run UCB1 "
                "generation automatically."
            )

        self.expansion_mode = "automatic"

        # monotonic makes it simple to measure runtime without worrying about clock changes. We add a stop and calc diff at the end
        pipeline_start = time.monotonic()

        # Initialize history object for this run.
        # It will keep track of all phases, iterations, rewards, and graph stats for post-hoc analysis and debugging.
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

        # ── Phase 1: Seed generation ─────────────────────────────────
        print_phase(1, "Seed Generation")

        phase_start = time.monotonic()
        logger.info("Phase 1: Generating seed from domain %s", self.domain)

        seed = self.generate_initial_terms(num_classes=self.initial_seed_terms)
        if seed is None:
            raise ValueError(
                "Seed generation failed: LLM returned None or invalid structure")

        num_top_classes = len(seed["taxonomy"])
        phase_duration = time.monotonic() - phase_start
        self.history.phases.append(PhaseRecord(
            phase=1, name="Seed generation", duration_seconds=phase_duration,
            details={"top_level_classes": num_top_classes},
        ))
        logger.info("Seed generated: %d top-level classes", num_top_classes)
        print(
            f"  ✓ Generated {num_top_classes} top-level classes ({phase_duration:.1f}s)")

        # ── Phase 2: Seed → DiGraph ──────────────────────────────────
        print_phase(2, "Graph Construction")

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
        logger.info(
            "Ontology graph created: %d nodes, %d edges", num_nodes, num_edges)
        print(
            f"  ✓ Created graph: {num_nodes} nodes, {num_edges} edges ({phase_duration:.1f}s)")

        # ── Phase 3: Structural validation ────────────────────────────
        print_phase(3, "Structural Validation")
        phase_start = time.monotonic()
        logger.info("Phase 3: Validating structure and pruning weak edges")

        validation_summary = self.validate_structure()
        phase_duration = time.monotonic() - phase_start
        self.history.phases.append(PhaseRecord(
            phase=3, name="Structural validation", duration_seconds=phase_duration,
            details=validation_summary,
        ))
        logger.info(
            "Validation complete: %d edges pruned, %d orphaned nodes",
            validation_summary["edges_pruned"], validation_summary["orphaned_nodes"]
        )
        print(
            f"  ✓ Pruned {validation_summary['edges_pruned']} edges, "
            f"{validation_summary['orphaned_nodes']} orphaned nodes ({phase_duration:.1f}s)"
        )

        # ── Phase 4: Expansion ────────────────────────────────────────
        expansion_state = self._run_expansion_phase(pipeline_start)
        iteration_count = expansion_state.iteration_count
        termination_reason = expansion_state.termination_reason

        # ── Phase 5: RDF serialization ────────────────────────────────
        print_phase(5, "RDF Serialization")
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
        print_phase(6, "File Output")
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
        logger.info("Ontology written to %s", output_path.absolute())
        print(
            f"  ✓ Written to {output_path.absolute()} ({phase_duration:.1f}s)")

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
            "Pipeline complete: %d nodes, %d RDF triples "
            "in %.1f s", self.history.final_nodes, self.history.final_triples, total_duration
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
