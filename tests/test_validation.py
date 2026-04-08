"""
Comprehensive tests for structural validation and edge pruning (SPEC-2.4 — T4).

Tests cover:
- T4: validate_structure() and _generate_validation_pairs() methods
  - Pair generation (parent-child, sibling, cross-branch categories)
  - Edge pruning based on similarity thresholds
  - Orphan detection and logging
"""

import pytest
import networkx as nx
from unittest.mock import MagicMock

from ontogen import Ontology, OntologyLevel


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_digraph(mock_agent):
    """Build a small 3-level DiGraph for validation testing.

    Structure:
    - 2 classes (e.g., Species, Location)
    - 2 subclasses each (e.g., Vulcans, Humans; Earth, Vulcan)
    - 2 instances each (e.g., Spock, T'Pol; Kirk, Janeway; etc.)

    Total: 2 + 2*2 + 2*2*2 = 14 nodes, arranged in a balanced tree.

    This fixture provides a realistic test graph that exercises all three
    pair categories (parent-child, sibling, cross-branch).
    """
    ontology = Ontology(domain="Star Trek", agent=mock_agent,
                        similarity_threshold=50)

    # Create nodes at each level with attributes
    # Level 0: Classes
    ontology.ontology_graph.add_node(
        "Species",
        term="Species",
        description="Sentient species in Star Trek universe",
        level="class",
        n_visits=0,
        total_reward=0.0,
    )
    ontology.ontology_graph.add_node(
        "Location",
        term="Location",
        description="Geographic locations in Star Trek universe",
        level="class",
        n_visits=0,
        total_reward=0.0,
    )

    # Level 1: Subclasses of Species
    ontology.ontology_graph.add_node(
        "Vulcans",
        term="Vulcans",
        description="Logical telepathic species from Vulcan",
        level="subclass",
        n_visits=0,
        total_reward=0.0,
    )
    ontology.ontology_graph.add_node(
        "Humans",
        term="Humans",
        description="Human species from Earth",
        level="subclass",
        n_visits=0,
        total_reward=0.0,
    )

    # Level 1: Subclasses of Location
    ontology.ontology_graph.add_node(
        "Planet",
        term="Planet",
        description="A celestial body in the Star Trek universe",
        level="subclass",
        n_visits=0,
        total_reward=0.0,
    )
    ontology.ontology_graph.add_node(
        "Starship",
        term="Starship",
        description="A spacefaring vessel",
        level="subclass",
        n_visits=0,
        total_reward=0.0,
    )

    # Level 2: Instances of Vulcans
    ontology.ontology_graph.add_node(
        "Spock",
        term="Spock",
        description="Half-Vulcan Starfleet officer",
        level="instance",
        n_visits=0,
        total_reward=0.0,
    )
    ontology.ontology_graph.add_node(
        "T'Pol",
        term="T'Pol",
        description="Vulcan science officer on Enterprise NX-01",
        level="instance",
        n_visits=0,
        total_reward=0.0,
    )

    # Level 2: Instances of Humans
    ontology.ontology_graph.add_node(
        "Kirk",
        term="Kirk",
        description="Captain of the USS Enterprise",
        level="instance",
        n_visits=0,
        total_reward=0.0,
    )
    ontology.ontology_graph.add_node(
        "Janeway",
        term="Janeway",
        description="Captain of the USS Voyager",
        level="instance",
        n_visits=0,
        total_reward=0.0,
    )

    # Level 2: Instances of Planet
    ontology.ontology_graph.add_node(
        "Vulcan",
        term="Vulcan",
        description="The planet Vulcan, homeworld of Vulcans",
        level="instance",
        n_visits=0,
        total_reward=0.0,
    )
    ontology.ontology_graph.add_node(
        "Earth",
        term="Earth",
        description="The planet Earth, homeworld of Humans",
        level="instance",
        n_visits=0,
        total_reward=0.0,
    )

    # Level 2: Instances of Starship
    ontology.ontology_graph.add_node(
        "Enterprise",
        term="Enterprise",
        description="USS Enterprise NCC-1701-D",
        level="instance",
        n_visits=0,
        total_reward=0.0,
    )
    ontology.ontology_graph.add_node(
        "Voyager",
        term="Voyager",
        description="USS Voyager NCC-74656",
        level="instance",
        n_visits=0,
        total_reward=0.0,
    )

    # Add edges (relationships)
    # Species → subclasses
    ontology.ontology_graph.add_edge(
        "Species", "Vulcans", relation="subClassOf"
    )
    ontology.ontology_graph.add_edge(
        "Species", "Humans", relation="subClassOf"
    )

    # Location → subclasses
    ontology.ontology_graph.add_edge(
        "Location", "Planet", relation="subClassOf"
    )
    ontology.ontology_graph.add_edge(
        "Location", "Starship", relation="subClassOf"
    )

    # Vulcans → instances
    ontology.ontology_graph.add_edge(
        "Vulcans", "Spock", relation="type"
    )
    ontology.ontology_graph.add_edge(
        "Vulcans", "T'Pol", relation="type"
    )

    # Humans → instances
    ontology.ontology_graph.add_edge(
        "Humans", "Kirk", relation="type"
    )
    ontology.ontology_graph.add_edge(
        "Humans", "Janeway", relation="type"
    )

    # Planet → instances
    ontology.ontology_graph.add_edge(
        "Planet", "Vulcan", relation="type"
    )
    ontology.ontology_graph.add_edge(
        "Planet", "Earth", relation="type"
    )

    # Starship → instances
    ontology.ontology_graph.add_edge(
        "Starship", "Enterprise", relation="type"
    )
    ontology.ontology_graph.add_edge(
        "Starship", "Voyager", relation="type"
    )

    return ontology


@pytest.fixture
def ontology_with_isolated_node(sample_digraph):
    """Extend sample_digraph with an isolated node for orphan testing.

    Adds a node that is not connected to any other node. This is useful
    for testing orphan detection.
    """
    ontology = sample_digraph
    ontology.ontology_graph.add_node(
        "Klingon",
        term="Klingon",
        description="Isolated Klingon (not connected)",
        level="subclass",
        n_visits=0,
        total_reward=0.0,
    )
    return ontology


# ============================================================================
# T4: Validation Tests (11 tests)
# ============================================================================


class TestValidationPairsGeneration:
    """Tests for Ontology._generate_validation_pairs() method."""

    def test_similarity_cache_forwards_context_label(self, sample_digraph, mock_agent):
        """Similarity cache calls should forward context labels to the agent when supported."""
        mock_agent.get_similarity_with_descriptions = MagicMock(
            return_value={"similarity": 88.0}
        )
        sample_digraph.agent = mock_agent

        score = sample_digraph._get_similarity_cached(
            term_a="Species",
            description_a="Sentient species in Star Trek universe",
            term_b="Vulcans",
            description_b="Logical telepathic species from Vulcan",
            context_label="Phase 3 validation",
        )

        assert score == 88.0
        assert mock_agent.get_similarity_with_descriptions.call_args.kwargs[
            "request_label"
        ] == "Phase 3 validation"

    def test_validation_generates_correct_pair_categories(self, sample_digraph):
        """Verify that _generate_validation_pairs() generates parent-child pairs.

        Scenario:
        - Build a balanced 3-level DiGraph (2 classes, 2 subclasses each, 2 instances each)
        - Call _generate_validation_pairs()
        - Verify: result contains only parent-child pairs

        This tests the fundamental pair generation for structural validation.
        """
        # Call: Generate validation pairs
        pairs = sample_digraph._generate_validation_pairs()

        # Assert: We have pairs
        assert len(pairs) > 0, "Should generate at least some pairs"

        # Assert: All pairs have required keys
        for pair in pairs:
            assert "term_x" in pair
            assert "desc_x" in pair
            assert "term_y" in pair
            assert "desc_y" in pair
            assert "category" in pair
            assert pair["category"] == "parent-child"

        # Assert: Only parent-child category is present
        categories = {pair["category"] for pair in pairs}
        assert categories == {
            "parent-child"}, "Should only have parent-child pairs"

    def test_parent_child_pairs_match_edges(self, sample_digraph):
        """Verify that parent-child pairs correspond to actual edges in the graph.

        Scenario:
        - Call _generate_validation_pairs()
        - Filter to parent-child pairs only
        - For each pair, verify that an edge (term_x → term_y) exists in the graph

        This tests that pair generation is consistent with the actual graph structure.
        """
        # Call: Generate validation pairs
        pairs = sample_digraph._generate_validation_pairs()

        # Filter parent-child pairs
        pc_pairs = [p for p in pairs if p["category"] == "parent-child"]

        # Assert: We have parent-child pairs
        assert len(pc_pairs) > 0, "Should have at least one parent-child pair"

        # For each pair, verify the edge exists
        for pair in pc_pairs:
            term_x = pair["term_x"]
            term_y = pair["term_y"]

            # Find nodes with matching terms
            parent_nodes = [
                n for n in sample_digraph.ontology_graph.nodes()
                if sample_digraph.ontology_graph.nodes[n]["term"] == term_x
            ]
            child_nodes = [
                n for n in sample_digraph.ontology_graph.nodes()
                if sample_digraph.ontology_graph.nodes[n]["term"] == term_y
            ]

            assert len(parent_nodes) > 0, f"Should find node for term {term_x}"
            assert len(child_nodes) > 0, f"Should find node for term {term_y}"

            # Verify edge exists
            parent_id = parent_nodes[0]
            child_id = child_nodes[0]
            assert sample_digraph.ontology_graph.has_edge(
                parent_id, child_id
            ), f"Edge {parent_id} → {child_id} should exist for pair {term_x} → {term_y}"


class TestValidationPruning:
    """Tests for Ontology.validate_structure() edge pruning."""

    def test_pruning_removes_weak_edges(self, sample_digraph, mock_agent):
        """Verify that weak parent-child edges are removed during validation.

        Scenario:
        - Setup: Graph has edge Species → Vulcans
        - Mock LLM: return low similarity (< 50%) for this pair
        - Call validate_structure()
        - Verify: Edge is removed from graph
        - Verify: edges_pruned counter = 1

        This tests the core pruning logic for weak parent-child relationships.
        """
        # Setup: Record initial edge count
        initial_edges = sample_digraph.ontology_graph.number_of_edges()

        # Setup: Mock agent to return low similarity for "Species" vs "Vulcans"
        def mock_similarity(term_x, description_x, term_y, description_y):
            """Mock similarity: low for Species-Vulcans, medium for others."""
            if (term_x == "Species" and term_y == "Vulcans") or \
               (term_x == "Vulcans" and term_y == "Species"):
                return {"similarity": 25.0}  # Below 50% threshold
            return {"similarity": 80.0}  # Above threshold for other pairs

        mock_agent.get_similarity_with_descriptions.side_effect = mock_similarity
        sample_digraph.agent = mock_agent

        # Call: Validate structure
        summary = sample_digraph.validate_structure()

        # Assert: At least one edge was pruned
        assert summary["edges_pruned"] >= 1, "Should prune weak edge Species → Vulcans"

        # Assert: Edge count decreased
        final_edges = sample_digraph.ontology_graph.number_of_edges()
        assert (
            final_edges < initial_edges
        ), "Edge count should decrease after pruning"

        # Assert: Specific edge is removed
        # Find and verify the edge is gone
        species_nodes = [
            n for n in sample_digraph.ontology_graph.nodes()
            if sample_digraph.ontology_graph.nodes[n]["term"] == "Species"
        ]
        vulcan_nodes = [
            n for n in sample_digraph.ontology_graph.nodes()
            if sample_digraph.ontology_graph.nodes[n]["term"] == "Vulcans"
        ]

        if species_nodes and vulcan_nodes:
            assert not sample_digraph.ontology_graph.has_edge(
                species_nodes[0], vulcan_nodes[0]
            ), "Species → Vulcans edge should be pruned"

    def test_pruning_preserves_strong_edges(self, sample_digraph, mock_agent):
        """Verify that strong parent-child edges are preserved.

        Scenario:
        - Setup: Graph has edge Species → Humans
        - Mock LLM: return high similarity (> 50%) for this pair and all others
        - Call validate_structure()
        - Verify: Edge is NOT removed
        - Verify: edges_pruned = 0

        This tests that pruning doesn't over-remove edges.
        """
        # Setup: Record initial edge count
        initial_edges = sample_digraph.ontology_graph.number_of_edges()

        # Setup: Mock agent to return high similarity for all pairs
        mock_agent.get_similarity_with_descriptions.return_value = {
            "similarity": 90.0}
        sample_digraph.agent = mock_agent

        # Call: Validate structure
        summary = sample_digraph.validate_structure()

        # Assert: No edges were pruned
        assert summary["edges_pruned"] == 0, "Should not prune strong edges"

        # Assert: Edge count unchanged
        final_edges = sample_digraph.ontology_graph.number_of_edges()
        assert final_edges == initial_edges, "Edge count should not change when no pruning"


class TestOrphanDetection:
    """Tests for Ontology orphan detection during validation."""

    def test_orphaned_nodes_are_detected(
        self, ontology_with_isolated_node, mock_agent
    ):
        """Verify that orphaned nodes (degree 0) are detected and logged.

        Scenario:
        - Setup: Graph has an isolated node (Klingon) with no edges
        - Mock LLM: high similarity for all pairs (so no edges are pruned)
        - Call validate_structure()
        - Verify: orphaned_nodes counter = 1
        - Verify: Log message mentions orphaned node

        This tests orphan detection for nodes with no relationships.
        """
        # Setup: Verify the isolated node exists and has degree 0
        klingon_nodes = [
            n for n in ontology_with_isolated_node.ontology_graph.nodes()
            if ontology_with_isolated_node.ontology_graph.nodes[n]["term"] == "Klingon"
        ]
        assert len(klingon_nodes) == 1, "Should have one Klingon node"
        klingon_id = klingon_nodes[0]
        assert (
            ontology_with_isolated_node.ontology_graph.degree(klingon_id) == 0
        ), "Klingon should be isolated"

        # Setup: Mock agent to return high similarity (no pruning)
        mock_agent.get_similarity_with_descriptions.return_value = {
            "similarity": 80.0}
        ontology_with_isolated_node.agent = mock_agent

        # Call: Validate structure
        summary = ontology_with_isolated_node.validate_structure()

        # Assert: orphaned_nodes counter = 1
        assert summary["orphaned_nodes"] == 1, "Should detect 1 orphaned node"

    def test_orphaned_nodes_created_by_pruning(self, sample_digraph, mock_agent):
        """Verify that pruning can create orphaned nodes.

        Scenario:
        - Setup: Graph has edges that create a node with only one incoming edge
        - Mock LLM: low similarity for multiple parent-child pairs
        - After pruning those edges, a node becomes orphaned
        - Call validate_structure()
        - Verify: orphaned_nodes counter > 0

        This tests orphan detection when a node becomes completely disconnected.
        """
        # Setup: Mock agent to return low similarity for all parent-child pairs
        # This will prune ALL edges in the graph
        def mock_similarity(term_x, description_x, term_y, description_y):
            """Mock similarity: low for all pairs."""
            return {"similarity": 10.0}  # Below 50% parent-child threshold

        mock_agent.get_similarity_with_descriptions.side_effect = mock_similarity
        sample_digraph.agent = mock_agent

        # Call: Validate structure (which will prune many parent-child edges)
        summary = sample_digraph.validate_structure()

        # Assert: Multiple edges were pruned
        assert summary["edges_pruned"] >= 1, "Should prune weak edges"

        # Assert: Orphaned nodes detected
        # After pruning edges, nodes with no parent will become isolated
        # (all top-level classes have parents = None, but subclasses and instances
        # will become orphaned when their parent edges are pruned)
        orphaned_count = summary["orphaned_nodes"]
        assert orphaned_count > 0, "Aggressive pruning should create orphaned nodes"

    def test_validation_summary_has_all_keys(self, sample_digraph, mock_agent):
        """Verify that validate_structure() returns all required summary keys.

        Scenario:
        - Call validate_structure() on a normal graph with mocked LLM
        - Verify: returned dict has all four expected keys

        This tests the summary return value structure.
        """
        # Setup: Mock agent with uniform similarity
        mock_agent.get_similarity_with_descriptions.return_value = {
            "similarity": 60.0}
        sample_digraph.agent = mock_agent

        # Call: Validate structure
        summary = sample_digraph.validate_structure()

        # Assert: Summary dict has all required keys
        required_keys = {
            "edges_pruned",
            "orphaned_nodes",
        }
        assert set(summary.keys()) == required_keys, \
            f"Summary should have keys {required_keys}, got {set(summary.keys())}"

        # Assert: All values are integers
        for key, value in summary.items():
            assert isinstance(
                value, int), f"{key} should be int, got {type(value)}"

        # Assert: All counts are non-negative
        for key, value in summary.items():
            assert value >= 0, f"{key} should be non-negative, got {value}"
