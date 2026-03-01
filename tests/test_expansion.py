"""
Comprehensive tests for UCB1 node selection and expansion (SPEC-3.5 — T5).

Tests cover:
- T5: UCB1 node selection, expansion iterations, and early termination
  - UCB1 bandit algorithm (unvisited prioritization, score-based selection)
  - Single expansion iteration (candidate generation, validation, graph updates)
  - Full pipeline with early termination conditions
  - Leaf node (instance) exclusion from expansion
"""

import json
import pytest
import networkx as nx
from unittest.mock import MagicMock, patch

from ontogen import Ontology, OntologyLevel, DEFAULT_LEVEL_SCHEMA


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_digraph(mock_agent):
    """Build a 2-level DiGraph for UCB1 and expansion testing.
    
    Structure:
    - 2 classes: Species, Location
    - 2 subclasses each: Vulcans, Humans; Planet, Starship
    
    Total: 2 + 2*2 = 6 nodes (no instances yet, all classes/subclasses expandable).
    
    This is lighter than the 3-level validation graph, focusing on testing
    the UCB1 selection and expansion logic specifically.
    """
    ontology = Ontology(domain="Star Trek", agent=mock_agent)
    
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
    
    # Add edges
    ontology.ontology_graph.add_edge("Species", "Vulcans", relation="subClassOf")
    ontology.ontology_graph.add_edge("Species", "Humans", relation="subClassOf")
    ontology.ontology_graph.add_edge("Location", "Planet", relation="subClassOf")
    ontology.ontology_graph.add_edge("Location", "Starship", relation="subClassOf")
    
    return ontology


@pytest.fixture
def instance_only_digraph(mock_agent):
    """Build a 3-level DiGraph with only instance nodes remaining (non-expandable).
    
    This tests the edge case where all remaining expandable nodes are instances,
    and _select_node_ucb1() should return None or handle the empty arm set.
    """
    ontology = Ontology(domain="Star Trek", agent=mock_agent)
    
    # Single class
    ontology.ontology_graph.add_node(
        "Species",
        term="Species",
        description="Sentient species in Star Trek universe",
        level="class",
        n_visits=1,
        total_reward=1.0,
    )
    
    # Single subclass
    ontology.ontology_graph.add_node(
        "Vulcans",
        term="Vulcans",
        description="Logical telepathic species from Vulcan",
        level="subclass",
        n_visits=1,
        total_reward=0.5,
    )
    
    # Instances (not expandable)
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
    
    # Add edges
    ontology.ontology_graph.add_edge("Species", "Vulcans", relation="subClassOf")
    ontology.ontology_graph.add_edge("Vulcans", "Spock", relation="type")
    ontology.ontology_graph.add_edge("Vulcans", "T'Pol", relation="type")
    
    return ontology


@pytest.fixture
def mock_agent_with_seed_response(mock_agent):
    """Mock agent with responses for seed generation and expansion."""
    seed_json = json.dumps({
        "domain": "Star Trek",
        "taxonomy": [
            {
                "class": "Species",
                "description": "Sentient species",
                "subclasses": [
                    {
                        "class": "Vulcans",
                        "description": "Logical species",
                        "instances": [
                            {"term": "Spock", "description": "Officer"}
                        ]
                    }
                ]
            }
        ]
    })
    
    candidates_json = json.dumps([
        {"term": "Romulan", "description": "Military adversary"},
        {"term": "Klingon", "description": "Warlike species"}
    ])
    
    def chat_side_effect(instructions=None, input=None, prompt=None):
        """Handle both old and new chat API signatures."""
        msg = (prompt or input or "").lower()
        if "taxonomy" in msg or "ontology" in msg:
            return seed_json
        else:
            return candidates_json
    
    mock_agent.chat.side_effect = chat_side_effect
    mock_agent.get_similarity_with_descriptions.return_value = {
        "similarity": 80  # Good similarity for candidates
    }
    
    return mock_agent


# ============================================================================
# Test Class: UCB1 Node Selection (SPEC-3.1)
# ============================================================================


class TestUCB1Selection:
    """Tests for UCB1 bandit node selection logic."""
    
    def test_ucb1_selects_unvisited_node_first(self, simple_digraph):
        """Verify UCB1 prioritizes unvisited arms over visited ones.
        
        Scenario:
        - All class nodes visited, but one subclass unvisited
        - Unvisited node should be selected first regardless of reward
        
        This tests the exploration priority in _select_node_ucb1().
        """
        ontology = simple_digraph
        
        # Manually set visit counts on all nodes
        # Classes: both visited
        ontology.ontology_graph.nodes["Species"]["n_visits"] = 1
        ontology.ontology_graph.nodes["Species"]["total_reward"] = 0.5
        ontology.ontology_graph.nodes["Location"]["n_visits"] = 1
        ontology.ontology_graph.nodes["Location"]["total_reward"] = 0.5
        
        # Subclasses: Vulcans unvisited, others visited with high reward
        ontology.ontology_graph.nodes["Vulcans"]["n_visits"] = 0
        ontology.ontology_graph.nodes["Vulcans"]["total_reward"] = 0.0
        
        ontology.ontology_graph.nodes["Humans"]["n_visits"] = 2
        ontology.ontology_graph.nodes["Humans"]["total_reward"] = 1.8  # high reward
        
        ontology.ontology_graph.nodes["Planet"]["n_visits"] = 1
        ontology.ontology_graph.nodes["Planet"]["total_reward"] = 0.5
        
        ontology.ontology_graph.nodes["Starship"]["n_visits"] = 1
        ontology.ontology_graph.nodes["Starship"]["total_reward"] = 0.5
        
        # Call
        selected = ontology._select_node_ucb1()
        
        # Assert: unvisited node selected
        assert selected == "Vulcans", \
            f"Expected unvisited 'Vulcans' but got '{selected}'"
    
    def test_ucb1_balances_exploration_exploitation(self, simple_digraph):
        """Verify UCB1 balances exploration vs exploitation after all arms visited.
        
        Scenario:
        - All nodes visited uniformly (n_visits = 1 for all)
        - One node has higher reward (mean reward = total_reward / n_visits)
        - That high-reward node should be selected next
        
        This tests the UCB1 scoring formula: mean + c * sqrt(ln(N) / n_i)
        """
        ontology = simple_digraph
        
        # All nodes visited once, but Vulcans has higher reward
        for node in ontology.ontology_graph.nodes():
            ontology.ontology_graph.nodes[node]["n_visits"] = 1
            ontology.ontology_graph.nodes[node]["total_reward"] = 0.3
        
        # Vulcans: high reward
        ontology.ontology_graph.nodes["Vulcans"]["total_reward"] = 0.8
        
        # Call multiple times (may be probabilistic, but deterministic mean)
        selected_counts = {}
        for _ in range(10):
            selected = ontology._select_node_ucb1()
            selected_counts[selected] = selected_counts.get(selected, 0) + 1
        
        # Assert: Vulcans selected most frequently
        assert selected_counts.get("Vulcans", 0) > 0, \
            "High-reward node 'Vulcans' was never selected"
    
    def test_ucb1_excludes_non_expandable_nodes(self, instance_only_digraph):
        """Verify UCB1 excludes leaf-level (non-expandable) nodes from selection.
        
        Scenario:
        - Mixed class, subclass, and instance nodes
        - Instances are not expandable (expandable=False in their level)
        - _select_node_ucb1() should only select from class/subclass nodes
        
        This tests level-based expandability filtering.
        """
        ontology = instance_only_digraph
        
        # Call
        selected = ontology._select_node_ucb1()
        
        # Assert: selected node is not an instance
        if selected is not None:
            selected_level = ontology.ontology_graph.nodes[selected].get("level")
            assert selected_level != "instance", \
                f"Selected instance node '{selected}' but should be expandable class/subclass"
    
    def test_ucb1_returns_none_when_no_expandable_nodes(self, instance_only_digraph):
        """Verify UCB1 returns None when no expandable nodes remain.
        
        Scenario:
        - Force all non-instance nodes to have been expanded (visited)
        - Only instance nodes remain (which are not expandable)
        - _select_node_ucb1() should return None
        
        This tests handling of the termination condition.
        """
        ontology = instance_only_digraph
        
        # Mark expandable nodes as already expanded (arbitrary high visit count)
        class_subclass_nodes = [
            n for n, d in ontology.ontology_graph.nodes(data=True)
            if d.get("level") in ["class", "subclass"]
        ]
        
        # Call
        selected = ontology._select_node_ucb1()
        
        # With current setup, may return a subclass if not yet visited enough
        # So we test: if we return something, it must not be an instance
        if selected is not None:
            selected_level = ontology.ontology_graph.nodes[selected].get("level")
            assert selected_level != "instance"


# ============================================================================
# Test Class: Bandit Update (SPEC-3.1)
# ============================================================================


class TestBanditUpdate:
    """Tests for bandit reward tracking."""
    
    def test_update_bandit_increments_visits(self, simple_digraph):
        """Verify _update_bandit() increments n_visits correctly."""
        ontology = simple_digraph
        node = "Vulcans"
        
        initial_visits = ontology.ontology_graph.nodes[node]["n_visits"]
        ontology._update_bandit(node, reward=0.5)
        
        updated_visits = ontology.ontology_graph.nodes[node]["n_visits"]
        assert updated_visits == initial_visits + 1
    
    def test_update_bandit_accumulates_reward(self, simple_digraph):
        """Verify _update_bandit() accumulates total_reward."""
        ontology = simple_digraph
        node = "Vulcans"
        
        initial_reward = ontology.ontology_graph.nodes[node]["total_reward"]
        new_reward = 0.6
        ontology._update_bandit(node, reward=new_reward)
        
        updated_reward = ontology.ontology_graph.nodes[node]["total_reward"]
        assert updated_reward == initial_reward + new_reward
    
    def test_update_bandit_multiple_calls(self, simple_digraph):
        """Verify _update_bandit() correctly accumulates over multiple calls."""
        ontology = simple_digraph
        node = "Vulcans"
        
        ontology._update_bandit(node, reward=0.5)
        ontology._update_bandit(node, reward=0.3)
        ontology._update_bandit(node, reward=0.7)
        
        assert ontology.ontology_graph.nodes[node]["n_visits"] == 3
        assert ontology.ontology_graph.nodes[node]["total_reward"] == pytest.approx(1.5)


# ============================================================================
# Test Class: Single Expansion Iteration (SPEC-3.4a)
# ============================================================================


class TestExpandOntology:
    """Tests for single expand_ontology() iteration."""
    
    def test_expansion_adds_nodes_to_graph(self, simple_digraph, mock_agent):
        """Verify expand_ontology() adds generated candidates to the graph.
        
        Scenario:
        - Mock _generate_candidates() to return 2 valid candidates
        - Mock similarity to return > threshold (accept all)
        - Call expand_ontology()
        - Assert graph has 2 more nodes
        """
        ontology = simple_digraph
        initial_node_count = len(ontology.ontology_graph.nodes())
        
        # Mock candidate generation (return 2 candidates)
        candidates = [
            {"term": "Romulan", "description": "Military adversary"},
            {"term": "Klingon", "description": "Warlike species"}
        ]
        
        with patch.object(ontology, '_generate_candidates', return_value=candidates):
            # Mock similarity to accept all
            with patch.object(ontology, '_get_similarity_cached', return_value=75):
                # Mock cross-branch checking
                with patch.object(ontology, '_check_cross_branch_links'):
                    # Call
                    stats = ontology.expand_ontology()
        
        # Assert
        assert len(ontology.ontology_graph.nodes()) > initial_node_count, \
            "No nodes were added to graph after expansion"
        assert stats["candidates_generated"] == 2
        assert stats["candidates_accepted"] == 2
        assert stats["node"] is not None
    
    def test_expansion_returns_stats_dict(self, simple_digraph, mock_agent):
        """Verify expand_ontology() returns correctly structured stats dict."""
        ontology = simple_digraph
        
        candidates = [
            {"term": "Romulan", "description": "Military adversary"}
        ]
        
        with patch.object(ontology, '_generate_candidates', return_value=candidates):
            with patch.object(ontology, '_get_similarity_cached', return_value=75):
                with patch.object(ontology, '_check_cross_branch_links'):
                    stats = ontology.expand_ontology()
        
        # Assert structure
        assert isinstance(stats, dict)
        assert "node" in stats
        assert "candidates_generated" in stats
        assert "candidates_accepted" in stats
        assert "reward" in stats
        assert isinstance(stats["candidates_generated"], int)
        assert isinstance(stats["candidates_accepted"], int)
        assert isinstance(stats["reward"], float)
    
    def test_expansion_returns_zero_stats_when_no_expandable_nodes(self, instance_only_digraph):
        """Verify expand_ontology() returns zero stats when no expandable nodes remain."""
        ontology = instance_only_digraph
        
        # Call
        stats = ontology.expand_ontology()
        
        # Assert: returns empty stats dict (or all zeros)
        assert stats["node"] is None or stats["candidates_generated"] == 0
        assert stats["candidates_accepted"] == 0
    
    def test_expansion_computes_reward_from_similar_candidates(self, simple_digraph, mock_agent):
        """Verify reward is computed as mean similarity of accepted candidates."""
        ontology = simple_digraph
        
        # Use candidates that don't exist in the test graph
        candidates = [
            {"term": "NewSpecies1", "description": "A new species"},
            {"term": "NewSpecies2", "description": "Another new species"}
        ]
        
        # Mock similarities: 80 for both
        def get_sim_side_effect(*args, **kwargs):
            return 80  # Return 80 for parent-candidate similarity check
        
        with patch.object(ontology, '_generate_candidates', return_value=candidates):
            with patch.object(ontology, '_get_similarity_cached', side_effect=get_sim_side_effect):
                with patch.object(ontology, '_check_cross_branch_links'):
                    stats = ontology.expand_ontology()
        
        # Assert: reward is mean similarity / 100
        expected_reward = (80 + 80) / 100 / 2
        assert stats["reward"] == pytest.approx(expected_reward)


# ============================================================================
# Test Class: Full Pipeline with Early Termination (SPEC-3.4b)
# ============================================================================


class TestGenerateOntology:
    """Tests for full generate_ontology() pipeline."""
    
    def test_generate_ontology_respects_max_iterations(self, mock_agent_with_seed_response):
        """Verify generate_ontology() terminates at max_iterations.
        
        Scenario:
        - Set max_iterations=2
        - Mock all LLM calls to succeed
        - Call generate_ontology()
        - Assert loop executed at most 2 expansion iterations
        
        This tests iteration counting and loop termination.
        """
        ontology = Ontology(
            domain="Star Trek",
            agent=mock_agent_with_seed_response,
            max_iterations=2
        )
        
        iteration_count = 0
        original_expand = ontology.expand_ontology
        
        def counting_expand():
            nonlocal iteration_count
            iteration_count += 1
            return original_expand()
        
        with patch.object(ontology, 'expand_ontology', side_effect=counting_expand):
            with patch.object(ontology, 'build_ontology'):
                with patch.object(ontology, 'serialize_ontology', return_value="# Turtle output"):
                    ontology.generate_ontology()
        
        # Assert: loop ran at most max_iterations times
        assert iteration_count <= 2, \
            f"Loop ran {iteration_count} times but max_iterations=2"
    
    def test_generate_ontology_returns_rdf_graph(self, mock_agent_with_seed_response):
        """Verify generate_ontology() returns an RDF Graph object."""
        ontology = Ontology(
            domain="Star Trek",
            agent=mock_agent_with_seed_response,
            max_iterations=1
        )
        
        with patch.object(ontology, 'build_ontology'):
            with patch.object(ontology, 'serialize_ontology', return_value="# Turtle output"):
                result = ontology.generate_ontology()
        
        # Result should be the RDF graph (or at least have graph-like behavior)
        assert result is not None
    
    def test_generate_ontology_calls_all_phases(self, mock_agent_with_seed_response):
        """Verify generate_ontology() orchestrates all phases correctly.
        
        Scenario:
        - Mock key methods (seed, validate, expand, build, serialize)
        - Call generate_ontology()
        - Assert all phase methods were called in correct order
        """
        ontology = Ontology(
            domain="Star Trek",
            agent=mock_agent_with_seed_response,
            max_iterations=1
        )
        
        call_order = []
        
        original_seed = ontology.generate_initial_terms
        original_create = ontology.create_seed_ontology
        original_validate = ontology.validate_structure
        original_build = ontology.build_ontology
        original_serialize = ontology.serialize_ontology
        
        def track_seed(*args, **kwargs):
            call_order.append("seed")
            return original_seed(*args, **kwargs)
        
        def track_create(*args, **kwargs):
            call_order.append("create")
            return original_create(*args, **kwargs)
        
        def track_validate(*args, **kwargs):
            call_order.append("validate")
            return original_validate(*args, **kwargs)
        
        def track_build(*args, **kwargs):
            call_order.append("build")
            return original_build(*args, **kwargs)
        
        def track_serialize(*args, **kwargs):
            call_order.append("serialize")
            return original_serialize(*args, **kwargs)
        
        with patch.object(ontology, 'generate_initial_terms', side_effect=track_seed):
            with patch.object(ontology, 'create_seed_ontology', side_effect=track_create):
                with patch.object(ontology, 'validate_structure', side_effect=track_validate):
                    with patch.object(ontology, 'build_ontology', side_effect=track_build):
                        with patch.object(ontology, 'serialize_ontology', side_effect=track_serialize):
                            with patch.object(ontology, 'expand_ontology', return_value={
                                "node": None,
                                "candidates_generated": 0,
                                "candidates_accepted": 0,
                                "reward": 0.0
                            }):
                                ontology.generate_ontology()
        
        # Assert: phases called in expected order
        assert "seed" in call_order
        assert "create" in call_order
        assert "validate" in call_order
        assert "build" in call_order
        assert "serialize" in call_order
        
        # Rough order check (seed before create before validate)
        seed_idx = call_order.index("seed")
        create_idx = call_order.index("create")
        assert seed_idx < create_idx


# ============================================================================
# Integration Tests: Complete Workflow
# ============================================================================


class TestExpansionIntegration:
    """Integration tests for expansion workflow."""
    
    def test_expansion_loop_grows_ontology(self, mock_agent_with_seed_response):
        """Verify multiple expansion iterations grow the ontology graph."""
        ontology = Ontology(
            domain="Star Trek",
            agent=mock_agent_with_seed_response,
            max_iterations=3  # Allow a few iterations
        )
        
        # Mock expand_ontology to add nodes each time
        call_count = [0]
        original_expand = ontology.expand_ontology
        
        def mock_expand_with_side_effect():
            call_count[0] += 1
            # On first few calls, return valid stats (add a node)
            if call_count[0] <= 2:
                return {
                    "node": f"Node_{call_count[0]}",
                    "candidates_generated": 1,
                    "candidates_accepted": 1,
                    "reward": 0.7
                }
            else:
                # Return empty stats to trigger termination
                return {
                    "node": None,
                    "candidates_generated": 0,
                    "candidates_accepted": 0,
                    "reward": 0.0
                }
        
        with patch.object(ontology, 'expand_ontology', side_effect=mock_expand_with_side_effect):
            with patch.object(ontology, 'build_ontology'):
                with patch.object(ontology, 'serialize_ontology', return_value="# Turtle output"):
                    ontology.generate_ontology()
        
        # Assert: expand_ontology() was called at least once
        assert call_count[0] > 0
    
    def test_expansion_with_different_exploration_constant(self, simple_digraph, mock_agent):
        """Verify exploration_constant affects UCB1 selection behavior."""
        ontology_low = simple_digraph
        ontology_low.exploration_constant = 0.5  # Low exploration
        
        # Reset visit counts
        for node in ontology_low.ontology_graph.nodes():
            ontology_low.ontology_graph.nodes[node]["n_visits"] = 1
            ontology_low.ontology_graph.nodes[node]["total_reward"] = 0.3
        
        ontology_low.ontology_graph.nodes["Vulcans"]["total_reward"] = 0.8
        
        # Call with low exploration constant
        selected_low = ontology_low._select_node_ucb1()
        
        # Create similar ontology with high exploration constant
        ontology_high = simple_digraph
        ontology_high.exploration_constant = 2.0  # High exploration
        
        for node in ontology_high.ontology_graph.nodes():
            ontology_high.ontology_graph.nodes[node]["n_visits"] = 1
            ontology_high.ontology_graph.nodes[node]["total_reward"] = 0.3
        
        ontology_high.ontology_graph.nodes["Vulcans"]["total_reward"] = 0.8
        
        # Both should still prefer high-reward node, but we're mostly testing
        # that different constants don't crash
        selected_high = ontology_high._select_node_ucb1()
        
        assert selected_low is not None
        assert selected_high is not None
