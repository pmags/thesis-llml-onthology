"""
Comprehensive tests for seed generation and seed-to-graph conversion (SPEC-1.1 & SPEC-1.2).

Tests cover:
- T1: generate_initial_terms() method (7 tests)
- T2: create_seed_ontology() method (8 tests)
"""

import json
import pytest
import networkx as nx

from ontogen import Ontology, OntologyLevel


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def ontology_empty(mock_agent):
    """Fresh Ontology instance with no seed, for testing generate_initial_terms()."""
    return Ontology(domain="Star Trek", agent=mock_agent)


@pytest.fixture
def ontology_with_seed(mock_agent):
    """Ontology instance with pre-populated seed, for testing create_seed_ontology()."""
    ontology = Ontology(domain="Star Trek", agent=mock_agent)
    ontology.seed = {
        "domain": "Star Trek",
        "taxonomy": [
            {
                "class": "Species",
                "description": "Sentient species in the Star Trek universe",
                "subclasses": [
                    {
                        "class": "Vulcans",
                        "description": "Logical, telepathic species from planet Vulcan",
                        "instances": [
                            {"term": "Spock",
                                "description": "Half-Vulcan Starfleet officer"},
                            {"term": "T'Pol",
                                "description": "Vulcan science officer on Enterprise NX-01"}
                        ]
                    },
                    {
                        "class": "Humans",
                        "description": "Human species from Earth",
                        "instances": [
                            {"term": "Kirk", "description": "Captain of the Enterprise"},
                            {"term": "Janeway", "description": "Captain of the Voyager"}
                        ]
                    }
                ]
            }
        ]
    }
    return ontology


# ============================================================================
# T1: generate_initial_terms() Tests (7 tests)
# ============================================================================


class TestGenerateInitialTerms:
    """Tests for Ontology.generate_initial_terms() method."""

    def test_generate_initial_terms_success(self, ontology_empty, mock_agent):
        """Verify generate_initial_terms() parses valid LLM response and returns seed.

        Scenario: LLM returns valid JSON matching expected schema
        """
        valid_response = json.dumps({
            "domain": "Star Trek",
            "taxonomy": [
                {
                    "class": "Species",
                    "description": "Sentient species in the Star Trek universe",
                    "subclasses": [
                        {
                            "class": "Vulcans",
                            "description": "Logical, telepathic species from planet Vulcan",
                            "instances": [
                                {"term": "Spock",
                                    "description": "Half-Vulcan Starfleet officer"},
                                {"term": "T'Pol",
                                    "description": "Vulcan science officer"}
                            ]
                        }
                    ]
                }
            ]
        })
        mock_agent.chat.return_value = valid_response
        ontology_empty.agent = mock_agent

        # Call
        result = ontology_empty.generate_initial_terms(num_classes=1)

        # Assert
        assert result is not None
        assert result["domain"] == "Star Trek"
        assert len(result["taxonomy"]) == 1
        assert result["taxonomy"][0]["class"] == "Species"
        assert ontology_empty.seed == result

    def test_generate_initial_terms_invalid_json(self, ontology_empty, mock_agent):
        """Verify generate_initial_terms() handles malformed JSON gracefully.

        Scenario: LLM returns invalid/malformed JSON
        """
        malformed = "{ this is not valid json: 123 }"
        mock_agent.chat.return_value = malformed
        ontology_empty.agent = mock_agent

        # Call
        result = ontology_empty.generate_initial_terms()

        # Assert
        assert result is None
        assert ontology_empty.seed is None

    def test_generate_initial_terms_missing_domain(self, ontology_empty, mock_agent):
        """Verify handling of JSON missing 'domain' key.

        Scenario: JSON is valid but missing required 'domain' key
        """
        incomplete = json.dumps({"taxonomy": []})
        mock_agent.chat.return_value = incomplete
        ontology_empty.agent = mock_agent

        # Call
        result = ontology_empty.generate_initial_terms()

        # Assert
        assert result is None
        assert ontology_empty.seed is None

    def test_generate_initial_terms_missing_taxonomy(self, ontology_empty, mock_agent):
        """Verify handling of JSON missing 'taxonomy' key.

        Scenario: JSON is valid but missing required 'taxonomy' key
        """
        incomplete = json.dumps({"domain": "Star Trek"})
        mock_agent.chat.return_value = incomplete
        ontology_empty.agent = mock_agent

        # Call
        result = ontology_empty.generate_initial_terms()

        # Assert
        assert result is None
        assert ontology_empty.seed is None

    def test_generate_initial_terms_empty_taxonomy(self, ontology_empty, mock_agent):
        """Verify empty taxonomy list is rejected.

        Scenario: Response is valid JSON with empty "taxonomy" list
        """
        empty_response = json.dumps({"domain": "Star Trek", "taxonomy": []})
        mock_agent.chat.return_value = empty_response
        ontology_empty.agent = mock_agent

        # Call
        result = ontology_empty.generate_initial_terms()

        # Assert
        assert result is None
        assert ontology_empty.seed is None

    def test_generate_initial_terms_sets_seed(self, ontology_empty, mock_agent):
        """Verify self.seed is set after successful call.

        Scenario: Successful call updates `self.seed`
        """
        valid_response = json.dumps({
            "domain": "Star Trek",
            "taxonomy": [{"class": "Species", "description": "Test", "subclasses": []}]
        })
        mock_agent.chat.return_value = valid_response
        ontology_empty.agent = mock_agent

        # Precondition
        assert ontology_empty.seed is None

        # Call
        result = ontology_empty.generate_initial_terms()

        # Assert
        assert ontology_empty.seed is not None
        assert ontology_empty.seed == result
        assert ontology_empty.seed["domain"] == "Star Trek"

    def test_generate_initial_terms_custom_num_classes(self, ontology_empty, mock_agent):
        """Verify num_classes parameter is passed to LLM.

        Scenario: Respects the `num_classes` parameter
        """
        valid_response = json.dumps({
            "domain": "Star Trek",
            "taxonomy": [{"class": "Class1", "description": "Test", "subclasses": []}]
        })
        mock_agent.chat.return_value = valid_response
        ontology_empty.agent = mock_agent

        # Call
        result = ontology_empty.generate_initial_terms(num_classes=3)

        # Assert
        assert result is not None
        call_args = mock_agent.chat.call_args
        prompt = call_args[1]["input"]  # Second argument is the prompt
        assert "3" in prompt, "num_classes=3 should appear in prompt"

    def test_generate_initial_terms_default_schema(self, ontology_empty, mock_agent):
        """Verify method works with default 3-level schema.

        Scenario: Works with the default 3-level schema (class → subclass → instance)
        """
        # Verify default schema is in place
        from ontogen.ontology import DEFAULT_LEVEL_SCHEMA
        assert len(ontology_empty.level_schema) == 3
        assert ontology_empty.level_schema[0].name == "class"
        assert ontology_empty.level_schema[1].name == "subclass"
        assert ontology_empty.level_schema[2].name == "instance"

        valid_response = json.dumps({
            "domain": "Star Trek",
            "taxonomy": [{
                "class": "Species",
                "description": "Species",
                "subclasses": [{
                    "class": "Vulcans",
                    "description": "Vulcans",
                    "instances": [{"term": "Spock", "description": "Spock"}]
                }]
            }]
        })
        mock_agent.chat.return_value = valid_response
        ontology_empty.agent = mock_agent

        # Call
        result = ontology_empty.generate_initial_terms()

        # Assert
        assert result is not None
        assert "taxonomy" in result

    def test_build_count_instructions_uses_level_plural_names(self, mock_agent):
        """Seed count instructions should use schema-provided plural labels."""
        custom_schema = [
            OntologyLevel(
                name="category",
                plural_name="categories",
                seed_key="category",
                children_key="analysis_groups",
            ),
            OntologyLevel(
                name="analysis_group",
                plural_name="analysis groups",
                relation_to_parent="subClassOf",
                seed_key="analysis_group",
                children_key=None,
            ),
        ]
        ontology = Ontology(
            domain="Science",
            agent=mock_agent,
            level_schema=custom_schema,
        )

        instructions = ontology._build_count_instructions(num_classes=3)

        assert "Generate exactly 3 top-level categories." in instructions
        assert "For each category, include 2-4 analysis groups." in instructions


# ============================================================================
# T2: create_seed_ontology() Tests (8 tests)
# ============================================================================


class TestCreateSeedOntology:
    """Tests for Ontology.create_seed_ontology() method."""

    def test_create_seed_ontology_success(self, ontology_with_seed):
        """Verify create_seed_ontology() builds a valid graph.

        Scenario: Happy path with valid, pre-populated seed
        """
        # Precondition: seed is populated
        assert ontology_with_seed.seed is not None

        # Call
        ontology_with_seed.create_seed_ontology()

        # Assert: Graph is populated
        assert len(ontology_with_seed.ontology_graph.nodes()) > 0
        assert len(ontology_with_seed.ontology_graph.edges()) > 0

    def test_create_seed_ontology_graph_structure(self, ontology_with_seed):
        """Verify graph structure matches seed structure.

        Scenario: Verify the graph has correct node counts and relationships

        From the seed fixture:
        - 1 top-level class: Species
        - 2 subclasses: Vulcans, Humans
        - 4 instances: Spock, T'Pol, Kirk, Janeway
        Total: 7 nodes, 6 edges (1 class→subclass, 1 class→subclass, 2 subclass→instance, 2 subclass→instance)
        """
        ontology_with_seed.create_seed_ontology()

        graph = ontology_with_seed.ontology_graph
        assert len(
            graph.nodes()) == 7, f"Expected 7 nodes, got {len(graph.nodes())}"
        assert len(
            graph.edges()) == 6, f"Expected 6 edges, got {len(graph.edges())}"

    def test_create_seed_ontology_node_attributes(self, ontology_with_seed):
        """Verify each node has required attributes.

        Scenario: Each node has required attributes: term, description, level, n_visits, total_reward
        """
        ontology_with_seed.create_seed_ontology()

        graph = ontology_with_seed.ontology_graph
        required_attrs = {"term", "description",
                          "level", "n_visits", "total_reward"}

        for node_id in graph.nodes():
            node_attrs = set(graph.nodes[node_id].keys())
            assert required_attrs.issubset(node_attrs), \
                f"Node {node_id} missing attributes. Has: {node_attrs}, needs: {required_attrs}"

            # Check types
            assert isinstance(graph.nodes[node_id]["term"], str)
            assert isinstance(graph.nodes[node_id]["description"], str)
            assert graph.nodes[node_id]["level"] in {
                "class", "subclass", "instance"}
            assert isinstance(graph.nodes[node_id]["n_visits"], int)
            assert isinstance(graph.nodes[node_id]
                              ["total_reward"], (int, float))

    def test_create_seed_ontology_edge_attributes(self, ontology_with_seed):
        """Verify each edge has a relation attribute.

        Scenario: Each edge has a "relation" attribute (subClassOf or type)
        """
        ontology_with_seed.create_seed_ontology()

        graph = ontology_with_seed.ontology_graph

        for source, target in graph.edges():
            edge_attrs = graph[source][target]
            assert "relation" in edge_attrs, f"Edge {source}→{target} missing relation attribute"
            relation = edge_attrs["relation"]
            assert relation in {"subClassOf",
                                "type"}, f"Invalid relation: {relation}"

        # Verify specific relationships
        assert graph["Species"]["Vulcans"]["relation"] == "subClassOf"
        assert graph["Species"]["Humans"]["relation"] == "subClassOf"
        assert graph["Vulcans"]["Spock"]["relation"] == "type"
        assert graph["Vulcans"]["T'Pol"]["relation"] == "type"

    def test_create_seed_ontology_duplicate_handling(self, ontology_empty):
        """Verify duplicate terms are handled gracefully.

        Scenario: Duplicate terms in seed are skipped gracefully
        """
        seed_with_duplicates = {
            "domain": "Test",
            "taxonomy": [
                {
                    "class": "TopClass",
                    "description": "Top class",
                    "subclasses": [
                        {
                            "class": "SubClass1",
                            "description": "Sub 1",
                            "instances": [
                                {"term": "Instance1", "description": "Inst 1"},
                                # Duplicate
                                {"term": "Instance1",
                                    "description": "Inst 1 duplicate"}
                            ]
                        }
                    ]
                }
            ]
        }

        ontology_empty.seed = seed_with_duplicates

        # Call
        ontology_empty.create_seed_ontology()

        graph = ontology_empty.ontology_graph
        # Should have 3 unique nodes: TopClass, SubClass1, Instance1
        assert len(
            graph.nodes()) == 3, f"Expected 3 unique nodes, got {len(graph.nodes())}"

    def test_create_seed_ontology_no_seed_error(self, ontology_empty):
        """Verify create_seed_ontology() handles missing seed gracefully.

        Scenario: Calling without seed populated returns gracefully
        """
        # Precondition: no seed
        assert ontology_empty.seed is None

        # Call (should not raise exception)
        ontology_empty.create_seed_ontology()

        # Graph should be empty
        assert len(ontology_empty.ontology_graph.nodes()) == 0

    def test_create_seed_ontology_graph_is_dag(self, ontology_with_seed):
        """Verify the ontology graph is acyclic.

        Scenario: Resulting graph is a directed acyclic graph (DAG)
        """
        ontology_with_seed.create_seed_ontology()

        graph = ontology_with_seed.ontology_graph
        assert nx.is_directed_acyclic_graph(graph), "Graph contains cycles"

    def test_create_seed_ontology_idempotent(self, ontology_with_seed):
        """Verify create_seed_ontology() can be called multiple times safely.

        Scenario: Calling multiple times is safe and idempotent
        """
        # First call
        ontology_with_seed.create_seed_ontology()
        graph1_nodes = set(ontology_with_seed.ontology_graph.nodes())
        graph1_edges = set(ontology_with_seed.ontology_graph.edges())

        # Second call
        ontology_with_seed.create_seed_ontology()
        graph2_nodes = set(ontology_with_seed.ontology_graph.nodes())
        graph2_edges = set(ontology_with_seed.ontology_graph.edges())

        # Should be identical
        assert graph1_nodes == graph2_nodes, "Graph nodes changed between calls"
        assert graph1_edges == graph2_edges, "Graph edges changed between calls"
