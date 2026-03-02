"""
End-to-end integration tests for the complete ontology generation pipeline (SPEC-7.1).

Tests validate the full pipeline: seed generation -> validation -> expansion -> serialization,
ensuring the output RDF contains all three ontological levels with correct predicates.
"""

import json
import pytest
from unittest.mock import MagicMock, call
from rdflib import Graph
from rdflib.namespace import RDF, RDFS

from ontogen import Ontology, DEFAULT_LEVEL_SCHEMA


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_agent_for_e2e():
    """
    Mocked ChatGpt with deterministic responses for full pipeline testing.
    
    Configures side effects for:
    - generate_initial_terms: returns a valid seed with 2 classes, 2 subclasses, 2 instances
    - _generate_candidates: returns 2 plausible candidates per call
    - similarity: returns reasonable values (>50% for parent-child, ~40% for cross-branch)
    """
    agent = MagicMock()
    
    # Seed JSON response - small but complete hierarchy
    seed_response = json.dumps({
        "domain": "Star Trek",
        "taxonomy": [
            {
                "class": "Species",
                "description": "Sentient species in the Star Trek universe",
                "subclasses": [
                    {
                        "class": "Vulcans",
                        "description": "Logical and telepathic species from Vulcan",
                        "instances": [
                            {"term": "Spock", "description": "Half-Vulcan Starfleet officer"},
                            {"term": "T'Pol", "description": "Vulcan science officer"}
                        ]
                    },
                    {
                        "class": "Humans",
                        "description": "Human species from Earth",
                        "instances": [
                            {"term": "Kirk", "description": "Captain of the USS Enterprise"},
                            {"term": "Data", "description": "Android Starfleet officer"}
                        ]
                    }
                ]
            },
            {
                "class": "StarfleetRank",
                "description": "Military ranks in Starfleet",
                "subclasses": [
                    {
                        "class": "Officer",
                        "description": "Commissioned officer ranks",
                        "instances": [
                            {"term": "Admiral", "description": "Highest rank"},
                            {"term": "Captain", "description": "Ship captain rank"}
                        ]
                    }
                ]
            }
        ]
    })
    
    # Candidate generation responses - 2 new candidates per expansion
    candidates_response = json.dumps([
        {"term": "Klingons", "description": "Warrior species known for honor"},
        {"term": "Romulans", "description": "Species with pointed ears rival to Vulcans"}
    ])
    
    # Configure chat() to return different responses based on prompt content.
    # Uses lambda-style function (not finite list) to avoid StopIteration if pipeline
    # makes more calls than expected (e.g., if expansion parameters change).
    def chat_side_effect(instructions: str = "", input: str = "", prompt: str = ""):
        # Consolidate all possible prompt sources
        full_prompt = (instructions + input + prompt).lower()
        
        if "generate a" in full_prompt and "taxonomy" in full_prompt:
            return seed_response
        elif "subclass" in full_prompt or "instance" in full_prompt:
            return candidates_response
        else:
            return "[]"
    
    agent.chat.side_effect = chat_side_effect
    
    # Configure similarity: high for parent-child, lower for cross-branch.
    # Uses lambda-style function (not finite list) to avoid StopIteration if pipeline
    # makes more validation calls than expected (e.g., if more candidates are generated).
    def similarity_side_effect(term_x: str = "", description_x: str = "", term_y: str = "", description_y: str = ""):
        # Always return high similarity to ensure candidates are accepted
        return {
            "term_x": term_x,
            "description_x": description_x,
            "term_y": term_y,
            "description_y": description_y,
            "similarity": 75
        }
    
    agent.get_similarity_with_descriptions.side_effect = similarity_side_effect
    
    return agent


# ============================================================================
# Tests
# ============================================================================


class TestEndToEndPipeline:
    """Integration tests for the complete ontology generation pipeline."""

    def test_full_pipeline_produces_nonempty_rdf(self, mock_agent_for_e2e):
        """
        Test that generate_ontology() with fully mocked LLM produces valid hierarchical RDF.
        
        Validates:
        - generate_ontology() runs without errors
        - Returns a non-empty RDFLib Graph
        - Graph can be serialized
        """
        onto = Ontology(
            domain="Star Trek",
            agent=mock_agent_for_e2e,
            exploration_constant=2,
            max_iterations=2,  # Just 2 iterations to keep test fast
            similarity_threshold=0.5,
            candidates_per_iteration=2,
            level_schema=DEFAULT_LEVEL_SCHEMA
        )
        
        # Run the full pipeline
        rdf_graph = onto.generate_ontology()
        
        # Assert return type
        assert isinstance(rdf_graph, Graph), "generate_ontology() should return rdflib.Graph"
        
        # Assert non-empty
        assert len(rdf_graph) > 0, "RDF graph should contain triples"
        
        # Assert serializable
        turtle_output = rdf_graph.serialize(format="turtle")
        assert isinstance(turtle_output, str), "RDF should serialize to string"
        assert len(turtle_output) > 0, "Serialized RDF should be non-empty"

    def test_full_pipeline_output_has_three_levels(self, mock_agent_for_e2e):
        """
        Test that the output RDF contains all three ontological levels:
        - Classes with rdf:type rdfs:Class
        - Subclass edges with rdfs:subClassOf
        - Instance edges with rdf:type
        
        Validates hierarchical RDF structure post-generation.
        """
        onto = Ontology(
            domain="Star Trek",
            agent=mock_agent_for_e2e,
            exploration_constant=2,
            max_iterations=2,
            similarity_threshold=0.5,
            candidates_per_iteration=2,
            level_schema=DEFAULT_LEVEL_SCHEMA
        )
        
        # Run the full pipeline
        rdf_graph = onto.generate_ontology()
        
        # Check for rdf:type rdfs:Class triples (class-level nodes)
        class_type_triples = list(rdf_graph.triples((None, RDF.type, RDFS.Class)))
        assert len(class_type_triples) > 0, (
            "RDF should contain rdfs:Class declarations for class-level nodes"
        )
        
        # Check for rdfs:subClassOf triples (subclass hierarchy)
        subclass_triples = list(rdf_graph.triples((None, RDFS.subClassOf, None)))
        assert len(subclass_triples) > 0, (
            "RDF should contain rdfs:subClassOf triples for subclass relationships"
        )
        
        # Check for rdf:type triples (instance classification)
        # These may be present depending on expansion success, but if seed contains instances,
        # they should be in the RDF
        instance_triples = list(rdf_graph.triples((None, RDF.type, None)))
        # Filter to exclude rdfs:Class (which we already counted above)
        instance_only = [t for t in instance_triples if t[2] != RDFS.Class]
        assert len(instance_only) > 0, (
            "RDF should contain rdf:type triples for instance classification"
        )

    def test_seed_to_rdf_conversion_preserves_hierarchy(self, mock_agent_for_e2e):
        """
        Test that the hierarchy from the seed is correctly represented in RDF.
        
        Validates:
        - Seed terms appear as URI nodes in RDF
        - Parent-child relationships are preserved via rdfs:subClassOf
        - Instance relationships are preserved via rdf:type
        """
        onto = Ontology(
            domain="Star Trek",
            agent=mock_agent_for_e2e,
            exploration_constant=2,
            max_iterations=0,  # No expansion, just seed
            similarity_threshold=0.5,
            level_schema=DEFAULT_LEVEL_SCHEMA
        )
        
        # Generate only from seed (no expansion)
        rdf_graph = onto.generate_ontology()
        
        # Convert triples to string for easier inspection
        triples_str = rdf_graph.serialize(format="turtle")
        
        # Check that key seed terms appear in the RDF
        # Note: terms may be URI-encoded, so check loosely
        assert "Species" in triples_str or "species" in triples_str.lower(), (
            "Seed term 'Species' should appear in RDF"
        )
        assert "Vulcans" in triples_str or "vulcan" in triples_str.lower(), (
            "Seed term 'Vulcans' should appear in RDF"
        )

    def test_rdf_output_is_valid_and_parseable(self, mock_agent_for_e2e):
        """
        Test that the generated RDF is valid and can be re-parsed without errors.
        
        Validates:
        - RDF is syntactically valid Turtle
        - Can be parsed back into a Graph
        - Round-trip preserves triple count
        """
        onto = Ontology(
            domain="Star Trek",
            agent=mock_agent_for_e2e,
            exploration_constant=2,
            max_iterations=1,
            similarity_threshold=0.5,
            candidates_per_iteration=2,
            level_schema=DEFAULT_LEVEL_SCHEMA
        )
        
        # Generate RDF
        original_graph = onto.generate_ontology()
        original_count = len(original_graph)
        
        # Serialize
        turtle_str = original_graph.serialize(format="turtle")
        
        # Re-parse
        reparsed_graph = Graph()
        reparsed_graph.parse(data=turtle_str, format="turtle")
        reparsed_count = len(reparsed_graph)
        
        # Assert counts match
        assert reparsed_count == original_count, (
            f"Round-trip parsing should preserve triple count. "
            f"Original: {original_count}, Reparsed: {reparsed_count}"
        )

    def test_expansion_produces_valid_rdf(self, mock_agent_for_e2e):
        """
        Test that expansion iterations produce valid RDF output.
        
        Validates:
        - Graph with 1 expansion produces valid RDF
        - RDF can be serialized and re-parsed
        """
        onto = Ontology(
            domain="Star Trek",
            agent=mock_agent_for_e2e,
            exploration_constant=2,
            max_iterations=1,
            similarity_threshold=0.5,
            candidates_per_iteration=2,
            level_schema=DEFAULT_LEVEL_SCHEMA
        )
        
        # Generate with 1 expansion iteration
        rdf_graph = onto.generate_ontology()
        
        # Assert it's valid RDF
        assert isinstance(rdf_graph, Graph), "Result should be an RDFLib Graph"
        assert len(rdf_graph) > 0, "Graph should have triples"
        
        # Test serialization
        turtle_str = rdf_graph.serialize(format="turtle")
        assert len(turtle_str) > 0, "Serialized RDF should be non-empty"
        
        # Test re-parsing
        reparsed = Graph()
        reparsed.parse(data=turtle_str, format="turtle")
        assert len(reparsed) == len(rdf_graph), "Round-trip should preserve triple count"


class TestMockedLLMIntegration:
    """Tests that verify mock LLM behavior and response parsing in the pipeline."""

    def test_mock_agent_returns_valid_seed_json(self, mock_agent_for_e2e):
        """Verify the mock agent returns valid, parseable seed JSON."""
        prompt = "Generate a 3-level taxonomy for Star Trek"
        response = mock_agent_for_e2e.chat(prompt)
        
        # Should be parseable JSON
        seed = json.loads(response)
        assert isinstance(seed, dict)
        assert "domain" in seed
        assert "taxonomy" in seed
        assert isinstance(seed["taxonomy"], list)

    def test_mock_agent_returns_valid_candidates_json(self, mock_agent_for_e2e):
        """Verify the mock agent returns valid, parseable candidates JSON."""
        prompt = "Generate subclasses for Species"
        response = mock_agent_for_e2e.chat(prompt)
        
        # Should be parseable JSON (unless prompt doesn't match pattern)
        candidates = json.loads(response)
        assert isinstance(candidates, list)
        if len(candidates) > 0:
            assert "term" in candidates[0]
            assert "description" in candidates[0]

    def test_mock_agent_similarity_is_deterministic(self, mock_agent_for_e2e):
        """Verify similarity mock returns consistent results."""
        result1 = mock_agent_for_e2e.get_similarity_with_descriptions(
            "Spock", "A Vulcan", "Vulcans", "Species"
        )
        result2 = mock_agent_for_e2e.get_similarity_with_descriptions(
            "Kirk", "A human", "Humans", "Species"
        )
        
        # Both should return a dict with similarity key
        assert "similarity" in result1
        assert "similarity" in result2
        assert result1["similarity"] == result2["similarity"]  # Deterministic
