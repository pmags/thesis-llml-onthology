"""
Comprehensive tests for RDF serialization and URI sanitization (SPEC-4.1a & SPEC-4.2).

Tests cover:
- RDF triple mapping from DiGraph to RDF (classes, subclasses, instances)
- URI sanitization for special characters
- Serialization format support and round-trip parsing
"""

import io
import pytest
import networkx as nx
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, RDFS

from ontogen import Ontology, OntologyLevel, DEFAULT_LEVEL_SCHEMA


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_digraph():
    """Build a small 3-level DiGraph with 1 class, 1 subclass, 1 instance."""
    g = nx.DiGraph()
    
    # Class level
    g.add_node("Species", term="Species", description="Sentient species", level="class", n_visits=0, total_reward=0.0)
    
    # Subclass level  
    g.add_node("Vulcans", term="Vulcans", description="Logical, telepathic species", level="subclass", n_visits=0, total_reward=0.0)
    g.add_edge("Species", "Vulcans", relation="subClassOf")
    
    # Instance level
    g.add_node("Spock", term="Spock", description="Half-Vulcan Starfleet officer", level="instance", n_visits=0, total_reward=0.0)
    g.add_edge("Vulcans", "Spock", relation="type")
    
    return g


@pytest.fixture
def ontology_with_builtin_graph(mock_agent, sample_digraph):
    """Ontology instance with a pre-built DiGraph ready for RDF serialization."""
    ontology = Ontology(
        domain="Star Trek",
        agent=mock_agent,
        level_schema=DEFAULT_LEVEL_SCHEMA
    )
    ontology.ontology_graph = sample_digraph
    return ontology


# ============================================================================
# Test URI Sanitization (SPEC-4.1a)
# ============================================================================


class TestURISanitization:
    """Tests for _sanitize_uri() method."""

    def test_sanitize_uri_replaces_spaces(self, ontology_with_builtin_graph):
        """Spaces are replaced with underscores."""
        uri = ontology_with_builtin_graph._sanitize_uri("Star Trek Officer")
        assert "_" in str(uri)
        assert "Star_Trek_Officer" in str(uri)

    def test_sanitize_uri_removes_parentheses(self, ontology_with_builtin_graph):
        """Parentheses and their contents are removed."""
        uri = ontology_with_builtin_graph._sanitize_uri("Spock (TOS)")
        # After sanitization, "Spock_TOS" or "Spock"
        assert "Spock" in str(uri)
        assert "(" not in str(uri)
        assert ")" not in str(uri)

    def test_sanitize_uri_removes_special_chars(self, ontology_with_builtin_graph):
        """Special characters are stripped."""
        uri = ontology_with_builtin_graph._sanitize_uri("Q@#$%^&*Assistant")
        # After sanitization, only alphanumeric
        assert "QAssistant" in str(uri)
        assert "@" not in str(uri)
        assert "#" not in str(uri)

    def test_sanitize_uri_keeps_hyphens_and_underscores(self, ontology_with_builtin_graph):
        """Hyphens and underscores are preserved."""
        uri = ontology_with_builtin_graph._sanitize_uri("test_99-alpha")
        uri_str = str(uri)
        assert "test_99-alpha" in uri_str

    def test_sanitize_uri_fallback_for_empty(self, ontology_with_builtin_graph):
        """Empty sanitization falls back to 'unknown'."""
        uri = ontology_with_builtin_graph._sanitize_uri("@#$%^&*")
        assert "unknown" in str(uri)

    def test_sanitize_uri_returns_uriref(self, ontology_with_builtin_graph):
        """Return value is a valid URIRef."""
        from rdflib import URIRef
        uri = ontology_with_builtin_graph._sanitize_uri("Test")
        assert isinstance(uri, URIRef)


# ============================================================================
# Test RDF Triple Mapping (SPEC-4.1b)
# ============================================================================


class TestRDFTripleMapping:
    """Tests for build_ontology() RDF triple generation."""

    def test_classes_have_rdf_type_rdfs_class(self, ontology_with_builtin_graph):
        """Nodes marked with level='class' and is_rdf_class=True get rdf:type rdfs:Class."""
        ontology_with_builtin_graph.build_ontology()
        rdf = ontology_with_builtin_graph.rdf
        
        # Query for all triples with predicate rdf:type and object rdfs:Class
        class_triples = list(rdf.triples((None, RDF.type, RDFS.Class)))
        
        # For the default 3-level schema, both 'class' and 'subclass' levels have is_rdf_class=True
        # So we should have at least 2 triples (Species and Vulcans)
        assert len(class_triples) >= 2, f"Expected >=2 class-type triples, got {len(class_triples)}"

    def test_subclass_edges_use_rdfs_subclassof(self, ontology_with_builtin_graph):
        """Edges with relation='subClassOf' map to rdfs:subClassOf triples."""
        ontology_with_builtin_graph.build_ontology()
        rdf = ontology_with_builtin_graph.rdf
        
        # Query for all triples with predicate rdfs:subClassOf
        subclass_triples = list(rdf.triples((None, RDFS.subClassOf, None)))
        
        # We should have at least 1 (Species -> Vulcans)
        assert len(subclass_triples) >= 1, f"Expected >=1 subClassOf triple, got {len(subclass_triples)}"

    def test_instance_edges_use_rdf_type(self, ontology_with_builtin_graph):
        """Edges with relation='type' map to rdf:type triples."""
        ontology_with_builtin_graph.build_ontology()
        rdf = ontology_with_builtin_graph.rdf
        
        # Query for triples with predicate rdf:type (excluding rdfs:Class objects)
        # We're looking for instance->class type relationships
        type_triples = [
            triple for triple in rdf.triples((None, RDF.type, None))
            if triple[2] != RDFS.Class  # Exclude class-type declarations
        ]
        
        # We should have at least 1 (Spock -> Vulcans)
        assert len(type_triples) >= 1, f"Expected >=1 instance-type triple, got {len(type_triples)}"

    def test_nodes_have_rdfs_label(self, ontology_with_builtin_graph):
        """All nodes have rdfs:label triples with their term."""
        ontology_with_builtin_graph.build_ontology()
        rdf = ontology_with_builtin_graph.rdf
        
        # Query for all rdfs:label triples
        label_triples = list(rdf.triples((None, RDFS.label, None)))
        
        # We should have at least 3 (Species, Vulcans, Spock)
        assert len(label_triples) >= 3, f"Expected >=3 label triples, got {len(label_triples)}"

    def test_graph_has_expected_triple_count(self, ontology_with_builtin_graph):
        """RDF graph has expected number of triples."""
        ontology_with_builtin_graph.build_ontology()
        rdf = ontology_with_builtin_graph.rdf
        
        # For our 3-node graph with standard schema:
        # - 2 rdf:type rdfs:Class triples (Species, Vulcans)
        # - 1 rdfs:subClassOf triple (Vulcans subClassOf Species)
        # - 1 rdf:type triple (Spock type Vulcans)
        # - 3 rdfs:label triples (Species, Vulcans, Spock)
        # Total: 7 triples
        triple_count = len(rdf)
        assert triple_count >= 7, f"Expected >=7 triples, got {triple_count}"


# ============================================================================
# Test Serialization Format Support (SPEC-4.2)
# ============================================================================


class TestSerializationFormats:
    """Tests for serialize_ontology() format support."""

    def test_serialize_turtle_format(self, ontology_with_builtin_graph):
        """Turtle format serialization works."""
        ontology_with_builtin_graph.build_ontology()
        output = ontology_with_builtin_graph.serialize_ontology(format="turtle")
        
        # Turtle output should be a non-empty string
        assert isinstance(output, str)
        assert len(output) > 0
        # Turtle format typically contains @prefix declarations
        assert "@prefix" in output or "rdf:" in output

    def test_serialize_xml_format(self, ontology_with_builtin_graph):
        """XML format serialization works."""
        ontology_with_builtin_graph.build_ontology()
        output = ontology_with_builtin_graph.serialize_ontology(format="xml")
        
        assert isinstance(output, str)
        assert len(output) > 0
        # XML should contain RDF-XML tags
        assert "rdf:RDF" in output or "<rdf" in output or "RDF" in output

    def test_serialize_jsonld_format(self, ontology_with_builtin_graph):
        """JSON-LD format serialization works."""
        ontology_with_builtin_graph.build_ontology()
        output = ontology_with_builtin_graph.serialize_ontology(format="json-ld")
        
        assert isinstance(output, str)
        assert len(output) > 0

    def test_serialize_default_is_turtle(self, ontology_with_builtin_graph):
        """Default format is turtle."""
        ontology_with_builtin_graph.build_ontology()
        default_output = ontology_with_builtin_graph.serialize_ontology()
        turtle_output = ontology_with_builtin_graph.serialize_ontology(format="turtle")
        
        # Both should be identical (same object reference or same content)
        assert default_output == turtle_output

    def test_serialize_invalid_format_raises_error(self, ontology_with_builtin_graph):
        """Invalid format raises ValueError."""
        ontology_with_builtin_graph.build_ontology()
        
        with pytest.raises(ValueError) as exc_info:
            ontology_with_builtin_graph.serialize_ontology(format="invalid_format")
        
        assert "Unsupported format" in str(exc_info.value)
        assert "invalid_format" in str(exc_info.value)


# ============================================================================
# Test Round-Trip Parsing (SPEC-4.2)
# ============================================================================


class TestRoundTripParsing:
    """Tests for serializing and re-parsing the ontology."""

    def test_turtle_output_is_valid(self, ontology_with_builtin_graph):
        """Serialized Turtle can be parsed back without errors."""
        ontology_with_builtin_graph.build_ontology()
        turtle_output = ontology_with_builtin_graph.serialize_ontology(format="turtle")
        
        # Parse the Turtle output
        parsed_graph = Graph()
        try:
            parsed_graph.parse(data=turtle_output, format="turtle")
        except Exception as e:
            pytest.fail(f"Failed to parse serialized Turtle: {e}")
        
        # Check that we got triples back
        assert len(parsed_graph) > 0

    def test_round_trip_preserves_triple_count(self, ontology_with_builtin_graph):
        """Round-trip serialization and parsing preserves triple count."""
        ontology_with_builtin_graph.build_ontology()
        original_count = len(ontology_with_builtin_graph.rdf)
        
        # Serialize and reparse
        turtle_output = ontology_with_builtin_graph.serialize_ontology(format="turtle")
        parsed_graph = Graph()
        parsed_graph.parse(data=turtle_output, format="turtle")
        
        # Triple counts should match (or be very close due to formatting differences)
        assert len(parsed_graph) == original_count, \
            f"Expected {original_count} triples after round-trip, got {len(parsed_graph)}"

    def test_round_trip_preserves_class_relationships(self, ontology_with_builtin_graph):
        """Round-trip preserves class and subclass relationships."""
        ontology_with_builtin_graph.build_ontology()
        original_graph = ontology_with_builtin_graph.rdf
        
        # Serialize and reparse
        turtle_output = ontology_with_builtin_graph.serialize_ontology(format="turtle")
        parsed_graph = Graph()
        parsed_graph.parse(data=turtle_output, format="turtle")
        
        # Check that rdfs:Class relationships are preserved
        original_class_triples = list(original_graph.triples((None, RDF.type, RDFS.Class)))
        parsed_class_triples = list(parsed_graph.triples((None, RDF.type, RDFS.Class)))
        
        assert len(parsed_class_triples) == len(original_class_triples), \
            f"Expected {len(original_class_triples)} class triples after round-trip, " \
            f"got {len(parsed_class_triples)}"

    def test_round_trip_preserves_subclass_relationships(self, ontology_with_builtin_graph):
        """Round-trip preserves subClassOf relationships."""
        ontology_with_builtin_graph.build_ontology()
        original_graph = ontology_with_builtin_graph.rdf
        
        # Serialize and reparse
        turtle_output = ontology_with_builtin_graph.serialize_ontology(format="turtle")
        parsed_graph = Graph()
        parsed_graph.parse(data=turtle_output, format="turtle")
        
        # Check that rdfs:subClassOf relationships are preserved
        original_subclass_triples = list(original_graph.triples((None, RDFS.subClassOf, None)))
        parsed_subclass_triples = list(parsed_graph.triples((None, RDFS.subClassOf, None)))
        
        assert len(parsed_subclass_triples) == len(original_subclass_triples), \
            f"Expected {len(original_subclass_triples)} subClassOf triples after round-trip, " \
            f"got {len(parsed_subclass_triples)}"

    def test_round_trip_preserves_instance_relationships(self, ontology_with_builtin_graph):
        """Round-trip preserves rdf:type instance relationships."""
        ontology_with_builtin_graph.build_ontology()
        original_graph = ontology_with_builtin_graph.rdf
        
        # Serialize and reparse
        turtle_output = ontology_with_builtin_graph.serialize_ontology(format="turtle")
        parsed_graph = Graph()
        parsed_graph.parse(data=turtle_output, format="turtle")
        
        # Check that rdf:type instance relationships are preserved (excluding class declarations)
        original_instance_triples = [
            triple for triple in original_graph.triples((None, RDF.type, None))
            if triple[2] != RDFS.Class
        ]
        parsed_instance_triples = [
            triple for triple in parsed_graph.triples((None, RDF.type, None))
            if triple[2] != RDFS.Class
        ]
        
        assert len(parsed_instance_triples) == len(original_instance_triples), \
            f"Expected {len(original_instance_triples)} instance-type triples after round-trip, " \
            f"got {len(parsed_instance_triples)}"
