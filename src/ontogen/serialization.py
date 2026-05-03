"""RDF serialization helpers for the ontology pipeline."""

import logging
import re
from typing import TYPE_CHECKING, Optional

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

if TYPE_CHECKING:
    from ontogen.ontology import Ontology


logger = logging.getLogger("ontogen.ontology")


class SerializationMixin:
    """Mixin containing RDF URI sanitization and serialization logic."""

    def _sanitize_uri(self: "Ontology", term: str) -> URIRef:
        """Convert a term string to a valid RDF URI."""
        sanitized = term.replace(" ", "_")
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", sanitized)

        if not sanitized:
            logger.warning(
                "Term '%s' sanitized to empty string; using 'unknown'",
                term,
            )
            sanitized = "unknown"

        uri = self.base_namespace[sanitized]
        logger.debug("Sanitized term '%s' -> %s", term, uri)
        return uri

    def build_ontology(self: "Ontology") -> Graph:
        """Build an RDF ontology graph from the internal DiGraph representation."""
        g = Graph()
        g.bind("ex", self.base_namespace)

        def resolve_predicate(predicate_str: Optional[str]) -> Optional[URIRef]:
            """Convert a prefixed predicate string to an RDF namespace URI."""
            if not predicate_str:
                return None
            if predicate_str.startswith("rdfs:"):
                local_name = predicate_str.split(":", 1)[1]
                return getattr(RDFS, local_name)
            if predicate_str.startswith("rdf:"):
                local_name = predicate_str.split(":", 1)[1]
                return getattr(RDF, local_name)

            logger.warning(
                "Unknown predicate namespace in '%s'; using as custom URI",
                predicate_str,
            )
            return URIRef(predicate_str)

        logger.debug(
            "Building RDF ontology from DiGraph with %d nodes",
            self.ontology_graph.number_of_nodes(),
        )
        for node_id in self.ontology_graph.nodes():
            node_attrs = self.ontology_graph.nodes[node_id]
            term = node_attrs.get("term", node_id)
            level_name = node_attrs.get("level")

            if level_name:
                try:
                    level = self._get_level(level_name)
                    if getattr(level, "is_lexical", False):
                        logger.debug(
                            "Skipping lexical node '%s' during RDF serialization",
                            term,
                        )
                        continue
                except ValueError:
                    logger.warning(
                        "Level '%s' not found in schema; continuing serialization",
                        level_name,
                    )

            node_uri = self._sanitize_uri(term)
            g.add((node_uri, RDFS.label, Literal(term)))

            if level_name:
                try:
                    level = self._get_level(level_name)
                    if level.is_rdf_class:
                        g.add((node_uri, RDF.type, RDFS.Class))
                        logger.debug(
                            "Added rdf:type rdfs:Class for %s (level: %s)",
                            term,
                            level_name,
                        )
                except ValueError:
                    logger.warning(
                        "Level '%s' not found in schema; skipping rdf:type rdfs:Class",
                        level_name,
                    )

        logger.debug(
            "Building RDF edges from DiGraph with %d edges",
            self.ontology_graph.number_of_edges(),
        )
        for parent_id, child_id in self.ontology_graph.edges():
            edge_attrs = self.ontology_graph.edges[parent_id, child_id]
            relation = edge_attrs.get("relation")
            parent_attrs = self.ontology_graph.nodes[parent_id]
            parent_level_name = parent_attrs.get("level")
            child_attrs = self.ontology_graph.nodes[child_id]
            child_level_name = child_attrs.get("level")

            skip_edge = False
            for level_name in (parent_level_name, child_level_name):
                if not level_name:
                    continue
                try:
                    level = self._get_level(level_name)
                except ValueError:
                    continue
                if getattr(level, "is_lexical", False):
                    skip_edge = True
                    break
            if skip_edge:
                logger.debug(
                    "Skipping lexical edge '%s' -> '%s' during RDF serialization",
                    parent_id,
                    child_id,
                )
                continue

            parent_uri = self._sanitize_uri(
                parent_attrs.get("term", parent_id)
            )
            child_uri = self._sanitize_uri(
                self.ontology_graph.nodes[child_id].get("term", child_id)
            )

            predicate = None
            if child_level_name:
                try:
                    child_level = self._get_level(child_level_name)
                    predicate = resolve_predicate(child_level.rdf_predicate)
                except ValueError:
                    logger.warning(
                        "Could not resolve predicate for child level '%s'",
                        child_level_name,
                    )

            if not predicate and relation:
                logger.debug(
                    "No RDF predicate for relation '%s'; attempting fallback",
                    relation,
                )
                relation_to_predicate = {
                    "subClassOf": "rdfs:subClassOf",
                    "type": "rdf:type",
                }
                predicate_str = relation_to_predicate.get(
                    relation,
                    f"rdfs:{relation}",
                )
                predicate = resolve_predicate(predicate_str)

            if predicate:
                g.add((child_uri, predicate, parent_uri))
                logger.debug(
                    "Added edge: %s %s %s",
                    child_uri,
                    predicate,
                    parent_uri,
                )
            else:
                logger.warning(
                    "Could not determine RDF predicate for edge %s -> %s; skipping",
                    parent_id,
                    child_id,
                )

        self.rdf = g
        logger.info("RDF ontology built successfully: %d triples", len(g))
        return self.rdf

    def serialize_ontology(self: "Ontology", format: str = "turtle") -> str:
        """Serialize the ontology graph into the specified RDF format."""
        allowed_formats = {"turtle", "xml", "json-ld"}
        if format not in allowed_formats:
            raise ValueError(
                "Unsupported format '%s'. Must be one of: %s"
                % (format, ", ".join(sorted(allowed_formats)))
            )

        self.turtle = self.rdf.serialize(format=format)
        return self.turtle
