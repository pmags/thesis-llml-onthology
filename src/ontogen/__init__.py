"""
ontogen — LLM-driven ontology generation from domain descriptions.

This package provides tools to automatically generate RDF/OWL ontologies
from a user-specified domain using Large Language Models (LLMs).
"""

from ontogen.llm_client import ChatGpt
from ontogen.clustering import create_distance_matrix, build_similarity_graph
from ontogen.ontology import Ontology, OntologyLevel, DEFAULT_LEVEL_SCHEMA

__all__ = [
    "ChatGpt",
    "Ontology",
    "OntologyLevel",
    "DEFAULT_LEVEL_SCHEMA",
    "create_distance_matrix",
    "build_similarity_graph",
]
