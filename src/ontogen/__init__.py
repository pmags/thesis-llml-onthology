"""
ontogen — LLM-driven ontology generation from domain descriptions.

This package provides tools to automatically generate RDF/OWL ontologies
from a user-specified domain using Large Language Models (LLMs).
"""

from ontogen.llm_client import ChatGpt
from ontogen.models import OntologyLevel, DEFAULT_LEVEL_SCHEMA
from ontogen.ontology import (
    Ontology,
    ExpansionRecord,
    PhaseRecord,
    GenerationHistory,
)

__all__ = [
    "ChatGpt",
    "Ontology",
    "OntologyLevel",
    "DEFAULT_LEVEL_SCHEMA",
    "ExpansionRecord",
    "PhaseRecord",
    "GenerationHistory"
]
