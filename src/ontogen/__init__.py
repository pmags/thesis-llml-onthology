"""
ontogen — LLM-driven ontology generation from domain descriptions.

This package provides tools to automatically generate RDF/OWL ontologies
from a user-specified domain using Large Language Models (LLMs).
"""

from ontogen.llm_client import ChatGpt
from ontogen.models import OntologyLevel, DEFAULT_LEVEL_SCHEMA
from ontogen.mcp_schema import (
    build_clarification_request,
    build_clarification_response,
    build_correction_request,
    build_correction_response,
    build_ontology_update_proposal,
)
from ontogen.ontology import (
    Ontology,
    ExpansionRecord,
    PhaseRecord,
    GenerationHistory,
)
from ontogen.pbi_semantics import (
    RELATION_FAMILY_REGISTRY,
    RelationFamilySpec,
    count_feedback_matches,
    detect_pbi_resolution_gap,
    describe_metric_binding,
    diagnose_and_propose_pbi,
    feedback_promotion_recommendation,
    find_relation_metric_matches,
    get_relation_family_spec,
    inject_pbi_semantic_relations,
    inject_pbi_proposal,
    list_relation_family_specs,
    metric_resolution_specificity,
    plan_rule_based_metric_fix,
    register_negative_feedback,
)

__all__ = [
    "ChatGpt",
    "Ontology",
    "OntologyLevel",
    "DEFAULT_LEVEL_SCHEMA",
    "build_clarification_request",
    "build_clarification_response",
    "build_correction_request",
    "build_correction_response",
    "build_ontology_update_proposal",
    "ExpansionRecord",
    "PhaseRecord",
    "GenerationHistory",
    "RELATION_FAMILY_REGISTRY",
    "RelationFamilySpec",
    "count_feedback_matches",
    "detect_pbi_resolution_gap",
    "describe_metric_binding",
    "diagnose_and_propose_pbi",
    "feedback_promotion_recommendation",
    "find_relation_metric_matches",
    "get_relation_family_spec",
    "inject_pbi_semantic_relations",
    "inject_pbi_proposal",
    "list_relation_family_specs",
    "metric_resolution_specificity",
    "plan_rule_based_metric_fix",
    "register_negative_feedback",
]
