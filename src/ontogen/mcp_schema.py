"""Typed MCP interaction schemas for clarification and ontology relearning.

These payloads define the minimal structured interactions needed by an MCP
bridge around ontology-backed resolution:

- clarification when a phrase is ambiguous before execution
- correction after a wrong answer was produced
- ontology-update proposals once repeated evidence justifies persistence
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, TypedDict


class ConceptOption(TypedDict):
    """One candidate concept offered to a user during clarification."""

    concept: str
    semantic_type: str
    label: str
    confidence: float
    rationale: str


class ClarificationRequest(TypedDict):
    """Request asking the client to disambiguate a phrase before execution."""

    kind: Literal["clarification_request"]
    query: str
    ambiguous_span: str
    role: str
    question: str
    options: List[ConceptOption]
    allow_free_text: bool


class ClarificationResponse(TypedDict):
    """User response selecting the intended concept for an ambiguous phrase."""

    kind: Literal["clarification_response"]
    query: str
    ambiguous_span: str
    selected_concept: str
    apply_scope: Literal["session", "persistent"]
    free_text: str


class CorrectionRequest(TypedDict):
    """Request asking the client to correct a wrong resolution after execution."""

    kind: Literal["correction_request"]
    query: str
    resolved_metrics: List[str]
    resolved_dimensions: List[str]
    message: str


class CorrectionResponse(TypedDict):
    """Structured negative feedback returned by the client."""

    kind: Literal["correction_response"]
    query: str
    filter_terms: List[str]
    rejected_concepts: List[str]
    corrected_concept: str
    explanation: str
    apply_scope: Literal["session", "persistent"]


class OntologyChange(TypedDict, total=False):
    """Concrete ontology mutation proposed after repeated evidence."""

    change_kind: Literal[
        "add_lexical_alias",
        "reweight_resolution",
        "create_derived_metric",
        "map_existing_metric",
    ]
    target_concept: str
    aliases: List[str]
    avoid_concepts: List[str]
    dax_expression: str
    metadata: Dict[str, Any]


class OntologyUpdateProposal(TypedDict):
    """Proposal emitted when feedback is strong enough for persistent refinement."""

    kind: Literal["ontology_update_proposal"]
    query: str
    trigger: Literal["clarification", "correction", "repeated_feedback"]
    evidence_count: int
    promotion_threshold: int
    recommended_scope: Literal["session", "persistent"]
    rationale: str
    change: OntologyChange


class OntologyUpdateDecision(TypedDict):
    """Client-side approval or rejection of a proposed persistent update."""

    kind: Literal["ontology_update_decision"]
    accepted: bool
    apply_scope: Literal["session", "persistent", "rejected"]
    reason: str


def build_clarification_request(
    query_text: str,
    ambiguous_span: str,
    role: str,
    options: Sequence[ConceptOption],
    question: Optional[str] = None,
    allow_free_text: bool = True,
) -> ClarificationRequest:
    """Build a canonical clarification payload for an ambiguous phrase."""
    if not options:
        raise ValueError("Clarification requests require at least one candidate option")

    return {
        "kind": "clarification_request",
        "query": query_text,
        "ambiguous_span": ambiguous_span,
        "role": role,
        "question": question
        or f"When you said '{ambiguous_span}', which meaning did you intend?",
        "options": list(options),
        "allow_free_text": allow_free_text,
    }


def build_clarification_response(
    query_text: str,
    ambiguous_span: str,
    selected_concept: str,
    *,
    apply_scope: Literal["session", "persistent"] = "session",
    free_text: str = "",
) -> ClarificationResponse:
    """Build the structured client response to a clarification request."""
    return {
        "kind": "clarification_response",
        "query": query_text,
        "ambiguous_span": ambiguous_span,
        "selected_concept": selected_concept,
        "apply_scope": apply_scope,
        "free_text": free_text,
    }


def build_correction_request(
    query_text: str,
    resolved_intent: Mapping[str, Any],
    message: Optional[str] = None,
) -> CorrectionRequest:
    """Build a correction prompt after the system produced a wrong answer."""
    resolved_metrics = [
        metric.get("concept", "")
        for metric in resolved_intent.get("metrics", [])
        if metric.get("concept", "")
    ]
    resolved_dimensions = [
        dimension.get("concept", "")
        for dimension in resolved_intent.get("group_by", [])
        if dimension.get("concept", "")
    ]
    return {
        "kind": "correction_request",
        "query": query_text,
        "resolved_metrics": resolved_metrics,
        "resolved_dimensions": resolved_dimensions,
        "message": message or "That result looks wrong. What concept did you intend?",
    }


def build_correction_response(
    query_text: str,
    filter_terms: Sequence[str],
    rejected_concepts: Sequence[str],
    corrected_concept: str,
    explanation: str,
    *,
    apply_scope: Literal["session", "persistent"] = "session",
) -> CorrectionResponse:
    """Build structured negative feedback from a user correction."""
    return {
        "kind": "correction_response",
        "query": query_text,
        "filter_terms": [term.strip().lower() for term in filter_terms if term.strip()],
        "rejected_concepts": [concept for concept in rejected_concepts if concept],
        "corrected_concept": corrected_concept,
        "explanation": explanation,
        "apply_scope": apply_scope,
    }


def build_ontology_update_proposal(
    query_text: str,
    filter_terms: Sequence[str],
    corrected_concept: str,
    rejected_concepts: Sequence[str],
    evidence_count: int,
    promotion_threshold: int,
    rationale: str,
    *,
    change_kind: Literal[
        "add_lexical_alias",
        "reweight_resolution",
        "create_derived_metric",
        "map_existing_metric",
    ] = "add_lexical_alias",
    metadata: Optional[Mapping[str, Any]] = None,
    dax_expression: str = "",
) -> OntologyUpdateProposal:
    """Build a proposal for a persistent ontology update.

    The recommended scope remains session-scoped until the promotion threshold
    is met. This lets callers surface the proposal before mutating the
    persistent ontology.
    """
    aliases = [term.strip().lower() for term in filter_terms if term.strip()]
    recommended_scope: Literal["session", "persistent"]
    recommended_scope = (
        "persistent" if evidence_count >= promotion_threshold else "session"
    )
    change: OntologyChange = {
        "change_kind": change_kind,
        "target_concept": corrected_concept,
        "aliases": aliases,
        "avoid_concepts": [concept for concept in rejected_concepts if concept],
        "metadata": dict(metadata or {}),
    }
    if dax_expression:
        change["dax_expression"] = dax_expression

    return {
        "kind": "ontology_update_proposal",
        "query": query_text,
        "trigger": "repeated_feedback",
        "evidence_count": evidence_count,
        "promotion_threshold": promotion_threshold,
        "recommended_scope": recommended_scope,
        "rationale": rationale,
        "change": change,
    }