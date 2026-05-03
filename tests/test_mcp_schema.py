"""Tests for MCP clarification/correction/update payload builders."""

from ontogen.mcp_schema import (
    build_clarification_request,
    build_clarification_response,
    build_correction_request,
    build_correction_response,
    build_ontology_update_proposal,
)


def test_build_clarification_request_and_response():
    """Clarification payloads should preserve the ambiguous span and selection."""
    request = build_clarification_request(
        query_text="Which jobs are performing below expectations?",
        ambiguous_span="below expectations",
        role="filter",
        options=[
            {
                "concept": "BudgetVariance",
                "semantic_type": "Metric",
                "label": "[Budget Variance] (derived)",
                "confidence": 0.71,
                "rationale": "Phrase often means actual-versus-target variance.",
            },
            {
                "concept": "UtilizationRate",
                "semantic_type": "Metric",
                "label": "[Utilization Rate]",
                "confidence": 0.63,
                "rationale": "Phrase may mean operational underperformance.",
            },
        ],
    )

    response = build_clarification_response(
        query_text=request["query"],
        ambiguous_span=request["ambiguous_span"],
        selected_concept="UtilizationRate",
        apply_scope="session",
    )

    assert request["kind"] == "clarification_request"
    assert request["ambiguous_span"] == "below expectations"
    assert len(request["options"]) == 2
    assert response["kind"] == "clarification_response"
    assert response["selected_concept"] == "UtilizationRate"


def test_build_correction_payloads_extract_resolved_context():
    """Correction payloads should capture both the wrong answer and the fix."""
    intent = {
        "metrics": [{"concept": "BudgetVariance", "field": "[Budget Variance]"}],
        "group_by": [{"concept": "ProjectName", "field": "'Project'[Name]"}],
    }

    request = build_correction_request(
        query_text="Which jobs are performing below expectations?",
        resolved_intent=intent,
    )
    response = build_correction_response(
        query_text=request["query"],
        filter_terms=["below expectations"],
        rejected_concepts=request["resolved_metrics"],
        corrected_concept="UtilizationRate",
        explanation="This user means operational underperformance, not budget variance.",
    )

    assert request["resolved_metrics"] == ["BudgetVariance"]
    assert request["resolved_dimensions"] == ["ProjectName"]
    assert response["rejected_concepts"] == ["BudgetVariance"]
    assert response["apply_scope"] == "session"


def test_build_ontology_update_proposal_only_promotes_after_threshold():
    """Repeated corrections should change the recommended persistence scope."""
    proposal = build_ontology_update_proposal(
        query_text="Which jobs are performing below expectations?",
        filter_terms=["below expectations"],
        corrected_concept="UtilizationRate",
        rejected_concepts=["BudgetVariance"],
        evidence_count=2,
        promotion_threshold=3,
        rationale="Two users corrected the same ambiguous phrase.",
    )
    promoted = build_ontology_update_proposal(
        query_text="Which jobs are performing below expectations?",
        filter_terms=["below expectations"],
        corrected_concept="UtilizationRate",
        rejected_concepts=["BudgetVariance"],
        evidence_count=3,
        promotion_threshold=3,
        rationale="Repeated feedback confirms the phrase should map to utilization.",
    )

    assert proposal["recommended_scope"] == "session"
    assert proposal["change"]["change_kind"] == "add_lexical_alias"
    assert promoted["recommended_scope"] == "persistent"
    assert promoted["change"]["aliases"] == ["below expectations"]