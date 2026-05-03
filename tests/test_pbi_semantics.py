"""Tests for Power BI semantic planning helpers."""

import pytest

from ontogen import Ontology, OntologyLevel
from ontogen.pbi_semantics import (
    CONTRIBUTION_OF_RELATION,
    DERIVED_FROM_RELATION,
    RELATION_FAMILY_REGISTRY,
    RelationFamilySpec,
    count_feedback_matches,
    detect_pbi_resolution_gap,
    describe_metric_binding,
    feedback_promotion_recommendation,
    find_relation_metric_matches,
    inject_pbi_semantic_relations,
    inject_pbi_proposal,
    metric_resolution_specificity,
    plan_rule_based_metric_fix,
    register_negative_feedback,
)


PBI_LEVEL_SCHEMA = [
    OntologyLevel(
        name="semantic_type",
        relation_to_parent=None,
        rdf_predicate=None,
        is_rdf_class=True,
        expandable=False,
        seed_key="type",
        children_key="concepts",
        plural_name="semantic types",
    ),
    OntologyLevel(
        name="concept",
        relation_to_parent="hasType",
        rdf_predicate="rdfs:subClassOf",
        is_rdf_class=True,
        expandable=True,
        seed_key="concept",
        children_key="expressions",
        plural_name="concepts",
    ),
    OntologyLevel(
        name="expression",
        relation_to_parent="expressedAs",
        rdf_predicate="rdf:type",
        is_rdf_class=False,
        expandable=False,
        seed_key="term",
        children_key=None,
        plural_name="expressions",
        is_lexical=True,
    ),
]


@pytest.fixture(name="semantic_graph")
def _build_semantic_graph(mock_agent):
    """Build a small Power BI-shaped ontology graph."""
    ontology = Ontology(
        domain="Professional Services Analytics",
        agent=mock_agent,
        level_schema=PBI_LEVEL_SCHEMA,
    )

    graph = ontology.ontology_graph
    semantic_types = ["Metric", "Dimension", "TimeDimension", "QueryPattern"]
    for semantic_type in semantic_types:
        graph.add_node(
            semantic_type,
            term=semantic_type,
            level="semantic_type",
            n_visits=0,
            total_reward=0.0,
        )

    metric_nodes = [
        ("Revenue", "Revenue", "[Revenue]"),
        ("Budget", "Budget", "[Budget]"),
        ("UtilizationRate", "Utilization Rate", "[Utilization Rate]"),
    ]
    for node_id, term, field in metric_nodes:
        graph.add_node(
            node_id,
            term=term,
            description=f"Metric {term}",
            level="concept",
            pbi_field=field,
            dax_expression="",
            n_visits=0,
            total_reward=0.0,
        )
        graph.add_edge("Metric", node_id, relation="has_concept")

    graph.add_node(
        "ProjectName",
        term="Project Name",
        description="Project name",
        level="concept",
        pbi_field="'Project'[Name]",
        n_visits=0,
        total_reward=0.0,
    )
    graph.add_edge("Dimension", "ProjectName", relation="has_concept")

    graph.add_node(
        "CalendarYear",
        term="Calendar Year",
        description="Fiscal year",
        level="concept",
        pbi_field="'Calendar'[Year]",
        n_visits=0,
        total_reward=0.0,
    )
    graph.add_edge("TimeDimension", "CalendarYear", relation="has_concept")

    query_patterns = [
        ("NegativeFilter", "FILTER + negative condition"),
        ("Comparison", "CALCULATE + PREVIOUSPERIOD"),
        ("Contribution", "DIVIDE + CALCULATE(ALL)"),
    ]
    for node_id, dax_template in query_patterns:
        graph.add_node(
            node_id,
            term=node_id,
            description=node_id,
            level="concept",
            dax_template=dax_template,
            n_visits=0,
            total_reward=0.0,
        )
        graph.add_edge("QueryPattern", node_id, relation="has_concept")

    expression_edges = [
        ("Revenue", "revenue"),
        ("Budget", "budget"),
        ("UtilizationRate", "below expectations"),
        ("ProjectName", "job"),
        ("CalendarYear", "year"),
        ("NegativeFilter", "below expectations"),
        ("Comparison", "year over year"),
        ("Contribution", "share of total"),
    ]
    for parent, term in expression_edges:
        if term not in graph:
            graph.add_node(
                term,
                term=term,
                description=term,
                level="expression",
                is_lexical=True,
            )
        graph.add_edge(parent, term, relation="has_expression")

    graph.add_edge("Revenue", "ProjectName", relation="sliceable_by")
    graph.add_edge("Revenue", "CalendarYear", relation="filterable_by")
    graph.add_edge("UtilizationRate", "ProjectName", relation="sliceable_by")
    graph.add_edge("UtilizationRate", "CalendarYear", relation="filterable_by")

    return ontology


@pytest.fixture(name="seed_payload")
def _build_seed_payload():
    """Minimal seed structure for proposal injection tests."""
    return {
        "taxonomy": [
            {
                "type": "Dimension",
                "concepts": [{"concept": "ProjectName"}],
            },
            {
                "type": "TimeDimension",
                "concepts": [{"concept": "CalendarYear"}],
            },
        ]
    }


def test_expectation_query_plans_budget_variance(semantic_graph):
    """Expectation-like filters should produce an actual-vs-budget variance metric."""
    intent = {
        "metrics": [{"field": "[Utilization Rate]", "concept": "UtilizationRate"}],
        "group_by": [{"field": "'Project'[Name]", "concept": "ProjectName"}],
        "time_filters": [],
        "query_patterns": [{"pattern": "NegativeFilter", "dax_template": "FILTER"}],
        "resolution_log": [],
    }
    tokens = [
        {"text": "which", "role": "intent"},
        {"text": "job", "role": "dimension"},
        {"text": "below expectations", "role": "filter"},
    ]

    proposal = plan_rule_based_metric_fix(semantic_graph, intent, tokens)

    assert proposal is not None
    assert proposal["planner_kind"] == "actual_vs_target_variance"
    assert proposal["action"] == "create_new"
    assert proposal["concept"] == "BudgetVariance"
    assert proposal["dax_expression"] == "[Revenue] - [Budget]"


def test_share_query_plans_share_of_total_metric(semantic_graph):
    """Contribution queries should generate a share-of-total metric."""
    intent = {
        "metrics": [{"field": "[Revenue]", "concept": "Revenue"}],
        "group_by": [{"field": "'Project'[Name]", "concept": "ProjectName"}],
        "time_filters": [],
        "query_patterns": [{"pattern": "Contribution", "dax_template": "DIVIDE"}],
        "resolution_log": [],
    }
    tokens = [
        {"text": "project", "role": "dimension"},
        {"text": "share of total", "role": "filter"},
    ]

    proposal = plan_rule_based_metric_fix(semantic_graph, intent, tokens)

    assert proposal is not None
    assert proposal["planner_kind"] == "share_of_total"
    assert proposal["concept"] == "RevenueShareOfTotal"
    assert proposal["dax_expression"] == (
        "DIVIDE([Revenue], CALCULATE([Revenue], ALL('Project'[Name])))"
    )


def test_comparison_query_plans_period_variance_metric(semantic_graph):
    """Comparison queries should generate a period-over-period delta metric."""
    intent = {
        "metrics": [{"field": "[Revenue]", "concept": "Revenue"}],
        "group_by": [],
        "time_filters": [{"field": "'Calendar'[Year]", "concept": "CalendarYear"}],
        "query_patterns": [{"pattern": "Comparison", "dax_template": "CALCULATE"}],
        "resolution_log": [],
    }
    tokens = [
        {"text": "revenue", "role": "metric"},
        {"text": "year over year", "role": "filter"},
    ]

    proposal = plan_rule_based_metric_fix(semantic_graph, intent, tokens)

    assert proposal is not None
    assert proposal["planner_kind"] == "period_over_period_variance"
    assert proposal["concept"] == "RevenueYearOverYearChange"
    assert proposal["dax_expression"] == (
        "[Revenue] - CALCULATE([Revenue], SAMEPERIODLASTYEAR('Calendar'[Year]))"
    )


def test_metric_specificity_prefers_derived_metrics_for_expectation_filters():
    """Derived metrics should outrank base metrics on expectation-like phrases."""
    derived = {
        "planner_kind": "actual_vs_target_variance",
        "dax_expression": "[Revenue] - [Budget]",
    }
    base = {"planner_kind": "", "dax_expression": ""}

    assert metric_resolution_specificity("below expectations", "filter", derived) < (
        metric_resolution_specificity("below expectations", "filter", base)
    )


def test_describe_metric_binding_marks_virtual_metric_when_field_not_in_model():
    """Derived metrics should translate to inline DAX references, not fake fields."""
    binding = describe_metric_binding(
        {
            "pbi_field": "[Budget Variance]",
            "dax_expression": "[Revenue] - [Budget]",
            "metric_origin": "derived",
        },
        model_metric_fields=["[Revenue]", "[Budget]", "[Utilization Rate]"],
    )

    assert binding["is_derived"] is True
    assert binding["reference"] == "[Revenue] - [Budget]"
    assert binding["label"] == "[Budget Variance] (derived)"


def test_describe_metric_binding_keeps_model_measure_reference():
    """Physical model measures should keep their original measure reference."""
    binding = describe_metric_binding(
        {
            "pbi_field": "[Revenue]",
            "dax_expression": "SUM(Financials[Amount])",
            "metric_origin": "model",
        },
        model_metric_fields=["[Revenue]", "[Budget]", "[Utilization Rate]"],
    )

    assert binding["is_derived"] is False
    assert binding["reference"] == "[Revenue]"
    assert binding["label"] == "[Revenue]"


def test_detect_gap_and_inject_crosslinks_existing_expression(
    semantic_graph,
    seed_payload,
):
    """Gap detection should propose a variance metric and injection should reuse phrases."""
    intent = {
        "metrics": [{"field": "[Utilization Rate]", "concept": "UtilizationRate"}],
        "group_by": [{"field": "'Project'[Name]", "concept": "ProjectName"}],
        "time_filters": [],
        "query_patterns": [{"pattern": "NegativeFilter", "dax_template": "FILTER"}],
        "resolution_log": [
            "  > 'below expectations' (filter) -> 'below expectations' -> 'UtilizationRate' [Metric]",
        ],
    }
    tokens = [
        {"text": "job", "role": "dimension"},
        {"text": "below expectations", "role": "filter"},
    ]

    gap = detect_pbi_resolution_gap(semantic_graph, intent, tokens)

    assert gap["has_gap"] is True
    assert gap["gap_type"] == "semantic_mismatch"
    proposal = gap["context"]["deterministic_proposal"]
    assert proposal is not None

    modified = inject_pbi_proposal(
        semantic_graph,
        proposal,
        seed_payload,
    )

    assert modified is True
    assert "BudgetVariance" in semantic_graph.ontology_graph
    assert semantic_graph.ontology_graph.has_edge("BudgetVariance", "below expectations")
    assert semantic_graph.ontology_graph.nodes["BudgetVariance"]["planner_kind"] == (
        "actual_vs_target_variance"
    )
    assert semantic_graph.ontology_graph.nodes["BudgetVariance"]["metric_origin"] == "derived"
    assert semantic_graph.ontology_graph.nodes["BudgetVariance"]["source_metrics"] == [
        "Revenue",
        "Budget",
    ]
    assert semantic_graph.ontology_graph.edges["BudgetVariance", "Revenue"]["relation"] == (
        "derived_from"
    )
    assert semantic_graph.ontology_graph.edges["BudgetVariance", "Budget"]["relation"] == (
        "derived_from"
    )
    assert semantic_graph.ontology_graph.edges[
        "BudgetVariance", "NegativeFilter"
    ]["relation"] == "supports_query_pattern"


def test_negative_feedback_stays_session_scoped_until_repeated():
    """Negative feedback should first create a session override, not a global rewrite."""
    history = []
    overrides = {}
    tokens = [
        {"text": "job", "role": "dimension"},
        {"text": "below expectations", "role": "filter"},
    ]
    resolved_intent = {
        "metrics": [{"field": "[Budget Variance]", "concept": "BudgetVariance"}],
    }

    event = register_negative_feedback(
        query_text="Which jobs are performing below expectations?",
        extracted_tokens=tokens,
        resolved_intent=resolved_intent,
        corrected_concept="UtilizationRate",
        explanation="User clarified that low utilization is the intended meaning.",
        history=history,
        overrides=overrides,
    )

    assert event["filter_terms"] == ["below expectations"]
    assert event["rejected_concepts"] == ["BudgetVariance"]
    assert overrides == {
        "below expectations": {
            "prefer_concepts": ["UtilizationRate"],
            "avoid_concepts": ["BudgetVariance"],
            "reason": "User clarified that low utilization is the intended meaning.",
        }
    }
    assert count_feedback_matches(history, "below expectations", "UtilizationRate") == 1
    assert feedback_promotion_recommendation(
        history,
        "below expectations",
        "UtilizationRate",
        promotion_threshold=2,
    ) == "Keep as session override only"

    register_negative_feedback(
        query_text="Which jobs are performing below expectations?",
        extracted_tokens=tokens,
        resolved_intent=resolved_intent,
        corrected_concept="UtilizationRate",
        explanation="Repeated correction confirms the phrase means utilization here.",
        history=history,
        overrides=overrides,
    )

    assert count_feedback_matches(history, "below expectations", "UtilizationRate") == 2
    assert feedback_promotion_recommendation(
        history,
        "below expectations",
        "UtilizationRate",
        promotion_threshold=2,
    ) == "Candidate for persistent ontology refinement"


def test_lexical_alias_nodes_are_excluded_from_rdf_serialization(semantic_graph):
    """Lexical alias nodes should support resolution without becoming ontology RDF nodes."""
    rdf_graph = semantic_graph.build_ontology()
    lexical_uri = semantic_graph.base_namespace["below_expectations"]

    assert (lexical_uri, None, None) not in rdf_graph
    assert (None, None, lexical_uri) not in rdf_graph


def test_semantic_relation_injection_adds_base_neighborhood_edges(semantic_graph):
    """Base metric neighborhoods should expose semantic relations for ontology search."""
    added = inject_pbi_semantic_relations(semantic_graph)

    assert added > 0
    assert semantic_graph.ontology_graph.edges["Revenue", "Budget"]["relation"] == (
        "actual_vs_target"
    )
    assert "comparable_over_time" in semantic_graph.ontology_graph.edges[
        "Revenue", "CalendarYear"
    ]["semantic_relations"]
    assert semantic_graph.ontology_graph.edges["Revenue", "Comparison"]["relation"] == (
        "supports_query_pattern"
    )


def test_relation_family_registry_defines_governed_semantics():
    """The registry should expose the governed relation families and type constraints."""
    expected_families = {
        "actual_vs_target",
        "derived_from",
        "comparable_over_time",
        "contribution_of",
        "supports_query_pattern",
    }

    assert set(RELATION_FAMILY_REGISTRY) == expected_families
    actual_vs_target = RELATION_FAMILY_REGISTRY["actual_vs_target"]
    assert isinstance(actual_vs_target, RelationFamilySpec)
    assert actual_vs_target.source_semantic_types == frozenset({"Metric"})
    assert actual_vs_target.target_semantic_types == frozenset({"Metric"})
    assert actual_vs_target.search_mode == "paired_derived_metric"


def test_same_edge_can_carry_multiple_registered_relation_families(
    semantic_graph,
    seed_payload,
):
    """One source-target edge should retain multiple semantic families when needed."""
    intent = {
        "metrics": [{"field": "[Revenue]", "concept": "Revenue"}],
        "group_by": [{"field": "'Project'[Name]", "concept": "ProjectName"}],
        "time_filters": [],
        "query_patterns": [{"pattern": "Contribution", "dax_template": "DIVIDE"}],
        "resolution_log": [],
    }
    tokens = [
        {"text": "project", "role": "dimension"},
        {"text": "share of total", "role": "filter"},
    ]

    proposal = plan_rule_based_metric_fix(semantic_graph, intent, tokens)
    assert proposal is not None

    inject_pbi_proposal(semantic_graph, proposal, seed_payload)
    edge_attrs = semantic_graph.ontology_graph.edges["RevenueShareOfTotal", "Revenue"]

    assert edge_attrs["relation"] == DERIVED_FROM_RELATION
    assert CONTRIBUTION_OF_RELATION in edge_attrs["semantic_relations"]


def test_relation_metric_search_reports_expectation_neighborhood_before_materialization(
    semantic_graph,
):
    """Relation search should expose actual-vs-target evidence before a derived metric exists."""
    inject_pbi_semantic_relations(semantic_graph)

    result = find_relation_metric_matches(
        semantic_graph,
        "below expectations",
        {
            "metrics": [],
            "group_by": [{"field": "'Project'[Name]", "concept": "ProjectName"}],
            "time_filters": [],
            "query_patterns": [{"pattern": "NegativeFilter", "dax_template": "FILTER"}],
        },
    )

    assert result["matches"] == []
    assert any("actual_vs_target" in message for message in result["evidence"])


def test_relation_metric_search_finds_materialized_variance_metric(
    semantic_graph,
    seed_payload,
):
    """Relation search should surface a derived metric through derived_from edges."""
    inject_pbi_semantic_relations(semantic_graph)
    intent = {
        "metrics": [],
        "group_by": [{"field": "'Project'[Name]", "concept": "ProjectName"}],
        "time_filters": [],
        "query_patterns": [{"pattern": "NegativeFilter", "dax_template": "FILTER"}],
        "resolution_log": [],
    }
    tokens = [
        {"text": "which", "role": "intent"},
        {"text": "job", "role": "dimension"},
        {"text": "below expectations", "role": "filter"},
    ]
    proposal = plan_rule_based_metric_fix(semantic_graph, intent, tokens)
    assert proposal is not None
    inject_pbi_proposal(semantic_graph, proposal, seed_payload)

    result = find_relation_metric_matches(
        semantic_graph,
        "below expectations",
        {
            "metrics": [],
            "group_by": [{"field": "'Project'[Name]", "concept": "ProjectName"}],
            "time_filters": [],
            "query_patterns": [{"pattern": "NegativeFilter", "dax_template": "FILTER"}],
        },
    )

    assert any(match["concept_node"] == "BudgetVariance" for match in result["matches"])