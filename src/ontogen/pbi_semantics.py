"""Power BI semantic planning and query-repair helpers.

This module centralizes the Power BI-specific resolution heuristics that were
previously embedded in the experiment notebook. The helpers are designed to be
reused by future MCP-facing flows so notebook experiments and production code
share the same repair logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypedDict,
)

if TYPE_CHECKING:
    from ontogen.ontology import Ontology


logger = logging.getLogger(__name__)


EXPECTATION_TERMS = frozenset(
    {
        "expectation",
        "expectations",
        "budget",
        "target",
        "targets",
        "plan",
        "planned",
    }
)
SHARE_TERMS = frozenset(
    {
        "share",
        "share of total",
        "percentage of total",
        "percent of total",
        "contribution",
        "mix",
        "portion of total",
        "how much of the total",
    }
)
COMPARISON_TERMS = frozenset(
    {
        "compare",
        "comparison",
        "versus",
        "vs",
        "change",
        "delta",
        "difference",
        "variance",
        "year over year",
        "yoy",
        "quarter over quarter",
        "qoq",
        "month over month",
        "mom",
        "last year",
        "last quarter",
        "last month",
        "previous period",
        "prior period",
    }
)

ACTUAL_METRIC_ALIASES: Sequence[Sequence[str]] = (
    ("actual", "actuals"),
    ("revenue", "sales", "income", "amount"),
)
TARGET_METRIC_ALIASES: Sequence[Sequence[str]] = (
    ("budget", "budgeted amount"),
    ("target", "targets", "goal", "quota"),
    ("plan", "planned"),
    ("forecast",),
)


class MetricConcept(TypedDict):
    """Metric concept metadata projected from the ontology graph."""

    concept: str
    term: str
    pbi_field: str
    description: str
    dax_expression: str
    expressions: List[str]
    metric_origin: str


class MetricBinding(TypedDict):
    """Resolved metric binding metadata for query translation."""

    field: str
    reference: str
    dax_expression: str
    is_derived: bool
    label: str


class RepairProposal(TypedDict, total=False):
    """Structured ontology repair proposal."""

    action: str
    concept: str
    pbi_field: str
    dax_expression: str
    expressions: List[str]
    reasoning: str
    diagnosed_by: str
    planner_kind: str
    expected_metric_concepts: List[str]
    source_metrics: List[str]


class GapContext(TypedDict, total=False):
    """Extra context collected while diagnosing a resolution gap."""

    available_metrics: List[MetricConcept]
    patterns_matched: List[str]
    resolved_metrics: List[Dict[str, str]]
    deterministic_proposal: Optional[RepairProposal]


class ResolutionGap(TypedDict):
    """Structured gap diagnosis payload."""

    has_gap: bool
    gap_type: Optional[str]
    unresolved_tokens: List[str]
    mismatched_filters: List[Dict[str, str]]
    context: GapContext


class FeedbackOverride(TypedDict):
    """Session-scoped correction hints for ambiguous phrases."""

    prefer_concepts: List[str]
    avoid_concepts: List[str]
    reason: str


class FeedbackEvent(TypedDict):
    """Structured negative feedback captured from a rejected answer."""

    query: str
    feedback: str
    filter_terms: List[str]
    rejected_concepts: List[str]
    corrected_concept: str
    explanation: str
    scope: str


class RelationMetricMatch(TypedDict):
    """Metric candidate surfaced from relation-aware ontology search."""

    expression_node: str
    concept_node: str
    concept_attrs: Dict[str, Any]
    path: List[str]
    relation_reason: str


class RelationSearchResult(TypedDict):
    """Relation-aware metric search candidates and supporting evidence."""

    matches: List[RelationMetricMatch]
    evidence: List[str]


ACTUAL_VS_TARGET_RELATION = "actual_vs_target"
DERIVED_FROM_RELATION = "derived_from"
COMPARABLE_OVER_TIME_RELATION = "comparable_over_time"
CONTRIBUTION_OF_RELATION = "contribution_of"
SUPPORTS_QUERY_PATTERN_RELATION = "supports_query_pattern"


@dataclass(frozen=True)
class RelationFamilySpec:
    """Governed metadata describing one supported semantic relation family."""

    name: str
    description: str
    source_semantic_types: FrozenSet[str]
    target_semantic_types: FrozenSet[str]
    category: Literal["ontology", "planner"]
    creation_modes: FrozenSet[str]
    search_mode: Literal["none", "paired_derived_metric", "direct_metric"] = "none"
    trigger_terms: FrozenSet[str] = frozenset()
    trigger_patterns: FrozenSet[str] = frozenset()
    required_planner_kind: str = ""


def _build_relation_family_registry() -> Dict[str, RelationFamilySpec]:
    """Build the governed relation-family registry for Power BI semantics."""
    families = [
        RelationFamilySpec(
            name=ACTUAL_VS_TARGET_RELATION,
            description="Links an actual metric to its budget, target, or plan counterpart.",
            source_semantic_types=frozenset({"Metric"}),
            target_semantic_types=frozenset({"Metric"}),
            category="ontology",
            creation_modes=frozenset({"planner", "expansion"}),
            search_mode="paired_derived_metric",
            trigger_terms=frozenset(EXPECTATION_TERMS),
        ),
        RelationFamilySpec(
            name=DERIVED_FROM_RELATION,
            description="Links a derived metric to one of its source metrics.",
            source_semantic_types=frozenset({"Metric"}),
            target_semantic_types=frozenset({"Metric"}),
            category="planner",
            creation_modes=frozenset({"planner"}),
        ),
        RelationFamilySpec(
            name=COMPARABLE_OVER_TIME_RELATION,
            description="Links a metric to a time dimension that supports period comparisons.",
            source_semantic_types=frozenset({"Metric"}),
            target_semantic_types=frozenset({"TimeDimension"}),
            category="ontology",
            creation_modes=frozenset({"planner", "expansion"}),
            search_mode="direct_metric",
            trigger_terms=frozenset(COMPARISON_TERMS),
            trigger_patterns=frozenset({"Comparison"}),
            required_planner_kind="period_over_period_variance",
        ),
        RelationFamilySpec(
            name=CONTRIBUTION_OF_RELATION,
            description="Links a contribution metric to the base metric it expresses as a share.",
            source_semantic_types=frozenset({"Metric"}),
            target_semantic_types=frozenset({"Metric"}),
            category="ontology",
            creation_modes=frozenset({"planner", "expansion"}),
            search_mode="direct_metric",
            trigger_terms=frozenset(SHARE_TERMS),
            trigger_patterns=frozenset({"Contribution"}),
        ),
        RelationFamilySpec(
            name=SUPPORTS_QUERY_PATTERN_RELATION,
            description="Links a metric to a query-pattern concept it can satisfy.",
            source_semantic_types=frozenset({"Metric"}),
            target_semantic_types=frozenset({"QueryPattern"}),
            category="planner",
            creation_modes=frozenset({"planner", "expansion"}),
        ),
    ]
    return {family.name: family for family in families}


RELATION_FAMILY_REGISTRY = _build_relation_family_registry()


def get_relation_family_spec(relation: str) -> RelationFamilySpec:
    """Return the governed definition for a registered relation family."""
    try:
        return RELATION_FAMILY_REGISTRY[relation]
    except KeyError as exc:
        raise ValueError(f"Unregistered relation family: {relation}") from exc


def list_relation_family_specs() -> List[RelationFamilySpec]:
    """Return the registered semantic relation families in stable order."""
    return list(RELATION_FAMILY_REGISTRY.values())


def normalize_metric_key(value: str) -> str:
    """Normalize concept, term, and field names for loose matching."""
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def sanitize_concept_id(value: str) -> str:
    """Build a stable concept identifier while preserving existing capitals."""
    compact = re.sub(r"\s+", "", value or "")
    return re.sub(r"[^A-Za-z0-9]", "", compact)


def clean_display_name(value: str) -> str:
    """Convert a field or concept label into a display-friendly name."""
    stripped = (value or "").replace("[", " ").replace("]", " ")
    stripped = stripped.replace("'", " ")
    stripped = re.sub(r"[^A-Za-z0-9]+", " ", stripped)
    return " ".join(part for part in stripped.split() if part)


def _joined_token_text(tokens: Sequence[Mapping[str, str]]) -> str:
    return " ".join(token.get("text", "").strip().lower() for token in tokens)


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    """Return unique non-empty values while preserving their first occurrence."""
    seen = set()
    result: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _filter_terms_from_tokens(tokens: Sequence[Mapping[str, str]]) -> List[str]:
    """Extract normalized filter phrases from resolved query tokens."""
    return _dedupe_preserve_order(
        [
            token.get("text", "").strip().lower()
            for token in tokens
            if token.get("role") == "filter"
        ]
    )


def _contains_any_phrase(text: str, phrases: Sequence[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _edge_has_relation(edge_attrs: Mapping[str, Any], relation: str) -> bool:
    """Return whether an edge carries the requested relation semantics."""
    if edge_attrs.get("relation") == relation:
        return True
    semantic_relations = edge_attrs.get("semantic_relations", []) or []
    return relation in semantic_relations


def _attach_semantic_relation(
    onto: "Ontology",
    source: str,
    target: str,
    relation: str,
) -> int:
    """Attach a governed semantic relation without clobbering an existing DiGraph edge relation."""
    get_relation_family_spec(relation)
    if source not in onto.ontology_graph or target not in onto.ontology_graph:
        return 0

    if not _relation_allows_nodes(onto, relation, source, target):
        logger.debug(
            "Skipping semantic relation %s between %s and %s due to registry type constraints",
            relation,
            source,
            target,
        )
        return 0

    if onto.ontology_graph.has_edge(source, target):
        edge_attrs = onto.ontology_graph.edges[source, target]
        if _edge_has_relation(edge_attrs, relation):
            return 0
        semantic_relations = list(edge_attrs.get("semantic_relations", []) or [])
        semantic_relations.append(relation)
        edge_attrs["semantic_relations"] = _dedupe_preserve_order(semantic_relations)
        return 1

    onto.ontology_graph.add_edge(
        source,
        target,
        relation=relation,
        semantic_relations=[relation],
    )
    return 1


def _neighbors_by_relation(
    onto: "Ontology",
    node_id: str,
    relation: str,
    *,
    direction: str = "outgoing",
) -> List[str]:
    """Return neighboring nodes connected by the given relation semantics."""
    neighbors: List[str] = []

    if direction in ("outgoing", "both"):
        for _, target, edge_attrs in onto.ontology_graph.out_edges(node_id, data=True):
            if _edge_has_relation(edge_attrs, relation):
                neighbors.append(target)

    if direction in ("incoming", "both"):
        for source, _, edge_attrs in onto.ontology_graph.in_edges(node_id, data=True):
            if _edge_has_relation(edge_attrs, relation):
                neighbors.append(source)

    return _dedupe_preserve_order(neighbors)


def _relation_allows_nodes(
    onto: "Ontology",
    relation: str,
    source: str,
    target: str,
) -> bool:
    """Return whether the relation registry allows this source-target pairing."""
    spec = get_relation_family_spec(relation)
    source_types = _concept_semantic_types(onto, source)
    target_types = _concept_semantic_types(onto, target)
    return bool(source_types & spec.source_semantic_types) and bool(
        target_types & spec.target_semantic_types
    )


def _iter_edges_by_relation(
    onto: "Ontology",
    relation: str,
) -> List[tuple[str, str, Mapping[str, Any]]]:
    """Return graph edges carrying the requested relation semantics."""
    matching_edges: List[tuple[str, str, Mapping[str, Any]]] = []
    for source, target, edge_attrs in onto.ontology_graph.edges(data=True):
        if _edge_has_relation(edge_attrs, relation):
            matching_edges.append((source, target, edge_attrs))
    return matching_edges


def _relation_family_is_triggered(
    relation_spec: RelationFamilySpec,
    token_text: str,
    pattern_names: FrozenSet[str],
) -> bool:
    """Return whether a relation family should participate in the current search."""
    has_term_trigger = bool(relation_spec.trigger_terms) and _contains_any_phrase(
        token_text,
        sorted(relation_spec.trigger_terms),
    )
    has_pattern_trigger = bool(relation_spec.trigger_patterns & pattern_names)
    return has_term_trigger or has_pattern_trigger


def _triggered_relation_family_specs(
    token_text: str,
    pattern_names: FrozenSet[str],
) -> List[RelationFamilySpec]:
    """Return relation families that are relevant for the current query slice."""
    return [
        relation_spec
        for relation_spec in list_relation_family_specs()
        if relation_spec.search_mode != "none"
        and _relation_family_is_triggered(relation_spec, token_text, pattern_names)
    ]


def _collect_paired_derived_metric_matches(
    onto: "Ontology",
    relation_spec: RelationFamilySpec,
    result: RelationSearchResult,
    add_match: Any,
) -> None:
    """Collect metrics inferred from a paired relation plus derived-from evidence."""
    relation_name = relation_spec.name
    relation_edges = _iter_edges_by_relation(onto, relation_name)
    if not relation_edges:
        result["evidence"].append(
            f"no {relation_name} relation is available in the ontology neighborhood"
        )
        return

    for source, target, _ in relation_edges:
        derived_candidates = [
            metric["concept"]
            for metric in list_metric_concepts(onto)
            if source in _neighbors_by_relation(
                onto,
                metric["concept"],
                DERIVED_FROM_RELATION,
            )
            and target in _neighbors_by_relation(
                onto,
                metric["concept"],
                DERIVED_FROM_RELATION,
            )
        ]
        if derived_candidates:
            result["evidence"].append(
                f"relation {relation_name} links {source} to {target}"
            )
            for metric_node in derived_candidates:
                add_match(
                    metric_node,
                    f"{relation_name}:{source}->{target}",
                    [source, target, metric_node],
                )
            continue

        result["evidence"].append(
            f"relation {relation_name} links {source} to {target}, but no derived metric is materialized yet"
        )


def _collect_direct_metric_relation_matches(
    onto: "Ontology",
    relation_spec: RelationFamilySpec,
    result: RelationSearchResult,
    add_match: Any,
) -> None:
    """Collect metrics surfaced directly from a registered relation family."""
    relation_name = relation_spec.name
    found_relation = False

    for metric in list_metric_concepts(onto):
        neighbors = _neighbors_by_relation(onto, metric["concept"], relation_name)
        if not neighbors:
            continue
        found_relation = True
        result["evidence"].append(
            f"relation {relation_name} links {metric['concept']} to {', '.join(neighbors)}"
        )
        required_planner_kind = relation_spec.required_planner_kind
        if required_planner_kind and onto.ontology_graph.nodes[metric["concept"]].get(
            "planner_kind"
        ) != required_planner_kind:
            continue
        add_match(
            metric["concept"],
            f"{relation_name}:{metric['concept']}",
            [metric["concept"], *neighbors],
        )

    if not found_relation:
        result["evidence"].append(
            f"no {relation_name} relation is available in the ontology neighborhood"
        )


def _lexical_level_names(onto: "Ontology") -> frozenset[str]:
    """Return schema levels that behave as lexical alias mappings."""
    lexical_levels = [
        level.name for level in onto.level_schema if getattr(level, "is_lexical", False)
    ]
    if lexical_levels:
        return frozenset(lexical_levels)
    return frozenset(level.name for level in onto.level_schema if not level.expandable)


def _primary_lexical_level_name(onto: "Ontology") -> str:
    """Return the canonical graph level name for lexical alias nodes."""
    for level in onto.level_schema:
        if getattr(level, "is_lexical", False):
            return level.name
    for level in onto.level_schema:
        if not level.expandable:
            return level.name
    return "expression"


def _concept_semantic_types(onto: "Ontology", node_id: str) -> set[str]:
    semantic_types = set()
    for parent in onto.ontology_graph.predecessors(node_id):
        parent_attrs = onto.ontology_graph.nodes[parent]
        if parent_attrs.get("level") == "semantic_type":
            semantic_types.add(parent_attrs.get("term", parent))
    return semantic_types


def _list_concepts_for_semantic_type(
    onto: "Ontology",
    semantic_type: str,
) -> List[Dict[str, str]]:
    concepts: List[Dict[str, str]] = []
    for node_id, attrs in onto.ontology_graph.nodes(data=True):
        if attrs.get("level") != "concept":
            continue
        if semantic_type not in _concept_semantic_types(onto, node_id):
            continue
        concepts.append(
            {
                "concept": node_id,
                "term": attrs.get("term", node_id),
                "pbi_field": attrs.get("pbi_field", ""),
                "description": attrs.get("description", ""),
                "dax_expression": attrs.get("dax_expression", ""),
            }
        )
    return concepts


def list_metric_concepts(onto: "Ontology") -> List[MetricConcept]:
    """Return metric concept metadata from the ontology graph."""
    metrics: List[MetricConcept] = []
    lexical_levels = _lexical_level_names(onto)
    for node_id, attrs in onto.ontology_graph.nodes(data=True):
        if attrs.get("level") != "concept":
            continue
        if "Metric" not in _concept_semantic_types(onto, node_id):
            continue

        expressions = sorted(
            onto.ontology_graph.nodes[child].get("term", child)
            for child in onto.ontology_graph.successors(node_id)
            if onto.ontology_graph.nodes[child].get("level") in lexical_levels
        )
        metrics.append(
            MetricConcept(
                concept=node_id,
                term=attrs.get("term", node_id),
                pbi_field=attrs.get("pbi_field", ""),
                description=attrs.get("description", ""),
                dax_expression=attrs.get("dax_expression", ""),
                expressions=expressions,
                metric_origin=attrs.get("metric_origin", "model"),
            )
        )
    return metrics


def describe_metric_binding(
    concept_attrs: Mapping[str, Any],
    model_metric_fields: Optional[Sequence[str]] = None,
) -> MetricBinding:
    """Describe whether a resolved metric is physical or derived.

    Args:
        concept_attrs: Attributes from the resolved concept node.
        model_metric_fields: Physical metric fields that exist in the semantic
            model. If omitted, the helper falls back to the node metadata.

    Returns:
        A dict containing the display field, the executable DAX reference, and
        a derived/non-derived flag.
    """
    field = concept_attrs.get("pbi_field", "") or ""
    dax_expression = concept_attrs.get("dax_expression", "") or ""
    metric_origin = concept_attrs.get("metric_origin", "") or ""
    known_fields = set(model_metric_fields or [])

    is_derived = metric_origin == "derived"
    if not is_derived and dax_expression and field and known_fields:
        is_derived = field not in known_fields
    if not is_derived and dax_expression and not field:
        is_derived = True

    reference = dax_expression if is_derived and dax_expression else field
    label = f"{field} (derived)" if is_derived and field else field
    return MetricBinding(
        field=field,
        reference=reference,
        dax_expression=dax_expression,
        is_derived=is_derived,
        label=label,
    )


def find_metric_by_alias(
    onto: "Ontology",
    aliases: Sequence[str],
) -> Optional[MetricConcept]:
    """Find a metric concept by concept id, display term, or field alias."""
    alias_keys = {normalize_metric_key(alias) for alias in aliases if alias}
    if not alias_keys:
        return None

    for metric in list_metric_concepts(onto):
        metric_keys = {
            normalize_metric_key(metric["concept"]),
            normalize_metric_key(metric["term"]),
            normalize_metric_key(metric["pbi_field"]),
        }
        if alias_keys & metric_keys:
            return metric
    return None


def find_metric_by_dax(
    onto: "Ontology",
    dax_expression: str,
) -> Optional[MetricConcept]:
    """Find a metric concept whose stored DAX expression matches exactly."""
    normalized_target = re.sub(r"\s+", "", dax_expression or "").lower()
    if not normalized_target:
        return None

    for metric in list_metric_concepts(onto):
        normalized_metric_dax = re.sub(
            r"\s+",
            "",
            metric.get("dax_expression", "") or "",
        ).lower()
        if normalized_metric_dax == normalized_target:
            return metric
    return None


def _find_metric_by_concept(
    onto: "Ontology",
    concept_name: str,
) -> Optional[MetricConcept]:
    return find_metric_by_alias(onto, [concept_name])


def _resolved_metric_details(
    onto: "Ontology",
    intent: Mapping[str, Any],
) -> List[MetricConcept]:
    details: List[MetricConcept] = []
    for metric in intent.get("metrics", []):
        concept_name = metric.get("concept", "")
        found = _find_metric_by_concept(onto, concept_name)
        if found is not None:
            details.append(found)
    return details


def _prefer_alias_group(
    onto: "Ontology",
    alias_groups: Sequence[Sequence[str]],
) -> Optional[MetricConcept]:
    for alias_group in alias_groups:
        found = find_metric_by_alias(onto, alias_group)
        if found is not None:
            return found
    return None


def _infer_comparison_kind(text: str, time_field: str) -> tuple[str, str, List[str]]:
    lower_time = time_field.lower()
    if "year over year" in text or "yoy" in text or "year" in lower_time:
        return (
            "Year Over Year Change",
            f"SAMEPERIODLASTYEAR({time_field})",
            ["year over year", "yoy", "versus last year", "annual change"],
        )
    if "quarter over quarter" in text or "qoq" in text or "quarter" in lower_time:
        return (
            "Quarter Over Quarter Change",
            f"PREVIOUSQUARTER({time_field})",
            ["quarter over quarter", "qoq", "versus last quarter", "quarterly change"],
        )
    if "month over month" in text or "mom" in text or "month" in lower_time:
        return (
            "Month Over Month Change",
            f"PREVIOUSMONTH({time_field})",
            ["month over month", "mom", "versus last month", "monthly change"],
        )
    return (
        "Period Variance",
        f"PREVIOUSPERIOD({time_field})",
        ["compare", "change", "difference", "previous period"],
    )


def _existing_or_new_expected_concept(
    existing_metric: Optional[MetricConcept],
    concept_id: str,
) -> List[str]:
    if existing_metric is not None:
        return [existing_metric["concept"]]
    return [concept_id]


def plan_actual_vs_target_metric_fix(
    onto: "Ontology",
    intent: Mapping[str, Any],
    tokens: Sequence[Mapping[str, str]],
) -> Optional[RepairProposal]:
    """Plan a deterministic repair for actual-vs-target questions."""
    token_text = _joined_token_text(tokens)
    if not _contains_any_phrase(token_text, list(EXPECTATION_TERMS)):
        return None

    actual_metric = _prefer_alias_group(onto, ACTUAL_METRIC_ALIASES)
    target_metric = _prefer_alias_group(onto, TARGET_METRIC_ALIASES)
    if actual_metric is None or target_metric is None:
        return None

    target_term = clean_display_name(target_metric["term"])
    if normalize_metric_key(target_term) in {"budget", "target", "plan", "forecast"}:
        display_name = f"{target_term} Variance"
    else:
        actual_term = clean_display_name(actual_metric["term"])
        display_name = f"{actual_term} vs {target_term} Variance"

    concept_id = sanitize_concept_id(display_name)
    dax_expression = f"{actual_metric['pbi_field']} - {target_metric['pbi_field']}"
    existing_metric = find_metric_by_dax(onto, dax_expression)
    if existing_metric is None:
        existing_metric = find_metric_by_alias(onto, [display_name, concept_id])

    resolved_metric_concepts = {
        metric.get("concept", "")
        for metric in intent.get("metrics", [])
    }
    expected_concepts = _existing_or_new_expected_concept(existing_metric, concept_id)
    if resolved_metric_concepts & set(expected_concepts):
        return None

    expressions = [
        "below expectations",
        "below target",
        "under target",
        "under budget",
        "not meeting plan",
    ]
    reasoning = (
        f"In this model, expectations are represented by {target_metric['concept']} and "
        f"actual performance by {actual_metric['concept']}, so the query needs a "
        "variance metric rather than another synonym for a base KPI."
    )

    if existing_metric is not None:
        return RepairProposal(
            action="map_existing",
            concept=existing_metric["concept"],
            pbi_field=existing_metric.get("pbi_field", ""),
            dax_expression="",
            expressions=expressions,
            reasoning=reasoning,
            diagnosed_by="deterministic_rule",
            planner_kind="actual_vs_target_variance",
            expected_metric_concepts=expected_concepts,
            source_metrics=[actual_metric["concept"], target_metric["concept"]],
        )

    return RepairProposal(
        action="create_new",
        concept=concept_id,
        pbi_field=f"[{display_name}]",
        dax_expression=dax_expression,
        expressions=expressions,
        reasoning=reasoning,
        diagnosed_by="deterministic_rule",
        planner_kind="actual_vs_target_variance",
        expected_metric_concepts=expected_concepts,
        source_metrics=[actual_metric["concept"], target_metric["concept"]],
    )


def plan_share_of_total_metric_fix(
    onto: "Ontology",
    intent: Mapping[str, Any],
    tokens: Sequence[Mapping[str, str]],
) -> Optional[RepairProposal]:
    """Plan a deterministic repair for contribution/share queries."""
    token_text = _joined_token_text(tokens)
    patterns = {pattern.get("pattern", "") for pattern in intent.get("query_patterns", [])}
    if "Contribution" not in patterns and not _contains_any_phrase(token_text, list(SHARE_TERMS)):
        return None

    resolved_metrics = _resolved_metric_details(onto, intent)
    actual_metric = resolved_metrics[0] if resolved_metrics else _prefer_alias_group(
        onto,
        ACTUAL_METRIC_ALIASES,
    )
    if actual_metric is None:
        return None

    dimension = next(
        (
            item
            for item in intent.get("group_by", [])
            if item.get("field")
        ),
        None,
    )
    if dimension is None:
        return None

    display_name = f"{clean_display_name(actual_metric['term'])} Share Of Total"
    concept_id = sanitize_concept_id(display_name)
    dimension_field = dimension["field"]
    dax_expression = (
        f"DIVIDE({actual_metric['pbi_field']}, "
        f"CALCULATE({actual_metric['pbi_field']}, ALL({dimension_field})))"
    )
    existing_metric = find_metric_by_dax(onto, dax_expression)
    if existing_metric is None:
        existing_metric = find_metric_by_alias(onto, [display_name, concept_id])

    resolved_metric_concepts = {
        metric.get("concept", "")
        for metric in intent.get("metrics", [])
    }
    expected_concepts = _existing_or_new_expected_concept(existing_metric, concept_id)
    if resolved_metric_concepts & set(expected_concepts):
        return None

    reasoning = (
        f"A contribution-style query needs a share-of-total calculation for "
        f"{actual_metric['concept']} across {dimension.get('concept', dimension_field)}."
    )
    expressions = [
        "share of total",
        "percentage of total",
        "contribution",
        "mix",
        f"{clean_display_name(actual_metric['term']).lower()} share",
    ]

    if existing_metric is not None:
        return RepairProposal(
            action="map_existing",
            concept=existing_metric["concept"],
            pbi_field=existing_metric.get("pbi_field", ""),
            dax_expression="",
            expressions=expressions,
            reasoning=reasoning,
            diagnosed_by="deterministic_rule",
            planner_kind="share_of_total",
            expected_metric_concepts=expected_concepts,
            source_metrics=[actual_metric["concept"]],
        )

    return RepairProposal(
        action="create_new",
        concept=concept_id,
        pbi_field=f"[{display_name}]",
        dax_expression=dax_expression,
        expressions=expressions,
        reasoning=reasoning,
        diagnosed_by="deterministic_rule",
        planner_kind="share_of_total",
        expected_metric_concepts=expected_concepts,
        source_metrics=[actual_metric["concept"]],
    )


def plan_period_over_period_metric_fix(
    onto: "Ontology",
    intent: Mapping[str, Any],
    tokens: Sequence[Mapping[str, str]],
) -> Optional[RepairProposal]:
    """Plan a deterministic repair for period-over-period comparisons."""
    token_text = _joined_token_text(tokens)
    patterns = {pattern.get("pattern", "") for pattern in intent.get("query_patterns", [])}
    if "Comparison" not in patterns and not _contains_any_phrase(token_text, list(COMPARISON_TERMS)):
        return None

    resolved_metrics = _resolved_metric_details(onto, intent)
    actual_metric = resolved_metrics[0] if resolved_metrics else _prefer_alias_group(
        onto,
        ACTUAL_METRIC_ALIASES,
    )
    if actual_metric is None:
        return None

    time_reference = next(
        (
            item
            for item in intent.get("time_filters", [])
            if item.get("field")
        ),
        None,
    )
    if time_reference is None:
        time_concepts = _list_concepts_for_semantic_type(onto, "TimeDimension")
        if not time_concepts:
            return None
        time_reference = {
            "field": time_concepts[0].get("pbi_field", ""),
            "concept": time_concepts[0].get("concept", ""),
        }

    time_field = time_reference.get("field", "")
    if not time_field:
        return None

    label_suffix, previous_period_expr, expressions = _infer_comparison_kind(
        token_text,
        time_field,
    )
    display_name = f"{clean_display_name(actual_metric['term'])} {label_suffix}"
    concept_id = sanitize_concept_id(display_name)
    dax_expression = (
        f"{actual_metric['pbi_field']} - "
        f"CALCULATE({actual_metric['pbi_field']}, {previous_period_expr})"
    )
    existing_metric = find_metric_by_dax(onto, dax_expression)
    if existing_metric is None:
        existing_metric = find_metric_by_alias(onto, [display_name, concept_id])

    resolved_metric_concepts = {
        metric.get("concept", "")
        for metric in intent.get("metrics", [])
    }
    expected_concepts = _existing_or_new_expected_concept(existing_metric, concept_id)
    if resolved_metric_concepts & set(expected_concepts):
        return None

    reasoning = (
        f"A comparison query needs an explicit prior-period delta for {actual_metric['concept']} "
        f"using {time_reference.get('concept', time_field)} as the time reference."
    )

    if existing_metric is not None:
        return RepairProposal(
            action="map_existing",
            concept=existing_metric["concept"],
            pbi_field=existing_metric.get("pbi_field", ""),
            dax_expression="",
            expressions=expressions,
            reasoning=reasoning,
            diagnosed_by="deterministic_rule",
            planner_kind="period_over_period_variance",
            expected_metric_concepts=expected_concepts,
            source_metrics=[actual_metric["concept"]],
        )

    return RepairProposal(
        action="create_new",
        concept=concept_id,
        pbi_field=f"[{display_name}]",
        dax_expression=dax_expression,
        expressions=expressions,
        reasoning=reasoning,
        diagnosed_by="deterministic_rule",
        planner_kind="period_over_period_variance",
        expected_metric_concepts=expected_concepts,
        source_metrics=[actual_metric["concept"]],
    )


def metric_resolution_specificity(
    token_text: str,
    role: str,
    concept_attrs: Mapping[str, Any],
) -> int:
    """Return a lower-is-better specificity score for metric matches.

    Derived metrics are preferred over base metrics when the token text clearly
    implies a calculated concept such as actual-vs-target variance,
    period-over-period delta, or share of total.
    """
    if role != "filter":
        return 1

    text = token_text.strip().lower()
    dax_expression = concept_attrs.get("dax_expression", "") or ""
    planner_kind = concept_attrs.get("planner_kind", "") or ""
    normalized_dax = dax_expression.lower()

    if _contains_any_phrase(text, list(EXPECTATION_TERMS)):
        if planner_kind == "actual_vs_target_variance":
            return 0
        if dax_expression:
            return 1
        return 2

    if _contains_any_phrase(text, list(SHARE_TERMS)):
        if planner_kind == "share_of_total":
            return 0
        if "divide(" in normalized_dax and "all(" in normalized_dax:
            return 1
        return 2

    if _contains_any_phrase(text, list(COMPARISON_TERMS)):
        if planner_kind == "period_over_period_variance":
            return 0
        if any(
            marker in normalized_dax
            for marker in ("previousperiod(", "sameperiodlastyear(", "previousquarter(", "previousmonth(")
        ):
            return 1
        return 2

    return 1


def plan_rule_based_metric_fix(
    onto: "Ontology",
    intent: Mapping[str, Any],
    tokens: Sequence[Mapping[str, str]],
) -> Optional[RepairProposal]:
    """Plan a deterministic metric repair for common Power BI query families."""
    token_text = _joined_token_text(tokens)
    patterns = {pattern.get("pattern", "") for pattern in intent.get("query_patterns", [])}

    planners = []
    if _contains_any_phrase(token_text, list(EXPECTATION_TERMS)):
        planners.append(plan_actual_vs_target_metric_fix)
    if "Contribution" in patterns or _contains_any_phrase(token_text, list(SHARE_TERMS)):
        planners.append(plan_share_of_total_metric_fix)
    if "Comparison" in patterns or _contains_any_phrase(token_text, list(COMPARISON_TERMS)):
        planners.append(plan_period_over_period_metric_fix)

    for planner in planners:
        proposal = planner(onto, intent, tokens)
        if proposal is not None:
            return proposal
    return None


def inject_pbi_semantic_relations(onto: "Ontology") -> int:
    """Inject semantic relation edges that make analytical neighborhoods searchable.

    These relations complement lexical alias lookup. They encode reusable business
    semantics such as actual-vs-target, comparable-over-time, derived-from, and
    query-pattern support without replacing the procedural planner layer.
    """
    added = 0
    pattern_nodes = {
        concept["concept"] for concept in _list_concepts_for_semantic_type(onto, "QueryPattern")
    }
    time_nodes = {
        concept["concept"] for concept in _list_concepts_for_semantic_type(onto, "TimeDimension")
    }

    actual_metric = _prefer_alias_group(onto, ACTUAL_METRIC_ALIASES)
    target_metric = _prefer_alias_group(onto, TARGET_METRIC_ALIASES)
    if actual_metric is not None and target_metric is not None:
        added += _attach_semantic_relation(
            onto,
            actual_metric["concept"],
            target_metric["concept"],
            ACTUAL_VS_TARGET_RELATION,
        )

    for metric in list_metric_concepts(onto):
        metric_node = metric["concept"]
        attrs = onto.ontology_graph.nodes[metric_node]
        source_metrics = _dedupe_preserve_order(attrs.get("source_metrics", []) or [])
        planner_kind = attrs.get("planner_kind", "") or ""

        # Any metric that can be filtered by time can participate in time-based comparisons.
        comparable_targets = [
            target
            for target in _neighbors_by_relation(onto, metric_node, "filterable_by")
            if target in time_nodes
        ]
        for time_node in comparable_targets:
            added += _attach_semantic_relation(
                onto,
                metric_node,
                time_node,
                COMPARABLE_OVER_TIME_RELATION,
            )
        if comparable_targets:
            for pattern_name in ("Comparison", "Trend", "Forecast"):
                if pattern_name in pattern_nodes:
                    added += _attach_semantic_relation(
                        onto,
                        metric_node,
                        pattern_name,
                        SUPPORTS_QUERY_PATTERN_RELATION,
                    )

        for pattern_name in ("Aggregation", "Ranking"):
            if pattern_name in pattern_nodes:
                added += _attach_semantic_relation(
                    onto,
                    metric_node,
                    pattern_name,
                    SUPPORTS_QUERY_PATTERN_RELATION,
                )

        if planner_kind == "actual_vs_target_variance":
            for source_metric in source_metrics:
                added += _attach_semantic_relation(
                    onto,
                    metric_node,
                    source_metric,
                    DERIVED_FROM_RELATION,
                )
            if len(source_metrics) >= 2:
                added += _attach_semantic_relation(
                    onto,
                    source_metrics[0],
                    source_metrics[1],
                    ACTUAL_VS_TARGET_RELATION,
                )
            for pattern_name in ("NegativeFilter", "Threshold", "Ranking"):
                if pattern_name in pattern_nodes:
                    added += _attach_semantic_relation(
                        onto,
                        metric_node,
                        pattern_name,
                        SUPPORTS_QUERY_PATTERN_RELATION,
                    )

        if planner_kind == "share_of_total":
            for source_metric in source_metrics:
                added += _attach_semantic_relation(
                    onto,
                    metric_node,
                    source_metric,
                    DERIVED_FROM_RELATION,
                )
                added += _attach_semantic_relation(
                    onto,
                    metric_node,
                    source_metric,
                    CONTRIBUTION_OF_RELATION,
                )
            for pattern_name in ("Contribution", "Ranking"):
                if pattern_name in pattern_nodes:
                    added += _attach_semantic_relation(
                        onto,
                        metric_node,
                        pattern_name,
                        SUPPORTS_QUERY_PATTERN_RELATION,
                    )

        if planner_kind == "period_over_period_variance":
            for source_metric in source_metrics:
                added += _attach_semantic_relation(
                    onto,
                    metric_node,
                    source_metric,
                    DERIVED_FROM_RELATION,
                )
            targets = comparable_targets or sorted(time_nodes)
            for time_node in targets:
                added += _attach_semantic_relation(
                    onto,
                    metric_node,
                    time_node,
                    COMPARABLE_OVER_TIME_RELATION,
                )
            for pattern_name in ("Comparison", "Trend", "Forecast"):
                if pattern_name in pattern_nodes:
                    added += _attach_semantic_relation(
                        onto,
                        metric_node,
                        pattern_name,
                        SUPPORTS_QUERY_PATTERN_RELATION,
                    )

    return added


def find_relation_metric_matches(
    onto: "Ontology",
    token_text: str,
    intent: Mapping[str, Any],
) -> RelationSearchResult:
    """Search metric candidates through semantic relation neighborhoods.

    This complements lexical alias lookup by traversing semantic relations such
    as actual-vs-target, derived-from, comparable-over-time, and
    supports-query-pattern.
    """
    text = token_text.strip().lower()
    result: RelationSearchResult = {"matches": [], "evidence": []}
    if not text:
        return result

    pattern_names = frozenset(
        pattern.get("pattern", "") for pattern in intent.get("query_patterns", []) if pattern.get("pattern", "")
    )
    seen: set[str] = set()

    def _supports_current_patterns(metric_node: str) -> bool:
        if not pattern_names:
            return True
        supported_patterns = set(_neighbors_by_relation(onto, metric_node, "supports_query_pattern"))
        return not supported_patterns or bool(supported_patterns & pattern_names)

    def _add_match(metric_node: str, reason: str, path: Sequence[str]) -> None:
        if metric_node in seen or metric_node not in onto.ontology_graph:
            return
        if "Metric" not in _concept_semantic_types(onto, metric_node):
            return
        if not _supports_current_patterns(metric_node):
            return
        seen.add(metric_node)
        result["matches"].append(
            RelationMetricMatch(
                expression_node=f"relation:{reason}",
                concept_node=metric_node,
                concept_attrs=dict(onto.ontology_graph.nodes[metric_node]),
                path=list(path),
                relation_reason=reason,
            )
        )

    for relation_spec in _triggered_relation_family_specs(text, pattern_names):
        if relation_spec.search_mode == "paired_derived_metric":
            _collect_paired_derived_metric_matches(
                onto,
                relation_spec,
                result,
                _add_match,
            )
            continue

        if relation_spec.search_mode == "direct_metric":
            _collect_direct_metric_relation_matches(
                onto,
                relation_spec,
                result,
                _add_match,
            )

    return result


def detect_pbi_resolution_gap(
    onto: "Ontology",
    intent: Mapping[str, Any],
    tokens: Sequence[Mapping[str, str]],
) -> ResolutionGap:
    """Analyze a Power BI intent for missing or semantically wrong metrics."""
    patterns = [pattern.get("pattern", "") for pattern in intent.get("query_patterns", [])]
    metrics = list(intent.get("metrics", []))
    unresolved = [
        line
        for line in intent.get("resolution_log", [])
        if "NO MATCH" in line
    ]

    gap: ResolutionGap = {
        "has_gap": False,
        "gap_type": None,
        "unresolved_tokens": [],
        "mismatched_filters": [],
        "context": {},
    }

    metric_needing_patterns = {
        "NegativeFilter",
        "Ranking",
        "Comparison",
        "Decomposition",
        "Aggregation",
        "Trend",
        "Forecast",
        "Threshold",
        "Contribution",
    }
    if set(patterns) & metric_needing_patterns and not metrics:
        gap["has_gap"] = True
        gap["gap_type"] = "no_metric"

    filter_tokens = [token for token in tokens if token.get("role") == "filter"]
    for filter_token in filter_tokens:
        matches = onto.resolve_expression(filter_token.get("text", ""))
        resolved_types = set()
        for match in matches:
            concept = match.get("concept_node", "")
            resolved_types.update(_concept_semantic_types(onto, concept))
        if "QueryPattern" in resolved_types and "Metric" not in resolved_types:
            gap["has_gap"] = True
            gap["gap_type"] = gap["gap_type"] or "filter_without_metric"
            gap["mismatched_filters"].append(
                {
                    "token": filter_token.get("text", ""),
                    "resolved_as": "QueryPattern",
                    "missing": "Metric",
                }
            )

    if unresolved:
        gap["has_gap"] = True
        gap["gap_type"] = gap["gap_type"] or "unresolved_tokens"
        gap["unresolved_tokens"] = unresolved

    deterministic_proposal = plan_rule_based_metric_fix(onto, intent, tokens)
    if deterministic_proposal is not None:
        gap["has_gap"] = True
        gap["gap_type"] = gap["gap_type"] or (
            "semantic_mismatch" if metrics else "derived_metric_missing"
        )
        filter_label = ", ".join(token.get("text", "") for token in filter_tokens) or "<none>"
        resolved_as = ", ".join(metric.get("concept", "") for metric in metrics) or "none"
        gap["mismatched_filters"].append(
            {
                "token": filter_label,
                "resolved_as": resolved_as,
                "missing": deterministic_proposal.get("concept", ""),
            }
        )

    gap["context"]["available_metrics"] = list_metric_concepts(onto)
    gap["context"]["patterns_matched"] = patterns
    gap["context"]["resolved_metrics"] = metrics
    gap["context"]["deterministic_proposal"] = deterministic_proposal
    return gap


def diagnose_and_propose_pbi(
    onto: "Ontology",
    gap: ResolutionGap,
    user_query: str,
) -> Dict[str, Any]:
    """Return a deterministic proposal first, then fall back to the LLM."""
    deterministic_proposal = gap.get("context", {}).get("deterministic_proposal")
    if deterministic_proposal:
        return dict(deterministic_proposal)

    available_metrics_str = "\n".join(
        (
            f"  - {metric['concept']} (field: {metric['pbi_field']}, "
            f"description: {metric['description']}, "
            f"dax: {metric['dax_expression'] or '<base measure>'})"
        )
        for metric in gap.get("context", {}).get("available_metrics", [])
    )
    patterns_str = ", ".join(gap.get("context", {}).get("patterns_matched", []))
    filters_str = ", ".join(
        f'"{item["token"]}" (resolved as {item["resolved_as"]}, missing {item["missing"]})'
        for item in gap.get("mismatched_filters", [])
    )

    prompt = f"""A user asked: \"{user_query}\"

The ontology resolved query patterns [{patterns_str}] but has a semantic gap:
  Filter tokens with no metric: {filters_str}

Available metrics in the data model:
{available_metrics_str}

Analyze this gap and respond with ONE of two actions:

Option A - \"map_existing\": An existing metric semantically covers the user's intent.
  Return which metric and what new expression synonyms should be added to it.

Option B - \"create_new\": No existing metric covers this intent. A new derived metric is needed.
  Return the new metric name, its DAX definition using existing measures, and expression synonyms.

Return ONLY valid JSON:
{{
  \"action\": \"map_existing\" or \"create_new\",
  \"concept\": \"<existing or new concept name>\",
  \"pbi_field\": \"<existing [Field] or new [Field Name]>\",
  \"dax_expression\": \"<DAX expression if create_new, empty string if map_existing>\",
  \"expressions\": [\"synonym1\", \"synonym2\", \"...\"],
  \"reasoning\": \"<one sentence explanation>\"
}}"""

    raw = onto.agent.chat(
        instructions=(
            "You are a Power BI + ontology expert. Diagnose semantic gaps in query resolution "
            "and propose minimal ontology updates. Create a new metric when the user intent "
            "clearly implies an actual-versus-target, period-comparison, or share-of-total "
            "calculation."
        ),
        input=prompt,
    )
    logger.info("PBI diagnosis raw response: %s", raw)

    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            proposal = json.loads(json_match.group(0))
        else:
            proposal = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Could not parse LLM response as JSON")
        proposal = {"action": "error", "reasoning": raw}
    return proposal


def _resolve_existing_concept_id(
    onto: "Ontology",
    concept_name: str,
    dax_expression: str = "",
) -> Optional[str]:
    direct_id = sanitize_concept_id(concept_name)
    if direct_id in onto.ontology_graph:
        return direct_id

    by_alias = find_metric_by_alias(onto, [concept_name, direct_id])
    if by_alias is not None:
        return by_alias["concept"]

    if dax_expression:
        by_dax = find_metric_by_dax(onto, dax_expression)
        if by_dax is not None:
            return by_dax["concept"]

    return None


def inject_pbi_proposal(
    onto: "Ontology",
    proposal: Mapping[str, Any],
    seed: Mapping[str, Any],
) -> bool:
    """Inject a Power BI ontology repair proposal into the graph."""
    action = proposal.get("action", "")
    concept_name = proposal.get("concept", "").strip()
    expressions = proposal.get("expressions", [])
    pbi_field = proposal.get("pbi_field", "")
    lexical_level_name = _primary_lexical_level_name(onto)

    if not concept_name or not expressions:
        return False

    if action == "map_existing":
        concept_id = _resolve_existing_concept_id(
            onto,
            concept_name,
            proposal.get("dax_expression", ""),
        )
        if concept_id is None:
            return False

        if proposal.get("planner_kind"):
            onto.ontology_graph.nodes[concept_id]["planner_kind"] = proposal.get(
                "planner_kind",
                "",
            )
        if proposal.get("source_metrics"):
            onto.ontology_graph.nodes[concept_id]["source_metrics"] = list(
                proposal.get("source_metrics", [])
            )

        added = 0
        for expr in expressions:
            expr_clean = expr.strip().lower()
            if expr_clean in onto.ontology_graph:
                if not onto.ontology_graph.has_edge(concept_id, expr_clean):
                    onto.ontology_graph.add_edge(
                        concept_id,
                        expr_clean,
                        relation="has_expression",
                    )
                    added += 1
                continue

            onto.ontology_graph.add_node(
                expr_clean,
                term=expr_clean,
                description=f"Auto-injected from query feedback: maps to {concept_id}",
                level=lexical_level_name,
                is_lexical=True,
            )
            onto.ontology_graph.add_edge(
                concept_id,
                expr_clean,
                relation="has_expression",
            )
            added += 1
        relation_added = inject_pbi_semantic_relations(onto)
        return added > 0 or relation_added > 0

    if action != "create_new":
        return False

    concept_id = _resolve_existing_concept_id(
        onto,
        concept_name,
        proposal.get("dax_expression", ""),
    )
    created_new_concept = False
    if concept_id is None:
        concept_id = sanitize_concept_id(concept_name)
        created_new_concept = True
        onto.ontology_graph.add_node(
            concept_id,
            term=concept_name,
            description=proposal.get("reasoning", ""),
            level="concept",
            pbi_field=pbi_field,
            dax_expression=proposal.get("dax_expression", ""),
            metric_origin="derived",
            planner_kind=proposal.get("planner_kind", ""),
            source_metrics=list(proposal.get("source_metrics", [])),
            n_visits=0,
            total_reward=0.0,
        )
        onto.ontology_graph.add_edge("Metric", concept_id, relation="has_concept")

        for branch in seed.get("taxonomy", []):
            if branch.get("type") == "Dimension":
                for concept in branch.get("concepts", []):
                    onto.ontology_graph.add_edge(
                        concept_id,
                        concept["concept"],
                        relation="sliceable_by",
                    )
            elif branch.get("type") == "TimeDimension":
                for concept in branch.get("concepts", []):
                    onto.ontology_graph.add_edge(
                        concept_id,
                        concept["concept"],
                        relation="filterable_by",
                    )
    else:
        onto.ontology_graph.nodes[concept_id]["term"] = concept_name
        onto.ontology_graph.nodes[concept_id]["description"] = proposal.get(
            "reasoning",
            onto.ontology_graph.nodes[concept_id].get("description", ""),
        )
        onto.ontology_graph.nodes[concept_id]["pbi_field"] = (
            pbi_field or onto.ontology_graph.nodes[concept_id].get("pbi_field", "")
        )
        if proposal.get("dax_expression"):
            onto.ontology_graph.nodes[concept_id]["dax_expression"] = proposal.get(
                "dax_expression",
                "",
            )
            onto.ontology_graph.nodes[concept_id]["metric_origin"] = "derived"
        if proposal.get("planner_kind"):
            onto.ontology_graph.nodes[concept_id]["planner_kind"] = proposal.get(
                "planner_kind",
                "",
            )
        if proposal.get("source_metrics"):
            onto.ontology_graph.nodes[concept_id]["source_metrics"] = list(
                proposal.get("source_metrics", [])
            )

    added = 0
    for expr in expressions:
        expr_clean = expr.strip().lower()
        if expr_clean in onto.ontology_graph:
            if not onto.ontology_graph.has_edge(concept_id, expr_clean):
                onto.ontology_graph.add_edge(
                    concept_id,
                    expr_clean,
                    relation="has_expression",
                )
                added += 1
            continue

        onto.ontology_graph.add_node(
            expr_clean,
            term=expr_clean,
            description=f"Auto-injected: maps to {concept_id}",
            level=lexical_level_name,
            is_lexical=True,
        )
        onto.ontology_graph.add_edge(
            concept_id,
            expr_clean,
            relation="has_expression",
        )
        added += 1

    relation_added = inject_pbi_semantic_relations(onto)
    return created_new_concept or added > 0 or relation_added > 0


def register_negative_feedback(
    query_text: str,
    extracted_tokens: Sequence[Mapping[str, str]],
    resolved_intent: Mapping[str, Any],
    corrected_concept: str,
    explanation: str,
    history: Optional[List[FeedbackEvent]] = None,
    overrides: Optional[Dict[str, FeedbackOverride]] = None,
) -> FeedbackEvent:
    """Capture a wrong-answer event as a structured, session-scoped correction.

    The helper records which concepts were rejected for the current query and
    creates phrase-level overrides that can bias subsequent resolution without
    mutating the persistent ontology after a single correction.
    """
    filter_terms = _filter_terms_from_tokens(extracted_tokens)
    rejected_concepts = _dedupe_preserve_order(
        [
            metric.get("concept", "").strip()
            for metric in resolved_intent.get("metrics", [])
            if metric.get("concept", "").strip()
        ]
    )

    event: FeedbackEvent = {
        "query": query_text,
        "feedback": "negative",
        "filter_terms": filter_terms,
        "rejected_concepts": rejected_concepts,
        "corrected_concept": corrected_concept,
        "explanation": explanation,
        "scope": "session",
    }

    if history is not None:
        history.append(event)

    if overrides is not None:
        for term in filter_terms:
            overrides[term] = {
                "prefer_concepts": [corrected_concept],
                "avoid_concepts": rejected_concepts,
                "reason": explanation,
            }

    return event


def count_feedback_matches(
    history: Sequence[FeedbackEvent],
    filter_term: str,
    corrected_concept: str,
) -> int:
    """Count how often the same correction pattern has been observed."""
    normalized_term = filter_term.strip().lower()
    return sum(
        1
        for event in history
        if normalized_term in event.get("filter_terms", [])
        and event.get("corrected_concept") == corrected_concept
    )


def feedback_promotion_recommendation(
    history: Sequence[FeedbackEvent],
    filter_term: str,
    corrected_concept: str,
    promotion_threshold: int = 3,
) -> str:
    """Recommend whether feedback should stay session-scoped or become persistent."""
    matches = count_feedback_matches(history, filter_term, corrected_concept)
    if matches >= promotion_threshold:
        return "Candidate for persistent ontology refinement"
    return "Keep as session override only"