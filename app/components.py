"""Shared layout components for the Dash ontology application."""

from __future__ import annotations
from typing import Any, Dict, Iterable, List
from dash import dcc, html
import dash_bootstrap_components as dbc


PAGE_LABELS = {
    "/": "Initialize",
    "/explorer": "Explorer",
    "/automation": "Automation",
    "/export": "Export",
}

defaults = {
        "exploration_constant": 2.0,
        "max_iterations": 10,
        "similarity_threshold": 0.5,
        "confidence_threshold": 0.5,
        "candidates_per_iteration": 20,
        "cross_link_threshold": 70,
        "retirement_limit": 3,
        "initial_seed_terms": 5,
        "max_workers": 5,
    }

def _generation_parameters_form() -> html.Div:
    return html.Div(
        className="generation-parameters-form",
        children=[
            html.Label("Exploration Constant", className="form-label"),
            dbc.Input(
                id="init-exploration-constant",
                type="number",
                value=defaults["exploration_constant"],
                step=0.1,
                className="text-input",
            ),
            html.Label("Max Iterations", className="form-label"),
            dbc.Input(
                id="init-max-iterations",
                type="number",
                value=defaults["max_iterations"],
                min=1,
                className="text-input",
            ),
            html.Label("Similarity Threshold", className="form-label"),
            dbc.Input(
                id="init-similarity-threshold",
                type="number",
                value=defaults["similarity_threshold"],
                step=0.05,
                min=0,
                max=1,
                className="text-input",
            ),
            html.Label("Confidence Threshold", className="form-label"),
            dbc.Input(
                id="init-confidence-threshold",
                type="number",
                value=defaults["confidence_threshold"],
                step=0.05,
                min=0,
                max=1,
                className="text-input",
            ),
            html.Label("Candidates / Iteration", className="form-label"),
            dbc.Input(
                id="init-candidates-per-iteration",
                type="number",
                value=defaults["candidates_per_iteration"],
                min=1,
                className="text-input",
            ),
            html.Label("Cross-link Threshold", className="form-label"),
            dbc.Input(
                id="init-cross-link-threshold",
                type="number",
                value=defaults["cross_link_threshold"],
                min=0,
                max=100,
                className="text-input",
            ),
            html.Label("Retirement Limit", className="form-label"),
            dbc.Input(
                id="init-retirement-limit",
                type="number",
                value=defaults["retirement_limit"],
                min=1,
                className="text-input",
            ),
            html.Label("Initial Seed Terms", className="form-label"),
            dbc.Input(
                id="init-initial-seed-terms",
                type="number",
                value=defaults["initial_seed_terms"],
                min=1,
                className="text-input",
            ),
            html.Label("Max Workers", className="form-label"),
            dbc.Input(
                id="init-max-workers",
                type="number",
                value=defaults["max_workers"],
                min=1,
                className="text-input",
            ),
        ]
    )


def _provider_form() -> html.Div:
    """Build the provider configuration form shown in the sidebar."""
    return html.Div(
        className="provider-form",
        children=[
            html.Label("Provider", className="form-label"),
            dbc.Select(
                id="init-provider",
                options=[
                    {"label": "IAEDU", "value": "iaedu"},
                    {"label": "OpenAI", "value": "openai"},
                ],
                value="iaedu",
                className="text-input",
            ),
            html.Label("IAEDU API Key", className="form-label"),
            dbc.Input(
                id="init-iaedu-api-key",
                type="password",
                placeholder="Paste IAEDU key or rely on .env",
                className="text-input",
            ),
            html.Label("OpenAI API Key", className="form-label"),
            dbc.Input(
                id="init-openai-api-key",
                type="password",
                placeholder="sk-...",
                className="text-input",
            ),
            html.Label("OpenAI Model", className="form-label"),
            dbc.Input(
                id="init-openai-model",
                type="text",
                value="gpt-4.1-mini",
                className="text-input",
            ),
            html.Div(
                "IAEDU uses the notebook endpoint defaults unless workspace variables override them.",
                className="provider-note",
            ),
        ],
    )

def _create_sidebar_forms() -> html.Div:
    return html.Div(
        className="sidebar-forms",
        children=[
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            _generation_parameters_form()
                        ],
                        title="Generation Parameters",
                    ),
                    dbc.AccordionItem(
                        [_provider_form()],
                        title="Model Provider",
                    )
                ],
                start_collapsed=True  
            )
        ]
    )

def create_sidebar() -> html.Div:
    """Build the application sidebar."""
    return html.Div(
        className="sidebar",
        children=[
            html.H3("Ontogen", className="display-4"),
            html.Hr(),
            html.P("Ontology automatic generation", className="lead"),
            _create_sidebar_forms(),
            dbc.Button(
                "Generate",
                id="generate-button", 
                color="primary", 
                className="mb-3 mt-4 w-100", 
                n_clicks=0
            ),
            html.Div(
                className="theme-toggle",
                children=[
                    html.Span("Automatic"),
                    dbc.Switch(
                        id="mode-switch",
                        value=False,
                    ),
                    html.Span("Manual"),
                ],
            ),

        ]
    )

def create_modal_form() -> html.Div:

    domain_input = html.Div(
        [
            dbc.Label("Domain", html_for="domain-input"),
            dbc.Input(type="text", id="domain-input", placeholder="Enter domain"),
            dbc.FormText(
                "Enter the domain for the ontology generation",
                color="secondary",
            ),
        ],
        className="mb-3",
    )

    scope_input = html.Div(
        [
            dbc.Label("Scope", html_for="scope-input"),
            dbc.Textarea(
                id="scope-input",
                size="lg",
                placeholder="Enter scope",
            ),
        ],
        className="mb-3",
    )

    form = dbc.Form([domain_input, scope_input])

    return form


## OLD:


def create_topbar(pathname: str | None) -> html.Div:
    """Build the slim topbar for the current route."""
    section = PAGE_LABELS.get(pathname or "/", "Workspace")
    return html.Div(
        className="topbar",
        children=[
            html.Div("ontogen", className="topbar-title"),
            html.Div(section.upper(), className="topbar-page"),
        ],
    )


def empty_state(title: str, message: str) -> html.Div:
    """Render a simple empty-state card."""
    return html.Div(
        className="content-card empty-state",
        children=[
            html.H2(title, className="page-title"),
            html.P(message, className="page-subtitle"),
        ],
    )


def metric_card(label: str, value: Any) -> html.Div:
    """Render a compact metric card."""
    return html.Div(
        className="metric-card",
        children=[
            html.Div(label.upper(), className="metric-label"),
            html.Div(str(value), className="metric-value"),
        ],
    )


def kv_rows(items: Dict[str, Any]) -> html.Div:
    """Render key-value rows for config or metadata summaries."""
    rows: List[html.Div] = []
    for key, value in items.items():
        rows.append(
            html.Div(
                className="kv-row",
                children=[
                    html.Span(str(key).replace("_", " ").upper(), className="kv-key"),
                    html.Span(str(value), className="kv-value"),
                ],
            )
        )
    return html.Div(className="kv-grid", children=rows)


def render_relations(relations: Iterable[Dict[str, Any]]) -> html.Ul:
    """Render relation rows as a flat list."""
    items = [
        html.Li(
            f"{relation['relation']} {relation['direction']} {relation['node']}",
            className="relation-item",
        )
        for relation in relations
    ]
    if not items:
        items = [html.Li("No active relations", className="relation-item")]
    return html.Ul(items, className="relation-list")


def render_text_list(items: Iterable[str], empty_label: str) -> html.Ul:
    """Render a text-only list."""
    values = [html.Li(item, className="relation-item") for item in items]
    if not values:
        values = [html.Li(empty_label, className="relation-item")]
    return html.Ul(values, className="relation-list")


def node_detail_panel(details: Dict[str, Any] | None) -> html.Div:
    """Render the node details side panel."""
    if not details:
        return html.Div(
            className="node-panel-body",
            children=[
                html.Div("NODE_DETAILS", className="panel-eyebrow"),
                html.P(
                    "Select a node in the graph to inspect it, or right-click an expandable node to expand it directly.",
                    className="page-subtitle",
                ),
            ],
        )

    level_class = f"level-badge level-badge-{details['level']}"
    children: List[Any] = [
        html.Div("NODE_DETAILS", className="panel-eyebrow"),
        html.H3(details["term"], className="panel-title"),
        html.Div(details["level"].upper(), className=level_class),
        html.P(details.get("description") or "No description available.", className="panel-description"),
        html.Div(
            className="panel-section",
            children=[
                html.Div("CORE_PROPERTIES", className="panel-section-title"),
                kv_rows(
                    {
                        "node_id": details["id"],
                        "visits": details["visits"],
                        "total_reward": f"{details['total_reward']:.3f}",
                        "mean_reward": f"{details['mean_reward']:.3f}",
                    }
                ),
            ],
        ),
        html.Div(
            className="panel-section",
            children=[
                html.Div("PARENTS", className="panel-section-title"),
                render_text_list(details["parents"], "No parents"),
            ],
        ),
        html.Div(
            className="panel-section",
            children=[
                html.Div("CHILDREN", className="panel-section-title"),
                render_text_list(details["children"], "No children"),
            ],
        ),
        html.Div(
            className="panel-section",
            children=[
                html.Div("ACTIVE_RELATIONS", className="panel-section-title"),
                render_relations(details["incoming_relations"] + details["outgoing_relations"]),
            ],
        ),
    ]

    if details.get("retired"):
        children.append(html.Div("RETIRED", className="status-chip status-chip-danger"))

    if details.get("expandable") and not details.get("retired"):
        children.append(
            html.Button(
                "EXPAND_NODE",
                id="expand-node-button",
                className="primary-button",
                type="button",
            )
        )
    elif details.get("expandable"):
        children.append(
            html.Div(
                "This node is retired and cannot be expanded automatically.",
                className="panel-note",
            )
        )

    return html.Div(className="node-panel-body", children=children)


def manual_expansion_progress_panel(snapshot: Dict[str, Any]) -> html.Div:
    """Render live progress and logs for a background manual expansion."""
    logs = snapshot.get("recent_logs", [])
    log_output = "\n".join(logs[-80:]) if logs else "Waiting for manual expansion logs..."
    active = bool(snapshot.get("manual_expansion_active"))
    status = str(snapshot.get("generation_status", "idle"))
    node = snapshot.get("manual_expansion_node") or "-"
    last_error = snapshot.get("last_error")
    message = last_error or snapshot.get("last_message") or "Manual expansion queued."
    progress_value = 55 if active else 100
    progress_label = "Expanding..." if active else status.replace("_", " ").upper()
    status_class = "form-status form-status-warning" if active else "form-status form-status-success"
    if last_error:
        status_class = "form-status form-status-error"

    return html.Div(
        className="manual-expansion-progress",
        children=[
            html.Div("MANUAL_EXPANSION", className="panel-eyebrow"),
            html.H3(f"Expanding from: {node}", className="panel-title"),
            html.Div(message, className=status_class),
            dbc.Progress(
                value=progress_value,
                label=progress_label,
                striped=active,
                animated=active,
                className="mb-3",
            ),
            html.Div(
                className="metric-row",
                children=[
                    metric_card("Status", status.upper()),
                    metric_card("Iterations", len(snapshot.get("iteration_log", []))),
                    metric_card("Nodes", snapshot.get("node_count", 0)),
                    metric_card("Elapsed", f"{float(snapshot.get('elapsed_seconds', 0.0)):.1f}s"),
                ],
            ),
            html.Div("RUN_LOG", className="panel-eyebrow"),
            html.Pre(log_output, className="log-console manual-expansion-log"),
        ],
    )
