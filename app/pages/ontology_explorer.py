"""Main ontology workspace showing live progress, logs, and the final graph."""

from __future__ import annotations

from typing import Any, Dict

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from app.components import empty_state, manual_expansion_progress_panel, metric_card
from app.cytoscape_utils import (
    CYTOSCAPE_STYLESHEET,
    EXPAND_CONTEXT_MENU,
    EXPAND_CONTEXT_MENU_ID,
    graph_to_cytoscape,
)
from app.state import get_app_state


ACTIVE_STATUSES = {"initializing", "manual_expanding", "running", "paused"}
GRAPH_STATUSES = {"completed", "manual", "manual_expanding", "ready", "stopped"}


def _render_onboarding_view() -> html.Div:
    """Render the idle ontology explorer guidance canvas."""
    steps = [
        {
            "index": "01",
            "label": "Tune parameters",
            "copy": (
                "Open the first accordion and set the exploration constant, iteration "
                "budget, thresholds, seed size, and worker count before each run."
            ),
        },
        {
            "index": "02",
            "label": "Choose provider",
            "copy": (
                "Pick IAEDU or OpenAI, then add the matching credentials. If you use "
                "OpenAI, confirm the model field before starting ontogen."
            ),
        },
        {
            "index": "03",
            "label": "Pick mode",
            "copy": (
                "Automatic continues through the background expansion loop. Manual stops "
                "after the seed graph is ready so you can inspect the ontology first."
            ),
        },
        {
            "index": "04",
            "label": "Launch from modal",
            "copy": (
                "Click Generate, fill in the ontology domain and scope in the modal, then "
                "press Run Ontogen to launch the pipeline with the sidebar settings."
            ),
        },
        {
            "index": "05",
            "label": "Interact and explore",
            "copy": (
                "This workspace first becomes the live progress and run-log view. When the "
                "job finishes, it switches to the ontology graph so you can inspect nodes, "
                "relationships, and right-click expandable nodes for manual iterations."
            ),
        },
    ]

    return html.Div(
        className="explorer-onboarding",
        children=[
            html.Div(
                className="onboarding-stage",
                children=[
                    html.Div(
                        className="onboarding-sequence",
                        children=[
                            html.Div(
                                className="onboarding-step-card",
                                children=[
                                    html.Div(
                                        className="onboarding-step-marker",
                                        children=[
                                            html.Span(step["index"], className="onboarding-step-index"),
                                        ],
                                    ),
                                    html.Div(
                                        className="onboarding-step-content",
                                        children=[
                                            html.Div(step["label"], className="onboarding-step-title"),
                                            html.P(step["copy"], className="onboarding-step-copy"),
                                        ],
                                    ),
                                ],
                            )
                            for step in steps
                        ],
                    ),
                ],
            ),
        ],
    )


def _default_graph_layout() -> Dict[str, Any]:
    """Return the default Cytoscape layout configuration."""
    return {
        "name": "cose",
        "fit": True,
        "padding": 36,
        "animate": False,
    }


def _resolve_banner(snapshot: Dict[str, Any]) -> tuple[str, str]:
    """Map the current app snapshot to a banner message and style."""
    if snapshot["last_error"]:
        return snapshot["last_error"], "system-banner system-banner-error"
    if snapshot["generation_status"] in {"paused", "stopped"}:
        return snapshot["last_message"], "system-banner system-banner-warning"
    if snapshot["generation_status"] in ACTIVE_STATUSES:
        return snapshot["last_message"], "system-banner system-banner-warning"
    return snapshot["last_message"], "system-banner system-banner-ok"


def _calculate_progress(snapshot: Dict[str, Any]) -> int:
    """Estimate run progress from the initialization and iteration state."""
    status = snapshot["generation_status"]
    if status == "initializing":
        return 12

    max_iterations = int(snapshot["config"].get("max_iterations", 0) or 0)
    completed_iterations = len(snapshot["iteration_log"])
    if max_iterations > 0 and completed_iterations > 0:
        return max(15, min(100, int((completed_iterations / max_iterations) * 100)))

    if snapshot["has_ontology"] and not snapshot["auto_start_generation"]:
        return 100
    if snapshot["has_ontology"]:
        return 18
    return 0


def _info_stat(label: str, value: Any) -> html.Div:
    """Render one compact stat for the ontology info offcanvas."""
    return html.Div(
        className="ontology-info-stat",
        children=[
            html.Div(label, className="metric-label"),
            html.Div(str(value), className="ontology-info-value"),
        ],
    )


def _render_info_content(snapshot: Dict[str, Any]) -> html.Div:
    """Render offcanvas details for the generated ontology."""
    config = snapshot["config"]
    seed_summary = snapshot["seed_summary"]
    validation = seed_summary.get("validation", {}) if seed_summary else {}
    status = snapshot["generation_status"]
    status_message = snapshot["last_error"] or snapshot["last_message"] or "-"
    status_class = "ontology-info-status"
    if snapshot["last_error"]:
        status_class = f"{status_class} ontology-info-status-error"
    elif status in ACTIVE_STATUSES or status in {"paused", "stopped"}:
        status_class = f"{status_class} ontology-info-status-warning"
    else:
        status_class = f"{status_class} ontology-info-status-ok"
    settings = {
        "provider": config.get("provider", "-"),
        "model": config.get("model", "-"),
        "max_iterations": config.get("max_iterations", "-"),
        "similarity_threshold": config.get("similarity_threshold", "-"),
        "confidence_threshold": config.get("confidence_threshold", "-"),
        "candidates_per_iteration": config.get("candidates_per_iteration", "-"),
        "cross_link_threshold": config.get("cross_link_threshold", "-"),
        "retirement_limit": config.get("retirement_limit", "-"),
    }

    return html.Div(
        className="ontology-info-content",
        children=[
            html.Div(
                className=status_class,
                children=[
                    html.Div("Run Status", className="panel-eyebrow"),
                    html.Div(status.replace("_", " ").upper(), className="ontology-info-status-label"),
                    html.P(status_message, className="ontology-info-status-message"),
                ],
            ),
            html.Div(
                className="ontology-info-stat-grid",
                children=[
                    _info_stat("Domain", snapshot["domain"] or "-"),
                    _info_stat("Nodes", snapshot["node_count"]),
                    _info_stat("Edges", snapshot["edge_count"]),
                    _info_stat("Elapsed", f"{snapshot['elapsed_seconds']:.1f}s"),
                ],
            ),
            html.Div(
                className="ontology-info-section",
                children=[
                    html.Div("Run Settings", className="panel-eyebrow"),
                    html.Div(
                        className="ontology-info-list",
                        children=[
                            html.Div(
                                className="ontology-info-row",
                                children=[
                                    html.Span(str(key).replace("_", " "), className="kv-key"),
                                    html.Span(str(value), className="kv-value"),
                                ],
                            )
                            for key, value in settings.items()
                        ],
                    ),
                ],
            ),
            html.Div(
                className="ontology-info-section",
                children=[
                    html.Div("Seed Validation", className="panel-eyebrow"),
                    html.Div(
                        className="ontology-info-list",
                        children=[
                            html.Div(
                                className="ontology-info-row",
                                children=[
                                    html.Span(str(key).replace("_", " "), className="kv-key"),
                                    html.Span(str(value), className="kv-value"),
                                ],
                            )
                            for key, value in (validation or {"status": "-"}).items()
                        ],
                    ),
                ],
            ),
        ],
    )


def _render_progress_view(snapshot: Dict[str, Any]) -> html.Div:
    """Render the live progress and log console while ontogen is active."""
    progress_value = _calculate_progress(snapshot)
    banner_text, banner_class = _resolve_banner(snapshot)
    logs = "\n".join(snapshot["recent_logs"][-80:]) if snapshot["recent_logs"] else "Waiting for ontogen logs..."
    status = snapshot["generation_status"]
    progress_label = f"{progress_value}%"

    return html.Div(
        className="page page-ontology-explorer",
        children=[
            html.Div(
                className="page-header",
                children=[
                    html.H1("Ontology Explorer", className="page-title"),
                    html.P(
                        "Live initialization and automatic expansion progress for the active ontology run.",
                        className="page-subtitle",
                    ),
                ],
            ),
            html.Div(banner_text, className=banner_class),
            html.Div(
                className="content-card",
                children=[
                    dbc.Progress(
                        value=progress_value,
                        label=progress_label,
                        striped=status in ACTIVE_STATUSES,
                        animated=status in {"initializing", "running"},
                        className="mb-4",
                    ),
                    html.Div(
                        className="metric-row",
                        children=[
                            metric_card("Status", status.upper()),
                            metric_card("Iterations", len(snapshot["iteration_log"])),
                            metric_card("Nodes", snapshot["node_count"]),
                            metric_card("Elapsed", f"{snapshot['elapsed_seconds']:.1f}s"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="content-card",
                children=[
                    html.Div("RUN_LOG", className="panel-eyebrow"),
                    html.Pre(logs, className="log-console"),
                ],
            ),
        ],
    )


def _render_graph_view(snapshot: Dict[str, Any]) -> html.Div:
    """Render the final ontology graph once generation is finished."""
    app_state = get_app_state()
    ontology = app_state.ontology
    if ontology is None:
        return empty_state(
            "Ontology Explorer",
            "Configure the run in the sidebar and start ontogen from the modal.",
        )

    elements = graph_to_cytoscape(ontology.ontology_graph, app_state.get_expandable_nodes())
    banner_text, banner_class = _resolve_banner(snapshot)

    return html.Div(
        className="page page-ontology-explorer",
        children=[
            html.Div(
                className="page-header compact-header",
                children=[
                    html.H1("Ontology Explorer", className="page-title"),
                    html.P(
                        "The run is complete. Inspect the generated ontology graph below.",
                        className="page-subtitle",
                    ),
                ],
            ),
            html.Div(banner_text, className=banner_class),
            html.Div(
                className="metric-row",
                children=[
                    metric_card("Domain", snapshot["domain"] or "-"),
                    metric_card("Nodes", snapshot["node_count"]),
                    metric_card("Edges", snapshot["edge_count"]),
                    metric_card("Elapsed", f"{snapshot['elapsed_seconds']:.1f}s"),
                ],
            ),
            html.Div(
                className="content-card explorer-graph-card",
                children=[
                    cyto.Cytoscape(
                        id="ontology-explorer-graph",
                        elements=elements,
                        layout=_default_graph_layout(),
                        stylesheet=CYTOSCAPE_STYLESHEET,
                        style={"width": "100%", "height": "calc(100vh - 280px)"},
                        minZoom=0.2,
                        maxZoom=2.5,
                        zoom=1,
                        contextMenu=EXPAND_CONTEXT_MENU,
                    ),
                ],
            ),
        ],
    )


def layout() -> html.Div:
    """Return the ontology explorer shell for the root page."""
    return html.Div(
        className="page page-ontology-explorer",
        children=[
            dcc.Interval(
                id="ontology-explorer-interval",
                interval=1000,
                n_intervals=0,
                disabled=False,
            ),
            dcc.Store(id="ontology-explorer-refresh-signal", data=0),
            html.Div(id="ontology-explorer-banner", className="system-banner"),
            html.Div(id="ontology-explorer-empty-state"),
            html.Div(
                id="ontology-explorer-progress-section",
                className="run-progress-section",
                children=[
                    html.Div(
                        className="run-progress-panel",
                        children=[
                            dbc.Progress(
                                id="ontology-explorer-progress-bar",
                                value=0,
                                label="0%",
                                className="run-progress-bar mb-4",
                            ),
                            html.Div(
                                id="ontology-explorer-progress-metrics",
                                className="metric-row",
                            ),
                        ],
                    ),
                    html.Div(
                        className="run-log-panel",
                        children=[
                            html.Div("RUN_LOG", className="panel-eyebrow"),
                            html.Pre(
                                id="ontology-explorer-run-log",
                                className="log-console",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="ontology-explorer-graph-section",
                children=[
                    html.Div(
                        className="graph-action-bar",
                        children=[
                            dbc.Button(
                                [html.I(className="bi bi-info-circle me-2"), "Info"],
                                id="ontology-info-button",
                                className="graph-info-button",
                                color="primary",
                                n_clicks=0,
                            ),
                            dbc.Button(
                                html.I(className="bi bi-layout-sidebar-inset-reverse"),
                                id="sidebar-toggle",
                                className="sidebar-toggle-button sidebar-toggle-button-expanded",
                                color="light",
                                title="Hide sidebar",
                                n_clicks=0,
                                style={"display": "none"},
                            ),
                        ],
                    ),
                    html.Div(
                        id="ontology-explorer-action-message",
                        className="form-status",
                    ),
                    html.Div(
                        className="ontology-graph-panel explorer-graph-card",
                        children=[
                            cyto.Cytoscape(
                                id="ontology-explorer-graph",
                                elements=[],
                                layout=_default_graph_layout(),
                                stylesheet=CYTOSCAPE_STYLESHEET,
                                style={"width": "100%", "height": "calc(100vh - 164px)"},
                                minZoom=0.2,
                                maxZoom=2.5,
                                zoom=1,
                                contextMenu=EXPAND_CONTEXT_MENU,
                            ),
                        ],
                    ),
                    dbc.Offcanvas(
                        id="ontology-info-offcanvas",
                        title="Ontology Info",
                        placement="end",
                        is_open=False,
                        children=html.Div(id="ontology-info-content"),
                    ),
                    dbc.Offcanvas(
                        id="manual-expansion-offcanvas",
                        title="Manual Expansion Progress",
                        placement="bottom",
                        is_open=False,
                        scrollable=True,
                        children=html.Div(id="manual-expansion-content"),
                        style={"height": "70vh"},
                    ),
                ],
            ),
        ]
    )


def register_callbacks(app: Dash) -> None:
    """Register callbacks for the root ontology explorer page."""

    @app.callback(
        Output("ontology-explorer-banner", "children"),
        Output("ontology-explorer-banner", "className"),
        Output("ontology-explorer-banner", "style"),
        Output("ontology-explorer-empty-state", "children"),
        Output("ontology-explorer-progress-section", "style"),
        Output("ontology-explorer-progress-bar", "value"),
        Output("ontology-explorer-progress-bar", "label"),
        Output("ontology-explorer-progress-bar", "striped"),
        Output("ontology-explorer-progress-bar", "animated"),
        Output("ontology-explorer-progress-metrics", "children"),
        Output("ontology-explorer-run-log", "children"),
        Output("ontology-explorer-graph-section", "style"),
        Output("ontology-explorer-graph", "elements"),
        Output("ontology-info-content", "children"),
        Output("manual-expansion-content", "children"),
        Output("manual-expansion-offcanvas", "is_open"),
        Output("ontology-explorer-interval", "disabled"),
        Input("ontology-explorer-interval", "n_intervals"),
        Input("ontology-explorer-refresh-signal", "data"),
        Input("url", "pathname"),
    )
    def refresh_workspace(_interval: int, _refresh_signal: int, pathname: str | None):
        if pathname != "/":
            raise PreventUpdate

        snapshot = get_app_state().snapshot()
        status = snapshot["generation_status"]
        banner_text, banner_class = _resolve_banner(snapshot)
        progress_value = _calculate_progress(snapshot)
        progress_metrics = [
            metric_card("Status", status.upper()),
            metric_card("Iterations", len(snapshot["iteration_log"])),
            metric_card("Nodes", snapshot["node_count"]),
            metric_card("Elapsed", f"{snapshot['elapsed_seconds']:.1f}s"),
        ]
        logs = "\n".join(snapshot["recent_logs"][-80:]) if snapshot["recent_logs"] else "Waiting for ontogen logs..."
        info_content = _render_info_content(snapshot)
        manual_expansion_content = manual_expansion_progress_panel(snapshot)
        manual_expansion_active = bool(snapshot["manual_expansion_active"])

        empty_children = None
        banner_style = {"display": "block"}
        progress_style = {"display": "block"}
        graph_style = {"display": "none"}
        graph_elements: list[dict[str, Any]] = []

        if status in ACTIVE_STATUSES and not snapshot["last_error"]:
            banner_style = {"display": "none"}

        if not snapshot["has_ontology"] and status == "idle":
            banner_text = ""
            banner_class = "system-banner"
            banner_style = {"display": "none"}
            empty_children = _render_onboarding_view()
            progress_style = {"display": "none"}

        elif snapshot["has_ontology"] and status in GRAPH_STATUSES:
            banner_text = ""
            banner_class = "system-banner"
            banner_style = {"display": "none"}
            progress_style = {"display": "none"}
            graph_style = {"display": "block"}
            ontology = get_app_state().ontology
            if ontology is not None:
                graph_elements = graph_to_cytoscape(
                    ontology.ontology_graph,
                    get_app_state().get_expandable_nodes(),
                )

        return (
            banner_text,
            banner_class,
            banner_style,
            empty_children,
            progress_style,
            progress_value,
            f"{progress_value}%",
            status in ACTIVE_STATUSES,
            status in {"initializing", "running"},
            progress_metrics,
            logs,
            graph_style,
            graph_elements,
            info_content,
            manual_expansion_content,
            manual_expansion_active,
            False,
        )

    @app.callback(
        Output("ontology-explorer-action-message", "children"),
        Output("ontology-explorer-action-message", "className"),
        Output("ontology-explorer-refresh-signal", "data"),
        Input("ontology-explorer-graph", "contextMenuData"),
        State("ontology-explorer-refresh-signal", "data"),
        prevent_initial_call=True,
    )
    def expand_from_context_menu(
        context_menu_data: dict | None,
        refresh_signal: int,
    ):
        if not context_menu_data or context_menu_data.get("menuItemId") != EXPAND_CONTEXT_MENU_ID:
            raise PreventUpdate

        context_node_id = context_menu_data.get("elementId")
        if not context_node_id:
            raise PreventUpdate

        try:
            get_app_state().start_manual_expansion(context_node_id)
        except (RuntimeError, ValueError) as exc:
            return (
                str(exc),
                "form-status form-status-error",
                refresh_signal,
            )

        return (
            f"Expanding {context_node_id}. Progress and logs are shown below.",
            "form-status form-status-warning",
            refresh_signal + 1,
        )

    @app.callback(
        Output("ontology-info-offcanvas", "is_open"),
        Input("ontology-info-button", "n_clicks"),
        State("ontology-info-offcanvas", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_info_offcanvas(n_clicks: int | None, is_open: bool):
        if not n_clicks:
            raise PreventUpdate
        return not is_open
