"""Explorer page showing the interactive ontology graph."""

from __future__ import annotations

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import Dash, Input, Output, State, dcc, html, callback_context
from dash.exceptions import PreventUpdate

from app.components import empty_state, manual_expansion_progress_panel, node_detail_panel
from app.cytoscape_utils import (
    CYTOSCAPE_STYLESHEET,
    build_node_details,
    EXPAND_CONTEXT_MENU,
    EXPAND_CONTEXT_MENU_ID,
    graph_to_cytoscape,
)
from app.state import get_app_state


def _default_graph_layout(name: str) -> dict[str, object]:
    """Build a Cytoscape layout configuration."""
    return {
        "name": name,
        "fit": True,
        "padding": 36,
        "animate": False,
    }


def layout() -> html.Div:
    """Return the explorer page layout."""
    app_state = get_app_state()
    if not app_state.has_ontology():
        return empty_state(
            "Ontology Explorer",
            "Initialize an ontology environment before opening the explorer.",
        )

    ontology = app_state.ontology
    assert ontology is not None
    elements = graph_to_cytoscape(ontology.ontology_graph, app_state.get_expandable_nodes())

    return html.Div(
        className="page page-explorer",
        children=[
            dcc.Interval(id="explorer-interval", interval=1000, n_intervals=0, disabled=True),
            dcc.Store(id="explorer-selected-node"),
            dcc.Store(id="explorer-refresh-signal", data=0),
            html.Div(
                className="page-header compact-header",
                children=[
                    html.H1("Ontology Explorer", className="page-title"),
                    html.P(
                        "Inspect the live ontology graph, select nodes, and expand manually where needed.",
                        className="page-subtitle",
                    ),
                ],
            ),
            html.Div(
                id="explorer-system-message",
                className="system-banner",
            ),
            html.Div(
                className="explorer-layout",
                children=[
                    html.Div(
                        className="content-card explorer-graph-card",
                        children=[
                            cyto.Cytoscape(
                                id="ontology-graph",
                                elements=elements,
                                layout=_default_graph_layout("cose"),
                                stylesheet=CYTOSCAPE_STYLESHEET,
                                style={"width": "100%", "height": "calc(100vh - 260px)"},
                                minZoom=0.2,
                                maxZoom=2.5,
                                zoom=1,
                                contextMenu=EXPAND_CONTEXT_MENU,
                            ),
                            html.Div(
                                className="graph-toolbar",
                                children=[
                                    dcc.Dropdown(
                                        id="graph-layout-dropdown",
                                        options=[
                                            {"label": "COSE", "value": "cose"},
                                            {"label": "BREADTHFIRST", "value": "breadthfirst"},
                                            {"label": "CIRCLE", "value": "circle"},
                                        ],
                                        value="cose",
                                        clearable=False,
                                        className="graph-layout-dropdown",
                                    ),
                                    html.Div(
                                        className="toolbar-buttons",
                                        children=[
                                            html.Button("+", id="graph-zoom-in", className="icon-button", type="button"),
                                            html.Button("-", id="graph-zoom-out", className="icon-button", type="button"),
                                            html.Button("FIT", id="graph-fit", className="icon-button", type="button"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="content-card explorer-panel-card",
                        children=[
                            html.Div(id="node-detail-panel", children=node_detail_panel(None)),
                            html.Div(id="explorer-action-message", className="form-status explorer-action"),
                        ],
                    ),
                ],
            ),
            dbc.Offcanvas(
                id="explorer-manual-expansion-offcanvas",
                title="Manual Expansion Progress",
                placement="bottom",
                is_open=False,
                scrollable=True,
                children=html.Div(id="explorer-manual-expansion-content"),
                style={"height": "70vh"},
            ),
        ],
    )


def register_callbacks(app: Dash) -> None:
    """Register explorer-page callbacks."""

    @app.callback(
        Output("explorer-selected-node", "data"),
        Input("ontology-graph", "tapNodeData"),
        Input("ontology-graph", "contextMenuData"),
    )
    def remember_selected_node(node_data: dict | None, context_menu_data: dict | None):
        triggered_prop = callback_context.triggered[0]["prop_id"]
        if triggered_prop == "ontology-graph.contextMenuData" and context_menu_data:
            context_node_id = context_menu_data.get("elementId")
            if context_node_id:
                return context_node_id
        if not node_data:
            raise PreventUpdate
        return node_data.get("id")

    @app.callback(
        Output("ontology-graph", "elements"),
        Output("ontology-graph", "layout"),
        Output("ontology-graph", "zoom"),
        Output("explorer-system-message", "children"),
        Output("explorer-system-message", "className"),
        Output("explorer-interval", "disabled"),
        Output("explorer-manual-expansion-content", "children"),
        Output("explorer-manual-expansion-offcanvas", "is_open"),
        Input("explorer-interval", "n_intervals"),
        Input("explorer-refresh-signal", "data"),
        Input("graph-layout-dropdown", "value"),
        Input("graph-zoom-in", "n_clicks"),
        Input("graph-zoom-out", "n_clicks"),
        Input("graph-fit", "n_clicks"),
        State("ontology-graph", "zoom"),
        State("url", "pathname"),
    )
    def refresh_graph(
        _: int,
        refresh_signal: int,
        layout_name: str,
        zoom_in_clicks: int | None,
        zoom_out_clicks: int | None,
        fit_clicks: int | None,
        current_zoom: float | None,
        pathname: str | None,
    ):
        del refresh_signal
        del zoom_in_clicks, zoom_out_clicks, fit_clicks
        if pathname != "/explorer":
            raise PreventUpdate

        app_state = get_app_state()
        ontology = app_state.ontology
        if ontology is None:
            raise PreventUpdate

        elements = graph_to_cytoscape(ontology.ontology_graph, app_state.get_expandable_nodes())
        triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
        zoom_value = current_zoom if current_zoom is not None else 1
        if triggered_id == "graph-zoom-in":
            zoom_value = min((current_zoom or 1) + 0.15, 2.5)
        elif triggered_id == "graph-zoom-out":
            zoom_value = max((current_zoom or 1) - 0.15, 0.2)

        snapshot = app_state.snapshot()
        manual_expansion_active = bool(snapshot["manual_expansion_active"])
        interval_disabled = snapshot["generation_status"] not in {"manual_expanding", "running"}
        banner_class = "system-banner"
        banner_text = snapshot["last_message"]
        if snapshot["last_error"]:
            banner_text = snapshot["last_error"]
            banner_class += " system-banner-error"
        elif snapshot["mode"] == "manual":
            banner_class += " system-banner-warning"
        else:
            banner_class += " system-banner-ok"

        return (
            elements,
            _default_graph_layout(layout_name or "cose"),
            zoom_value,
            banner_text,
            banner_class,
            interval_disabled,
            manual_expansion_progress_panel(snapshot),
            manual_expansion_active,
        )

    @app.callback(
        Output("node-detail-panel", "children"),
        Input("explorer-selected-node", "data"),
        Input("explorer-interval", "n_intervals"),
        Input("explorer-refresh-signal", "data"),
        State("url", "pathname"),
    )
    def refresh_node_panel(
        selected_node: str | None,
        _interval: int,
        _refresh_signal: int,
        pathname: str | None,
    ):
        if pathname != "/explorer":
            raise PreventUpdate
        if not selected_node:
            return node_detail_panel(None)

        app_state = get_app_state()
        ontology = app_state.ontology
        if ontology is None or selected_node not in ontology.ontology_graph:
            return node_detail_panel(None)

        details = build_node_details(
            ontology.ontology_graph,
            selected_node,
            app_state.get_expandable_nodes(),
        )
        return node_detail_panel(details)

    @app.callback(
        Output("explorer-action-message", "children"),
        Output("explorer-action-message", "className"),
        Output("explorer-refresh-signal", "data"),
        Input("expand-node-button", "n_clicks"),
        Input("ontology-graph", "contextMenuData"),
        State("explorer-selected-node", "data"),
        State("explorer-refresh-signal", "data"),
        prevent_initial_call=True,
    )
    def manually_expand_node(
        n_clicks: int | None,
        context_menu_data: dict | None,
        selected_node: str | None,
        refresh_signal: int,
    ):
        triggered_prop = callback_context.triggered[0]["prop_id"]
        if triggered_prop == "ontology-graph.contextMenuData":
            if not context_menu_data or context_menu_data.get("menuItemId") != EXPAND_CONTEXT_MENU_ID:
                raise PreventUpdate
            node_to_expand = context_menu_data.get("elementId")
        else:
            if not n_clicks:
                raise PreventUpdate
            node_to_expand = selected_node

        if not node_to_expand:
            raise PreventUpdate

        try:
            get_app_state().start_manual_expansion(node_to_expand)
        except (RuntimeError, ValueError) as exc:
            return (
                str(exc),
                "form-status form-status-error explorer-action",
                refresh_signal,
            )

        return (
            f"Expanding {node_to_expand}. Progress and logs are shown below.",
            "form-status form-status-warning explorer-action",
            refresh_signal + 1,
        )
