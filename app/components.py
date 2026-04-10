"""Shared layout components for the Dash ontology application."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from dash import dcc, html


PAGE_LABELS = {
    "/": "Initialize",
    "/explorer": "Explorer",
    "/automation": "Automation",
    "/export": "Export",
}


def _nav_item(label: str, href: str, active: bool) -> html.Div:
    """Build a sidebar navigation row."""
    class_name = "nav-item"
    if active:
        class_name += " nav-item-active"
    return html.Div(dcc.Link(label.upper(), href=href, className="nav-link"), className=class_name)


def create_sidebar(pathname: str) -> html.Div:
    """Build the application sidebar."""
    return html.Div(
        className="sidebar",
        children=[
            html.Div(
                className="sidebar-brand",
                children=[
                    html.Div("ontogen", className="brand-title"),
                ],
            ),
            html.Div(
                className="sidebar-nav",
                children=[
                    _nav_item("Initialize", "/", pathname == "/"),
                    _nav_item("Explorer", "/explorer", pathname == "/explorer"),
                    _nav_item("Automation", "/automation", pathname == "/automation"),
                    _nav_item("Export", "/export", pathname == "/export"),
                ],
            ),
            html.Div(
                className="sidebar-footer",
                children=html.Button(
                    "NEW ONTOLOGY",
                    id="new-ontology-button",
                    className="primary-button sidebar-button",
                    type="button",
                ),
            ),
        ],
    )


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
                html.P("Select a node in the graph to inspect and expand it.", className="page-subtitle"),
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
