"""Automation page for live UCB1-driven expansion."""

from __future__ import annotations

from dash import Dash, Input, Output, State, dash_table, dcc, html, callback_context
from dash.exceptions import PreventUpdate

from app.components import empty_state, kv_rows, metric_card
from app.state import get_app_state


TABLE_COLUMNS = [
    {"name": "ITER", "id": "iteration"},
    {"name": "NODE", "id": "node"},
    {"name": "GEN", "id": "generated"},
    {"name": "ACC", "id": "accepted"},
    {"name": "RATE", "id": "acceptance_rate"},
    {"name": "REWARD", "id": "reward"},
    {"name": "NODES", "id": "nodes"},
    {"name": "EDGES", "id": "edges"},
    {"name": "STATUS", "id": "status"},
]


def _format_table_rows(rows: list[dict]) -> list[dict]:
    """Format numeric values for the automation table."""
    formatted = []
    for row in rows:
        formatted.append(
            {
                **row,
                "acceptance_rate": f"{row['acceptance_rate']:.1%}",
                "reward": f"{row['reward']:.3f}",
            }
        )
    return formatted


def layout() -> html.Div:
    """Return the automation page layout."""
    if not get_app_state().has_ontology():
        return empty_state(
            "Automation Console",
            "Initialize an ontology before starting automatic expansion.",
        )

    return html.Div(
        className="page page-automation",
        children=[
            dcc.Interval(id="automation-interval", interval=1000, n_intervals=0, disabled=True),
            html.Div(id="automation-command-signal", style={"display": "none"}),
            html.Div(
                className="page-header compact-header",
                children=[
                    html.H1("Automation Console", className="page-title"),
                    html.P(
                        "Run the UCB1 bandit loop in the background and inspect live expansion metrics.",
                        className="page-subtitle",
                    ),
                ],
            ),
            html.Div(
                className="page-grid automation-grid",
                children=[
                    html.Div(
                        className="content-card parameter-card",
                        children=[
                            html.Div("GLOBAL_PARAMETERS", className="panel-eyebrow"),
                            html.Div(id="automation-parameter-summary"),
                            html.Div(
                                className="automation-controls",
                                children=[
                                    html.Button("PLAY", id="automation-play", className="primary-button", type="button"),
                                    html.Button(
                                        "PAUSE",
                                        id="automation-pause-resume",
                                        className="secondary-button",
                                        type="button",
                                    ),
                                    html.Button("STOP", id="automation-stop", className="secondary-button", type="button"),
                                ],
                            ),
                            html.Div(id="automation-status-banner", className="form-status"),
                        ],
                    ),
                    html.Div(
                        className="content-card automation-table-card",
                        children=[
                            html.Div("ACTIVE_PROCESS_QUEUE", className="panel-eyebrow"),
                            dash_table.DataTable(
                                id="automation-table",
                                columns=TABLE_COLUMNS,
                                data=[],
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                style_header={
                                    "backgroundColor": "#111827",
                                    "color": "#f8fafc",
                                    "border": "none",
                                    "fontFamily": "IBM Plex Mono, monospace",
                                    "fontSize": "12px",
                                },
                                style_cell={
                                    "backgroundColor": "#f8fafc",
                                    "color": "#111827",
                                    "borderBottom": "1px solid #e5e7eb",
                                    "fontFamily": "IBM Plex Sans, Arial, sans-serif",
                                    "fontSize": "12px",
                                    "padding": "10px",
                                    "textAlign": "left",
                                },
                                page_size=10,
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(id="automation-metrics-row", className="metric-row"),
        ],
    )


def register_callbacks(app: Dash) -> None:
    """Register automation-page callbacks."""

    @app.callback(
        Output("automation-command-signal", "children"),
        Input("automation-play", "n_clicks"),
        Input("automation-pause-resume", "n_clicks"),
        Input("automation-stop", "n_clicks"),
        prevent_initial_call=True,
    )
    def control_automation(
        play_clicks: int | None,
        pause_clicks: int | None,
        stop_clicks: int | None,
    ) -> str:
        del play_clicks, pause_clicks, stop_clicks
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
        app_state = get_app_state()

        if triggered == "automation-play":
            try:
                app_state.start_generation()
            except RuntimeError as exc:
                app_state.last_error = str(exc)
                app_state.last_message = str(exc)
                return "error"
            return "play"
        if triggered == "automation-pause-resume":
            status = app_state.snapshot()["generation_status"]
            if status == "paused":
                app_state.resume_generation()
                return "resume"
            app_state.pause_generation()
            return "pause"
        if triggered == "automation-stop":
            app_state.stop_generation()
            return "stop"
        raise PreventUpdate

    @app.callback(
        Output("automation-parameter-summary", "children"),
        Output("automation-table", "data"),
        Output("automation-metrics-row", "children"),
        Output("automation-status-banner", "children"),
        Output("automation-status-banner", "className"),
        Output("automation-play", "disabled"),
        Output("automation-pause-resume", "disabled"),
        Output("automation-pause-resume", "children"),
        Output("automation-stop", "disabled"),
        Output("automation-interval", "disabled"),
        Input("automation-interval", "n_intervals"),
        Input("automation-command-signal", "children"),
        State("url", "pathname"),
    )
    def refresh_automation(_: int, _signal: str | None, pathname: str | None):
        if pathname != "/automation":
            raise PreventUpdate

        snapshot = get_app_state().snapshot()
        rows = _format_table_rows(snapshot["iteration_log"])
        metrics = [
            metric_card("Total Nodes", snapshot["node_count"]),
            metric_card("Total Edges", snapshot["edge_count"]),
            metric_card(
                "Overall Acceptance",
                (
                    f"{(sum(row['accepted'] for row in snapshot['iteration_log']) / sum(row['generated'] for row in snapshot['iteration_log'])):.1%}"
                    if snapshot["iteration_log"] and sum(row["generated"] for row in snapshot["iteration_log"]) > 0
                    else "0.0%"
                ),
            ),
            metric_card("Elapsed", f"{snapshot['elapsed_seconds']:.1f}s"),
        ]

        banner_class = "form-status"
        banner_text = snapshot["last_message"]
        if snapshot["last_error"]:
            banner_class += " form-status-error"
            banner_text = snapshot["last_error"]
        elif snapshot["mode"] == "manual":
            banner_class += " form-status-warning"
        else:
            banner_class += " form-status-success"

        status = snapshot["generation_status"]
        play_disabled = (not snapshot["has_ontology"]) or snapshot["mode"] == "manual" or status == "running"
        pause_disabled = status not in {"running", "paused"}
        pause_label = "RESUME" if status == "paused" else "PAUSE"
        stop_disabled = status not in {"running", "paused"}
        interval_disabled = status != "running"

        return (
            kv_rows(snapshot["config"]),
            rows,
            metrics,
            banner_text,
            banner_class,
            play_disabled,
            pause_disabled,
            pause_label,
            stop_disabled,
            interval_disabled,
        )
