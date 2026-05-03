"""Dash UI entry point for ontology generation."""

from __future__ import annotations

import os

from dash import Dash, Input, Output, dcc, html, no_update
from dash.exceptions import PreventUpdate

from app.components import create_sidebar, create_topbar, empty_state
from app.pages import automation, explorer, export, initialize
from app.state import get_app_state


def _resolve_layout(pathname: str | None) -> html.Div:
    """Route a pathname to the correct page layout."""
    app_state = get_app_state()
    path = pathname or "/"
    if path == "/":
        return initialize.layout()
    if not app_state.has_ontology():
        return empty_state(
            "Initialize Ontology Environment",
            "Create an ontology first to unlock the explorer, automation, and export views.",
        )
    if path == "/explorer":
        return explorer.layout()
    if path == "/automation":
        return automation.layout()
    if path == "/export":
        return export.layout()
    return empty_state("Route Not Found", "The requested view does not exist.")


def create_app() -> Dash:
    """Create and configure the Dash application."""
    app = Dash(__name__, suppress_callback_exceptions=True, update_title=None)
    app.title = "ontogen"
    app.layout = html.Div(
        className="app-root",
        children=[
            dcc.Location(id="url", refresh=False),
            dcc.ConfirmDialog(
                id="reset-confirm",
                message="A background ontology job is active. Reset the current session and discard the in-progress work?",
            ),
            html.Div(
                className="app-shell",
                children=[
                    html.Div(id="sidebar-container"),
                    html.Div(
                        className="main-column",
                        children=[
                            html.Div(id="topbar-container"),
                            html.Div(id="page-content", className="page-content"),
                        ],
                    ),
                ],
            ),
        ],
    )

    register_callbacks(app)
    initialize.register_callbacks(app)
    explorer.register_callbacks(app)
    automation.register_callbacks(app)
    export.register_callbacks(app)
    return app


def register_callbacks(app: Dash) -> None:
    """Register app-shell callbacks shared across pages."""

    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def render_page(pathname: str | None) -> html.Div:
        return _resolve_layout(pathname)

    @app.callback(
        Output("sidebar-container", "children"),
        Input("url", "pathname"),
    )
    def render_sidebar(pathname: str | None) -> html.Div:
        return create_sidebar(pathname or "/")


    @app.callback(
        Output("reset-confirm", "displayed"),
        Output("url", "pathname", allow_duplicate=True),
        Input("new-ontology-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def request_reset(n_clicks: int | None):
        if not n_clicks:
            raise PreventUpdate

        app_state = get_app_state()
        status = app_state.snapshot()["generation_status"]
        if status in {"initializing", "running", "paused"}:
            return True, no_update

        app_state.reset()
        return False, "/"

    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("reset-confirm", "displayed", allow_duplicate=True),
        Input("reset-confirm", "submit_n_clicks"),
        prevent_initial_call=True,
    )
    def confirm_reset(submit_n_clicks: int | None):
        if not submit_n_clicks:
            raise PreventUpdate
        get_app_state().reset()
        return "/", False


def main() -> None:
    """Launch the Dash application."""
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)


if __name__ == "__main__":
    main()
