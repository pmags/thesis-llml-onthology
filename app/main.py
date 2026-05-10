from __future__ import annotations

import os
from typing import cast

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html, no_update

from dash.exceptions import PreventUpdate

from app.components import create_modal_form, create_sidebar
from app.pages import ontology_explorer
from app.state import get_app_state

BRITE_THEME = "https://bootswatch.com/5/brite/bootstrap.min.css"
BOOTSTRAP_ICONS = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
GRAPH_READY_STATUSES = {"completed", "manual", "manual_expanding", "ready", "stopped"}


def _debug_enabled() -> bool:
    """Return whether Dash debug mode is explicitly enabled."""
    return os.getenv("ONTOGEN_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}


def _graph_is_ready() -> bool:
    """Return whether the current state can display the graph workspace."""
    snapshot = get_app_state().snapshot()
    return snapshot["has_ontology"] and snapshot["generation_status"] in GRAPH_READY_STATUSES


def _sidebar_render_state(visible: bool, graph_ready: bool):
    """Resolve sidebar and toggle presentation from visibility state."""
    if not graph_ready:
        return (
            "sidebar-container",
            "main-column",
            "sidebar-toggle-button sidebar-toggle-button-expanded",
            html.I(className="bi bi-layout-sidebar-inset-reverse"),
            "Hide sidebar",
            {"display": "none"},
        )

    if visible:
        return (
            "sidebar-container",
            "main-column",
            "sidebar-toggle-button sidebar-toggle-button-expanded",
            html.I(className="bi bi-layout-sidebar-inset-reverse"),
            "Hide sidebar",
            {"display": "inline-flex"},
        )

    return (
        "sidebar-container sidebar-container-hidden",
        "main-column main-column-full",
        "sidebar-toggle-button sidebar-toggle-button-collapsed",
        html.I(className="bi bi-layout-sidebar-inset"),
        "Show sidebar",
        {"display": "inline-flex"},
    )


def create_app() -> Dash:
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        update_title=cast(str, None),
        external_stylesheets=[BRITE_THEME, BOOTSTRAP_ICONS],
    )

    # Define layout and app conditions
    app.title = "ontogen"

    app.layout = html.Div(
        className="app-root",
        children=[
            dcc.Store(id="sidebar-visibility-store", data=True),
            dcc.Interval(
                id="sidebar-toggle-interval",
                interval=1000,
                n_intervals=0,
                disabled=False,
            ),
            html.Div(
                id="sidebar-container",
                className="sidebar-container",
                children=[
                    dcc.Location(id="url", refresh=False),
                    create_sidebar()
                ]
            ),
            html.Div(
                id="main-column",
                className="main-column",
                children=[
                    html.Div(
                        id="page-content", 
                        className="page-content"),
                    ],
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Please select domain and scope")),
                    dbc.ModalBody(
                        create_modal_form()
                    ),
                    dbc.ModalFooter(
                        dbc.Button("Run Ontogen", id="run-ontogen", className="ms-auto", n_clicks=0)
                    ),
                ],
                id="modal-xl",
                size="xl",
                centered=True,
                is_open=False,
            ),
            dcc.ConfirmDialog(
                id="stop-ontogen-confirm",
                message="Stop the current ontogen run? In-flight model requests may finish, but their results will be ignored.",
            ),
        ],

    )

    ontology_explorer.register_callbacks(app)

    @app.callback(
        Output("sidebar-visibility-store", "data"),
        Input("sidebar-toggle", "n_clicks"),
        State("sidebar-visibility-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_sidebar_visibility(n_clicks, stored_visibility):
        if not n_clicks:
            raise PreventUpdate
        if not _graph_is_ready():
            return True
        visible = True if stored_visibility is None else bool(stored_visibility)
        return not visible

    @app.callback(
        Output("sidebar-container", "className"),
        Output("main-column", "className"),
        Output("sidebar-toggle", "className"),
        Output("sidebar-toggle", "children"),
        Output("sidebar-toggle", "title"),
        Output("sidebar-toggle", "style"),
        Input("sidebar-visibility-store", "data"),
        Input("sidebar-toggle-interval", "n_intervals"),
    )
    def render_sidebar_visibility(stored_visibility, _n_intervals):
        visible = True if stored_visibility is None else bool(stored_visibility)
        return _sidebar_render_state(visible, _graph_is_ready())

    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname):
        if pathname == "/":
            return ontology_explorer.layout()
        # If the user tries to reach a different page, return a 404 message
        return html.Div(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ],
            className="p-3 bg-light rounded-3",
        )
    
    @app.callback(
        Output("modal-xl", "is_open"),
        Output("stop-ontogen-confirm", "displayed"),
        Input("generate-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_generate_modal(n_clicks):
        if not n_clicks:
            raise PreventUpdate

        status = get_app_state().snapshot()["generation_status"]
        if status in {"initializing", "manual_expanding", "running", "paused"}:
            return no_update, True
        return True, False

    @app.callback(
        Output("generate-button", "children"),
        Output("generate-button", "color"),
        Input("ontology-explorer-interval", "n_intervals"),
    )
    def refresh_generate_button(_n_intervals):
        status = get_app_state().snapshot()["generation_status"]
        if status in {"initializing", "manual_expanding", "running", "paused"}:
            return "Stop Ontogen", "danger"
        return "Generate", "primary"

    @app.callback(
        Output("stop-ontogen-confirm", "displayed", allow_duplicate=True),
        Input("stop-ontogen-confirm", "submit_n_clicks"),
        prevent_initial_call=True,
    )
    def stop_ontogen(submit_n_clicks):
        if not submit_n_clicks:
            raise PreventUpdate
        get_app_state().stop_run()
        return False

    @app.callback(
        Output("modal-xl", "is_open", allow_duplicate=True),
        Output("url", "pathname", allow_duplicate=True),
        Input("run-ontogen", "n_clicks"),
        State("domain-input", "value"),
        State("scope-input", "value"),
        State("mode-switch", "value"),
        State("init-provider", "value"),
        State("init-iaedu-api-key", "value"),
        State("init-openai-api-key", "value"),
        State("init-openai-model", "value"),
        State("init-exploration-constant", "value"),
        State("init-max-iterations", "value"),
        State("init-similarity-threshold", "value"),
        State("init-confidence-threshold", "value"),
        State("init-candidates-per-iteration", "value"),
        State("init-cross-link-threshold", "value"),
        State("init-retirement-limit", "value"),
        State("init-initial-seed-terms", "value"),
        State("init-max-workers", "value"),
        prevent_initial_call=True,
    )
    def run_ontogen(
        n_clicks,
        domain,
        scope,
        mode,
        provider,
        iaedu_api_key,
        openai_api_key,
        openai_model,
        exploration_constant,
        max_iterations,
        similarity_threshold,
        confidence_threshold,
        candidates_per_iteration,
        cross_link_threshold,
        retirement_limit,
        initial_seed_terms,
        max_workers,
        ):

        if not n_clicks:
            raise PreventUpdate

        app_state = get_app_state()
        provider_name = (provider or "iaedu").strip().lower()
        api_key = iaedu_api_key if provider_name == "iaedu" else openai_api_key
        automatic = not bool(mode)

        try:
            app_state.start_run(
                domain=domain or "",
                provider=provider_name,
                api_key=api_key,
                model=openai_model or "gpt-4.1-mini",
                scope_description=scope or "",
                sub_domains="",
                exploration_constant=exploration_constant if exploration_constant is not None else 2.0,
                max_iterations=max_iterations if max_iterations is not None else 10,
                similarity_threshold=similarity_threshold if similarity_threshold is not None else 0.5,
                confidence_threshold=confidence_threshold if confidence_threshold is not None else 0.5,
                candidates_per_iteration=candidates_per_iteration if candidates_per_iteration is not None else 20,
                cross_link_threshold=cross_link_threshold if cross_link_threshold is not None else 70,
                retirement_limit=retirement_limit if retirement_limit is not None else 3,
                initial_seed_terms=initial_seed_terms if initial_seed_terms is not None else 5,
                max_workers=max_workers if max_workers is not None else 5,
                automatic=automatic,
            )

        except (RuntimeError, ValueError) as e:
            print(f"Error initializing app state: {e}")
            return True, no_update

        return False, "/"

    return app



def main() -> None:
    """Launch the Dash application."""
    app = create_app()
    debug = _debug_enabled()
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8050")),
        debug=debug,
        use_reloader=debug,
        dev_tools_hot_reload=debug,
    )


if __name__ == "__main__":
    main()
