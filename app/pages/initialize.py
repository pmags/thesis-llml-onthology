"""Initialize page for creating a new ontology session."""

from __future__ import annotations

from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

from app.components import kv_rows
from app.state import get_app_state, provider_is_ready, provider_status_message


def layout() -> html.Div:
    """Return the initialize page layout."""
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

    return html.Div(
        className="page page-initialize",
        children=[
            dcc.Interval(id="initialize-interval", interval=1000, n_intervals=0, disabled=True),
            html.Div(id="initialize-command-signal", style={"display": "none"}),
            html.Div(
                className="page-header",
                children=[
                    html.H1("Initialize Ontology", className="page-title"),
                    html.P(
                        "Set the domain, choose the provider, and prepare the ontology session "
                        "before running generation.",
                        className="page-subtitle",
                    ),
                ],
            ),
            dcc.Loading(
                type="circle",
                children=html.Div(
                    className="page-grid initialize-grid",
                    children=[
                        html.Div(
                            className="content-card form-card",
                            children=[
                                html.Label("Primary Domain", className="form-label"),
                                dcc.Input(
                                    id="init-domain",
                                    type="text",
                                    placeholder="E.g. Molecular Biology",
                                    className="text-input",
                                ),
                                html.Label("Scope Description", className="form-label"),
                                dcc.Textarea(
                                    id="init-scope",
                                    placeholder="Optional scope guidance for the ontology seed.",
                                    className="text-area",
                                ),
                                html.Label("Sub-domains (comma separated)", className="form-label"),
                                dcc.Input(
                                    id="init-subdomains",
                                    type="text",
                                    placeholder="Genomics, Proteomics, Cell Signaling",
                                    className="text-input",
                                ),
                                html.Label("Provider", className="form-label"),
                                dcc.RadioItems(
                                    id="init-provider",
                                    options=[
                                        {"label": "IAEDU", "value": "iaedu"},
                                        {"label": "OpenAI", "value": "openai"},
                                    ],
                                    value="iaedu",
                                    className="radio-group",
                                    inputClassName="radio-input",
                                    labelClassName="radio-label",
                                ),
                                html.Div(
                                    id="init-iaedu-fields",
                                    children=[
                                        html.Label("IAEDU API Key", className="form-label"),
                                        dcc.Input(
                                            id="init-iaedu-api-key",
                                            type="password",
                                            placeholder="Paste IAEDU key or rely on .env",
                                            className="text-input",
                                        ),
                                        html.Div(
                                            "Endpoint and channel use the notebook defaults unless workspace variables override them.",
                                            className="provider-note",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="init-openai-fields",
                                    style={"display": "none"},
                                    children=[
                                        html.Label("OpenAI API Key", className="form-label"),
                                        dcc.Input(
                                            id="init-openai-api-key",
                                            type="password",
                                            placeholder="sk-...",
                                            className="text-input",
                                        ),
                                        html.Label("OpenAI Model", className="form-label"),
                                        dcc.Input(
                                            id="init-openai-model",
                                            type="text",
                                            value="gpt-4.1-mini",
                                            className="text-input",
                                        ),
                                    ],
                                ),
                                html.Div(id="provider-status", className="form-status form-status-warning"),
                                html.Div(id="initialize-status", className="form-status"),
                                html.Button(
                                    "START SESSION",
                                    id="start-environment-button",
                                    type="button",
                                    className="primary-button large-button",
                                ),
                            ],
                        ),
                        html.Div(
                            className="right-column-stack",
                            children=[
                                html.Div(
                                    className="content-card parameter-card",
                                    children=[
                                        html.Div("SYSTEM_PARAMETERS", className="panel-eyebrow"),
                                        html.Div(
                                            className="form-grid",
                                            children=[
                                                html.Div(
                                                    className="form-field",
                                                    children=[
                                                        html.Label("Exploration Constant", className="form-label"),
                                                        dcc.Input(
                                                            id="init-exploration-constant",
                                                            type="number",
                                                            value=defaults["exploration_constant"],
                                                            step=0.1,
                                                            className="text-input",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="form-field",
                                                    children=[
                                                        html.Label("Max Iterations", className="form-label"),
                                                        dcc.Input(
                                                            id="init-max-iterations",
                                                            type="number",
                                                            value=defaults["max_iterations"],
                                                            min=1,
                                                            className="text-input",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="form-field",
                                                    children=[
                                                        html.Label("Similarity Threshold", className="form-label"),
                                                        dcc.Input(
                                                            id="init-similarity-threshold",
                                                            type="number",
                                                            value=defaults["similarity_threshold"],
                                                            step=0.05,
                                                            min=0,
                                                            max=1,
                                                            className="text-input",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="form-field",
                                                    children=[
                                                        html.Label("Confidence Threshold", className="form-label"),
                                                        dcc.Input(
                                                            id="init-confidence-threshold",
                                                            type="number",
                                                            value=defaults["confidence_threshold"],
                                                            step=0.05,
                                                            min=0,
                                                            max=1,
                                                            className="text-input",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="form-field",
                                                    children=[
                                                        html.Label("Candidates / Iteration", className="form-label"),
                                                        dcc.Input(
                                                            id="init-candidates-per-iteration",
                                                            type="number",
                                                            value=defaults["candidates_per_iteration"],
                                                            min=1,
                                                            className="text-input",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="form-field",
                                                    children=[
                                                        html.Label("Cross-link Threshold", className="form-label"),
                                                        dcc.Input(
                                                            id="init-cross-link-threshold",
                                                            type="number",
                                                            value=defaults["cross_link_threshold"],
                                                            min=0,
                                                            max=100,
                                                            className="text-input",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="form-field",
                                                    children=[
                                                        html.Label("Retirement Limit", className="form-label"),
                                                        dcc.Input(
                                                            id="init-retirement-limit",
                                                            type="number",
                                                            value=defaults["retirement_limit"],
                                                            min=1,
                                                            className="text-input",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="form-field",
                                                    children=[
                                                        html.Label("Initial Seed Terms", className="form-label"),
                                                        dcc.Input(
                                                            id="init-initial-seed-terms",
                                                            type="number",
                                                            value=defaults["initial_seed_terms"],
                                                            min=1,
                                                            className="text-input",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="form-field",
                                                    children=[
                                                        html.Label("Max Workers", className="form-label"),
                                                        dcc.Input(
                                                            id="init-max-workers",
                                                            type="number",
                                                            value=defaults["max_workers"],
                                                            min=1,
                                                            className="text-input",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="content-card parameter-card",
                                    children=[
                                        html.Div("SESSION_NOTES", className="panel-eyebrow"),
                                        kv_rows(
                                            {
                                                "default_provider": "IAEDU",
                                                "automatic_generation": "UCB1 bandit",
                                                "manual_expansion": "node panel action",
                                                "graph_renderer": "dash_cytoscape",
                                            }
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="content-card parameter-card",
                                    children=[
                                        html.Div("SESSION_LOG", className="panel-eyebrow"),
                                        html.P(
                                            "Live app and ontogen logs appear here while the ontology is being prepared.",
                                            className="panel-note",
                                        ),
                                        html.Pre(
                                            "No logs yet.",
                                            id="initialize-log-output",
                                            className="log-console",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ),
        ],
    )


def register_callbacks(app: Dash) -> None:
    """Register initialize-page callbacks."""

    @app.callback(
        Output("init-iaedu-fields", "style"),
        Output("init-openai-fields", "style"),
        Output("provider-status", "children"),
        Output("provider-status", "className"),
        Output("start-environment-button", "disabled"),
        Input("init-provider", "value"),
        Input("init-iaedu-api-key", "value"),
        Input("init-openai-api-key", "value"),
        Input("initialize-interval", "n_intervals"),
        Input("initialize-command-signal", "children"),
    )
    def toggle_provider_fields(
        provider: str,
        iaedu_api_key: str | None,
        openai_api_key: str | None,
        _interval: int,
        _command_signal: str | None,
    ):
        active_api_key = iaedu_api_key if provider == "iaedu" else openai_api_key
        snapshot = get_app_state().snapshot()
        if snapshot["generation_status"] == "initializing":
            return (
                {"display": "none"} if provider == "openai" else {"display": "block"},
                {"display": "block"} if provider == "openai" else {"display": "none"},
                "Initialization is in progress. Live logs are streaming below.",
                "form-status form-status-warning",
                True,
            )

        ready = provider_is_ready(provider, active_api_key)
        message = provider_status_message(provider, active_api_key)
        class_name = "form-status form-status-success" if ready else "form-status form-status-warning"
        if provider == "openai":
            return {"display": "none"}, {"display": "block"}, message, class_name, not ready
        return {"display": "block"}, {"display": "none"}, message, class_name, not ready

    @app.callback(
        Output("initialize-command-signal", "children"),
        Output("initialize-status", "children"),
        Output("initialize-status", "className"),
        Output("initialize-interval", "disabled"),
        Input("start-environment-button", "n_clicks"),
        State("init-domain", "value"),
        State("init-scope", "value"),
        State("init-subdomains", "value"),
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
    def start_environment(
        n_clicks: int | None,
        domain: str | None,
        scope_description: str | None,
        sub_domains: str | None,
        provider: str | None,
        iaedu_api_key: str | None,
        api_key: str | None,
        model: str | None,
        exploration_constant: float,
        max_iterations: int,
        similarity_threshold: float,
        confidence_threshold: float,
        candidates_per_iteration: int,
        cross_link_threshold: float,
        retirement_limit: int,
        initial_seed_terms: int,
        max_workers: int,
    ):
        if not n_clicks:
            raise PreventUpdate

        try:
            selected_api_key = iaedu_api_key if provider == "iaedu" else api_key
            get_app_state().start_initialization(
                domain=domain or "",
                provider=provider or "iaedu",
                api_key=selected_api_key,
                model=model,
                scope_description=scope_description or "",
                sub_domains=sub_domains or "",
                exploration_constant=exploration_constant,
                max_iterations=max_iterations,
                similarity_threshold=similarity_threshold,
                confidence_threshold=confidence_threshold,
                candidates_per_iteration=candidates_per_iteration,
                cross_link_threshold=cross_link_threshold,
                retirement_limit=retirement_limit,
                initial_seed_terms=initial_seed_terms,
                max_workers=max_workers,
            )
        except (RuntimeError, ValueError) as exc:
            return no_update, str(exc), "form-status form-status-error", True

        return (
            f"start-{n_clicks}",
            "Initialization started. Live logs are streaming below.",
            "form-status form-status-warning",
            False,
        )

    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("initialize-status", "children", allow_duplicate=True),
        Output("initialize-status", "className", allow_duplicate=True),
        Output("initialize-log-output", "children"),
        Output("initialize-interval", "disabled", allow_duplicate=True),
        Input("initialize-interval", "n_intervals"),
        Input("initialize-command-signal", "children"),
        State("url", "pathname"),
        prevent_initial_call=True,
    )
    def refresh_initialization(
        _interval: int,
        _command_signal: str | None,
        pathname: str | None,
    ):
        if pathname != "/":
            raise PreventUpdate

        snapshot = get_app_state().snapshot()
        logs = snapshot["recent_logs"]
        log_output = "\n".join(logs[-80:]) if logs else "No logs yet."
        status = snapshot["generation_status"]

        if status == "initializing":
            return (
                no_update,
                snapshot["last_message"],
                "form-status form-status-warning",
                log_output,
                False,
            )

        if snapshot["last_error"]:
            return (
                no_update,
                snapshot["last_error"],
                "form-status form-status-error",
                log_output,
                True,
            )

        if snapshot["has_ontology"] and status == "ready":
            summary = snapshot["seed_summary"]
            return (
                "/explorer",
                (
                    f"Seed ready: {summary['top_level_classes']} top-level classes, "
                    f"{summary['nodes']} nodes, {summary['edges']} edges."
                ),
                "form-status form-status-success",
                log_output,
                True,
            )

        return (
            no_update,
            snapshot["last_message"],
            "form-status",
            log_output,
            True,
        )
