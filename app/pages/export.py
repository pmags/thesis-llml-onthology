"""Export page for RDF serialization preview and downloads."""

from __future__ import annotations

from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from app.components import empty_state, metric_card
from app.state import get_app_state


FORMAT_MAP = {
    "turtle": {"label": "Turtle", "extension": "ttl"},
    "json-ld": {"label": "JSON-LD", "extension": "jsonld"},
    "xml": {"label": "RDF/XML", "extension": "rdf"},
}


def _serialize_current_ontology(format_name: str) -> tuple[str, int, int, int]:
    """Build and serialize the current ontology in the requested format."""
    app_state = get_app_state()
    ontology = app_state.ontology
    if ontology is None:
        raise RuntimeError("No ontology available for export.")
    ontology.build_ontology()
    serialized = ontology.serialize_ontology(format=format_name)
    return (
        serialized,
        len(ontology.rdf),
        ontology.ontology_graph.number_of_nodes(),
        ontology.ontology_graph.number_of_edges(),
    )


def layout() -> html.Div:
    """Return the export page layout."""
    if not get_app_state().has_ontology():
        return empty_state(
            "Export & Registry Details",
            "Initialize an ontology before exporting RDF output.",
        )

    return html.Div(
        className="page page-export",
        children=[
            dcc.Download(id="download-ontology"),
            html.Div(
                className="page-header compact-header",
                children=[
                    html.H1("Export & Registry Details", className="page-title"),
                    html.P(
                        "Preview the generated RDF output and export it in standard formats.",
                        className="page-subtitle",
                    ),
                ],
            ),
            html.Div(
                className="page-grid export-grid",
                children=[
                    html.Div(
                        className="content-card export-controls-card",
                        children=[
                            html.Div("SELECT_FORMAT", className="panel-eyebrow"),
                            dcc.RadioItems(
                                id="export-format",
                                options=[
                                    {"label": "JSON_STRUCTURE", "value": "json-ld"},
                                    {"label": "TURTLE", "value": "turtle"},
                                    {"label": "XML_SCHEMA", "value": "xml"},
                                ],
                                value="turtle",
                                className="radio-list-vertical",
                                inputClassName="radio-input",
                                labelClassName="radio-card-label",
                            ),
                            html.Div(id="export-status", className="form-status"),
                            html.Button(
                                "EXECUTE_EXPORT",
                                id="download-ontology-button",
                                className="primary-button large-button",
                                type="button",
                            ),
                        ],
                    ),
                    html.Div(
                        className="content-card export-preview-card",
                        children=[
                            html.Div("SCHEMA_PREVIEW", className="panel-eyebrow"),
                            html.Pre(id="export-preview", className="code-preview"),
                        ],
                    ),
                ],
            ),
            html.Div(id="export-metrics-row", className="metric-row"),
        ],
    )


def register_callbacks(app: Dash) -> None:
    """Register export-page callbacks."""

    @app.callback(
        Output("export-preview", "children"),
        Output("export-metrics-row", "children"),
        Output("export-status", "children"),
        Output("export-status", "className"),
        Input("export-format", "value"),
        State("url", "pathname"),
    )
    def refresh_export_preview(format_name: str, pathname: str | None):
        if pathname != "/export":
            raise PreventUpdate
        try:
            serialized, triple_count, node_count, edge_count = _serialize_current_ontology(format_name)
        except RuntimeError as exc:
            return "", [], str(exc), "form-status form-status-error"

        preview = "\n".join(serialized.splitlines()[:50])
        metrics = [
            metric_card("Triple Count", triple_count),
            metric_card("Node Count", node_count),
            metric_card("Edge Count", edge_count),
            metric_card("Format", FORMAT_MAP[format_name]["label"]),
        ]
        return preview, metrics, "Export preview ready.", "form-status form-status-success"

    @app.callback(
        Output("download-ontology", "data"),
        Input("download-ontology-button", "n_clicks"),
        State("export-format", "value"),
        prevent_initial_call=True,
    )
    def download_export(n_clicks: int | None, format_name: str):
        if not n_clicks:
            raise PreventUpdate

        serialized, _, _, _ = _serialize_current_ontology(format_name)
        snapshot = get_app_state().snapshot()
        domain = snapshot["domain"].strip().lower().replace(" ", "_") or "ontology"
        extension = FORMAT_MAP[format_name]["extension"]
        filename = f"{domain}_ontology.{extension}"
        return dcc.send_string(serialized, filename)
