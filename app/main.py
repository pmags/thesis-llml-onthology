from __future__ import annotations
import os

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html, no_update

from dash.exceptions import PreventUpdate

from app.components import create_sidebar, create_topbar, empty_state
from app.pages import automation, explorer, export, initialize
from app.state import get_app_state

BRITE_THEME = "https://bootswatch.com/5/brite/bootstrap.min.css"


CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


def create_app() -> Dash:
    """Create and configure the Dash application."""
    app = Dash(__name__, suppress_callback_exceptions=True, update_title=None, external_stylesheets=[BRITE_THEME])

    # Define layout and app conditions
    app.title = "ontogen"

    app.layout = html.Div(
        className="app-root",
        children=[
            html.Div(
                id="sidebar-container",
                children=[
                    dcc.Location(id="url"),
                    create_sidebar()
                ]
            ),
            html.Div(
                className="main-column",
                children=[
                    html.Div(
                        id="page-content", 
                        className="page-content"),
                    ],
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Header")),
                    dbc.ModalBody("An extra large modal."),
                ],
                id="modal-xl",
                size="xl",
                is_open=False,
            ),
        ],

    )

    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname):
        if pathname == "/":
            return html.P("This is the content of the home page!")
        elif pathname == "/page-1":
            return html.P("This is the content of page 1. Yay!")
        elif pathname == "/page-2":
            return html.P("Oh cool, this is page 2!")
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
        Input("generate-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_generate_modal(n_clicks):
        return True

    return app



def main() -> None:
    """Launch the Dash application."""
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=True)


if __name__ == "__main__":
    main()
