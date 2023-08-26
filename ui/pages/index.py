import dash
from dash import html

dash.register_page(__name__, path_template="/")


def layout():
    return html.Div(
        children=[
            html.H1("Welcome to Expert GPTs!", className="p-2 mx-auto text-center"),
            html.P("Index WIP!", className="p-2 mx-auto text-justify"),
        ],
        className="h-100 mx-0 w-100",
    )
