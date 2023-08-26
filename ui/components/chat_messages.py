import dash_bootstrap_components as dbc
from dash import html


def get_user_chat_item(text, message_sent_at) -> dbc.Card:
    return dbc.Card(
        children=[
            dbc.CardBody(
                children=[
                    dbc.Badge(
                        message_sent_at,
                        color="secondary",
                        className="p-1 text-muted text-xs float-start my-auto width-auto",
                    ),
                    html.P(text, className="m-0"),
                ],
                className="p-2 flex",
            ),
        ],
        className="p-2 mb-4 bg-info text-end",
    )


def get_system_chat_item(text, show_btn_actions: bool = False) -> dbc.Card:
    return dbc.Card(
        children=[
            dbc.CardBody(
                children=[
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    *[html.P(t) for t in text.split("\n")],
                                ],
                                className="col-10 text-wrap",
                            ),
                            dbc.Col(
                                children=[
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button(
                                                "👍", color="success", className="p-1"
                                            ),
                                            dbc.Button(
                                                "👎", color="danger", className="p-1"
                                            ),
                                            dbc.Button(
                                                "🤷", color="warning", className="p-1"
                                            ),
                                        ],
                                        className="p-1",
                                    )
                                ],
                                className="col-2",
                            )
                            if show_btn_actions
                            else None,
                        ],
                        className="p-2",
                    )
                ],
                className="p-2",
            )
        ],
        className="p-2 mb-4 text-bg-primary text-start",
    )
