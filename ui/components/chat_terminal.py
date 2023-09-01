import dash_bootstrap_components as dbc
from dash import dcc, html

TERMINAL_COMPONENT_TEMPLATE = dbc.Card(
    [
        dbc.CardHeader(
            [
                "Terminal - " "LangChainLog",
                dbc.Button(
                    "Download",
                    id="download-log",
                    className="float-end",
                ),
            ]
        ),
        dbc.CardBody(
            [
                html.Pre(
                    style={
                        "height": "900px",
                        "width": "100%",
                        "overflow": "auto",
                    },
                    id="terminal",
                ),
                dcc.Download(id="download-terminal-log"),
                dcc.Store(id="last-log-download", data=0),
            ],
            style={
                "background-color": "black",
                "color": "white",
            },
        ),
    ],
    className="mt-2",
)
