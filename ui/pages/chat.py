import datetime
import os
import uuid
from dataclasses import asdict, dataclass

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, dcc, html

from mygpt.llms.main import LLMConfigBuilder
from shared.config import load_config

dash.register_page(__name__, path_template="/chat")

config = load_config(os.getenv("CONFIG_PATH", "mygpt.yaml"))
builder = LLMConfigBuilder(config)


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


def get_system_chat_item(text) -> dbc.Card:
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
                            ),
                        ],
                        className="p-2",
                    )
                ],
                className="p-2",
            )
        ],
        className="p-2 mb-4 text-bg-primary text-start",
    )


@dataclass
class WebChatPageState:
    """
    {'current_user_prompt': user_prompt, 'current_expert': current_expert, 'answer': None}
    """

    current_user_prompt: str
    current_expert: str
    answer: str = None


layout = dbc.Container(
    children=[
        dcc.Store(id="web-chat-page-memory"),
        dcc.Store(id="session", data=dict(uid=str(uuid.uuid4()))),
        dcc.Location(id="url"),
        html.Span(id="current_expert", className="d-none"),
        html.H1(id="chat_title"),
        dbc.Row(
            children=[
                dbc.Col(
                    children=[
                        dbc.ListGroup(
                            [
                                dbc.ListGroupItem(
                                    dbc.Button(
                                        f"{expert} bot",
                                        id={"type": "btn-expert-init", "index": expert},
                                        className=expert,
                                        value=expert,
                                    )
                                )
                                for expert, expert_config in config.experts.__root__.items()
                            ]
                            + [
                                dbc.ListGroupItem(
                                    dbc.Button(
                                        " bot",
                                        id={
                                            "type": "btn-expert-init",
                                            "index": "chain",
                                        },
                                        className="chain",
                                        value="chain",
                                    )
                                ),
                            ]
                        ),
                    ],
                    id="agents-container",
                    className="col-2",
                ),
                dbc.Col(
                    children=[
                        dbc.Card(
                            children=[],
                            className="p-2 overflow-y-scroll",
                            id="chat-history",
                            style={"height": "calc(100vh - 400px)"},
                        ),
                        dbc.Card(
                            children=[
                                dbc.InputGroup(
                                    [
                                        dbc.Textarea(
                                            placeholder="my questions", id="user-prompt"
                                        ),
                                        dbc.Button("Send", id="send-button"),
                                    ],
                                ),
                                dbc.Badge(
                                    [
                                        "Last message sent at:",
                                        html.Span(
                                            "-1",
                                            id="last-message-time",
                                            className="mx-2",
                                        ),
                                    ],
                                    color="secondary",
                                    className="mt-2 p-2 text-end",
                                ),
                            ],
                            className="p-2",
                        ),
                    ],
                    className="col-8",
                ),
                dbc.Col(children=[], className="col-2"),
            ],
            id="chat-container",
        ),
    ],
    className="h-100 mx-0",
)


@dash.callback(
    Output("chat-history", "children", allow_duplicate=True),
    Output("last-message-time", "children"),
    Output("web-chat-page-memory", "data", allow_duplicate=True),
    Input("current_expert", "children"),
    Input("last-message-time", "children"),
    Input("chat-history", "children"),
    Input("send-button", "n_clicks"),
    Input("send-button", "n_clicks_timestamp"),
    Input("user-prompt", "value"),
    prevent_initial_call=True,
)
def add_chat_item(
    current_expert, last_send, children, n_clicks, n_clicks_timestamp, user_prompt
):
    if not n_clicks or not current_expert:
        return dash.no_update
    # check if user actually hit the button
    if n_clicks_timestamp and last_send and int(n_clicks_timestamp) <= int(last_send):
        return dash.no_update
    message_sent_at = datetime.datetime.fromtimestamp(
        n_clicks_timestamp / 1000
    ).strftime("%Y-%m-%d %H:%M:%S")
    if user_prompt:
        chats = [get_user_chat_item(user_prompt, message_sent_at)]
        default_memory_state = asdict(
            WebChatPageState(
                current_expert=current_expert, current_user_prompt=user_prompt
            )
        )
        if not children:
            return [chats, n_clicks_timestamp, default_memory_state]
        return [children + chats, n_clicks_timestamp, default_memory_state]
    else:
        return dash.no_update


@dash.callback(
    Output("chat-history", "children", allow_duplicate=True),
    Output("web-chat-page-memory", "data", allow_duplicate=True),
    Input("web-chat-page-memory", "data"),
    Input("session", "data"),
    Input("chat-history", "children"),
    prevent_initial_call=True,
)
def get_answer(data, session, chats_prevs):
    if not data:
        return dash.no_update
    data = WebChatPageState(**data)
    if data.answer:
        return dash.no_update
    if data.current_expert != "chain":
        expert_chat = builder.get_expert_chat(
            data.current_expert, f"{data.current_expert}_{session['uid']}"
        )
        answer = expert_chat.ask(data.current_user_prompt)
    else:
        chain_chat = builder.get_chain_chat(session["uid"])
        answer = chain_chat.ask(data.current_user_prompt)
    chats = [get_system_chat_item(answer)]
    return [
        chats_prevs + chats,
        asdict(
            WebChatPageState(
                current_expert=data.current_expert,
                current_user_prompt=data.current_user_prompt,
                answer=answer,
            )
        ),
    ]


@dash.callback(
    Output("chat-history", "children", allow_duplicate=True),
    Output("current_expert", "children"),
    Output("chat_title", "children"),
    Input({"type": "btn-expert-init", "index": ALL}, "n_clicks"),
    Input({"type": "btn-expert-init", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def get_expert_chat(n_clicks, value):
    print(n_clicks, value)
    # zip n_clicks and value
    # get the last clicked button

    valued_n_clicks = [
        (n_click, val) for n_click, val in zip(n_clicks, value) if n_click
    ]
    last_clicked = sorted(valued_n_clicks, key=lambda x: x[0])[-1][1]
    if last_clicked == "chain":
        return [
            [
                get_system_chat_item(
                    "Hello I am a super assistant! I am able to connect to others experts and tools in this app to "
                    "help you."
                )
            ],
            last_clicked,
            last_clicked,
        ]
    return [
        [get_system_chat_item(f"Hello I am {last_clicked} assistant!")],
        last_clicked,
        config.experts.__root__[last_clicked].name or last_clicked,
    ]
