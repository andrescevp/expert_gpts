import datetime
import uuid
from dataclasses import asdict

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, dcc, html

from expert_gpts.llms.main import LLMConfigBuilder
from ui.components.chat_messages import get_system_chat_item, get_user_chat_item
from ui.config import get_configs
from ui.pages.states.web_chat_states import WebChatPageState

dash.register_page(__name__, path_template="/expert_chat/<config_key>")

configurations = get_configs()


def layout(config_key=None):
    config = configurations[config_key].config
    return html.Div(
        children=[
            html.H1(id="chat_title", className="p-2 mx-auto text-center"),
            dcc.Store(id="config_key", data=config_key),
            dcc.Store(id="web-chat-page-memory"),
            dcc.Store(id="session", data=dict(uid=str(uuid.uuid4()))),
            dcc.Location(id="url"),
            html.Span(id="current_expert", className="d-none"),
            dbc.Row(
                children=[
                    dbc.Col(
                        children=[
                            html.H4(
                                "Available chats", className="p-2 mx-auto text-center"
                            ),
                            dbc.ListGroup(
                                [
                                    dbc.ListGroupItem(
                                        dbc.Button(
                                            f"{configurations[config_key].config.chain.chain_key} Chain Chat",
                                            id={
                                                "type": "btn-expert-init",
                                                "index": "chain",
                                            },
                                            className="chain text-break",
                                            value="chain",
                                            color="primary",
                                            style={"width": "100%"},
                                        )
                                    ),
                                ]
                                + [
                                    dbc.ListGroupItem(
                                        dbc.Button(
                                            f"{expert} Chat",
                                            id={
                                                "type": "btn-expert-init",
                                                "index": expert,
                                            },
                                            className=f"{expert} text-break",
                                            value=expert,
                                            color="secondary",
                                        )
                                    )
                                    for expert, expert_config in config.experts.__root__.items()
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
                                                placeholder="my questions",
                                                id="user-prompt",
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
                className="w-100 h-100 mx-0",
            ),
        ],
        className="h-100 mx-0 w-100",
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
    Input("config_key", "data"),
    Input("chat-history", "children"),
    prevent_initial_call=True,
)
def get_answer(data, session, config_key, chats_prevs):
    if not data:
        return dash.no_update
    config = configurations[config_key].config
    builder = LLMConfigBuilder(config)
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
    Input("config_key", "data"),
    Input({"type": "btn-expert-init", "index": ALL}, "n_clicks"),
    Input({"type": "btn-expert-init", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def get_expert_chat(config_key, n_clicks, value):
    config = configurations[config_key].config
    valued_n_clicks = [
        (n_click, val) for n_click, val in zip(n_clicks, value) if n_click
    ]
    last_clicked = sorted(valued_n_clicks, key=lambda x: x[0])[-1][1]
    if last_clicked == "chain":
        return [
            [
                get_system_chat_item(
                    "Hello I am a super assistant! I am able to connect to others experts and tools in "
                    "this app to "
                    "help you."
                )
            ],
            last_clicked,
            f"{configurations[config_key].config.chain.chain_key} Chain",
        ]
    return [
        [get_system_chat_item(f"Hello I am {last_clicked} assistant!")],
        last_clicked,
        config.experts.__root__[last_clicked].name or last_clicked,
    ]
