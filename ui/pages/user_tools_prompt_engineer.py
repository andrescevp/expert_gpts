import datetime
import uuid
from dataclasses import asdict

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from expert_gpts.llms.expert_agents import ExpertAgentManager
from ui.components.chat_messages import get_system_chat_item, get_user_chat_item
from ui.pages.states.web_chat_states import WebChatPageState

COMPONENT_ID = "prompt-engineer"

dash.register_page(__name__, path_template="/user_tools/prompt_engineer")

# form with a textarea and a submit button to send a message to the chat
form = [
    dbc.Card(
        children=[],
        className="p-2 overflow-y-scroll",
        id="%s-chat-history" % COMPONENT_ID,
        style={"height": "calc(100vh - 400px)"},
    ),
    dbc.Card(
        children=[
            dbc.InputGroup(
                [
                    dbc.Textarea(
                        placeholder="my questions", id="%s-user-prompt" % COMPONENT_ID
                    ),
                    dbc.Button("Send", id="%s-send-button" % COMPONENT_ID),
                ],
            ),
            dbc.Badge(
                [
                    "Last message sent at:",
                    html.Span(
                        "-1",
                        id=("%s-last-message-time" % COMPONENT_ID),
                        className="mx-2",
                    ),
                ],
                color="secondary",
                className="mt-2 p-2 text-end",
            ),
        ],
        className="p-2",
    ),
]


def layout():
    title = "Prompt Engineer tool"
    description = "Prompt Engineer tool. This tool allows you to create a prompt for a specific chatbot."
    return dbc.Container(
        children=[
            dcc.Store(id=("%s-web-chat-page-memory" % COMPONENT_ID)),
            dcc.Store(id="session", data=dict(uid=str(uuid.uuid4()))),
            html.H1(title, className="p-2 mx-auto text-center"),
            dbc.Alert(description, className="p-2 mx-auto text-justify", color="info"),
            *form,
        ],
        className="h-100 mx-auto w-100",
    )


@dash.callback(
    Output("%s-chat-history" % COMPONENT_ID, "children", allow_duplicate=True),
    Output("%s-last-message-time" % COMPONENT_ID, "children"),
    Output("%s-web-chat-page-memory" % COMPONENT_ID, "data", allow_duplicate=True),
    Input("%s-last-message-time" % COMPONENT_ID, "children"),
    Input("%s-send-button" % COMPONENT_ID, "n_clicks"),
    Input("%s-send-button" % COMPONENT_ID, "n_clicks_timestamp"),
    Input("%s-user-prompt" % COMPONENT_ID, "value"),
    Input("%s-chat-history" % COMPONENT_ID, "children"),
    prevent_initial_call=True,
)
def add_chat_item(last_send, n_clicks, n_clicks_timestamp, user_prompt, history):
    if not n_clicks:
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
                current_expert=COMPONENT_ID, current_user_prompt=user_prompt
            )
        )
        if history:
            chats = history + chats
        return [chats, n_clicks_timestamp, default_memory_state]
    else:
        return dash.no_update


@dash.callback(
    Output("%s-chat-history" % COMPONENT_ID, "children", allow_duplicate=True),
    Output("%s-web-chat-page-memory" % COMPONENT_ID, "data", allow_duplicate=True),
    State("%s-web-chat-page-memory" % COMPONENT_ID, "data"),
    Input("%s-chat-history" % COMPONENT_ID, "children"),
    prevent_initial_call=True,
)
def get_answer(data, chats_prevs):
    if not data:
        return dash.no_update
    experts_manager = ExpertAgentManager()
    data = WebChatPageState(**data)
    if data.answer:
        return dash.no_update
    answer = experts_manager.optimize_prompt(data.current_user_prompt)
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
