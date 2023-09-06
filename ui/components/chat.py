import dash_bootstrap_components as dbc
from dash import html

from ui.components.chat_messages import get_system_chat_item
from ui.components.chat_terminal import TERMINAL_COMPONENT_TEMPLATE

CHAT_COMPONENT_TEMPLATE = [
    dbc.Card(
        children=[
            get_system_chat_item(
                "Hello I am a super assistant! I am able "
                "to "
                "connect to others experts and tools in "
                "this app to "
                "help you."
            )
        ],
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
            TERMINAL_COMPONENT_TEMPLATE,
        ],
        className="p-2",
    ),
]
