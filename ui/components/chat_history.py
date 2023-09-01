import dash_bootstrap_components as dbc
from dash import dcc, html


def get_chat_item(session_id, created_at):
    return dbc.ListGroupItem(
        [
            dbc.Badge(created_at),
            html.Div(
                [
                    # html.P(session_id, className='text-break'),
                    dbc.Button(
                        [
                            html.I(className="bi bi-cloud-arrow-up me-2"),
                            html.Span(session_id[:8]),
                        ],
                        id={
                            "type": "btn-chat-session-load",
                            "index": session_id,
                        },
                        className="text-break",
                        value=session_id,
                        color="success",
                    ),
                    dbc.ButtonGroup(
                        [
                            dbc.Button(
                                html.I(className="bi bi-trash"),
                                id={
                                    "type": "btn-chat-session-remove",
                                    "index": session_id,
                                },
                                className="text-break",
                                value=session_id,
                                color="danger",
                            ),
                        ],
                        className="float-end my-2",
                    ),
                ],
                className="d-flex " "justify-content-between align-items-center",
            ),
        ]
    )


def create_chat_list(chats_uuids):
    return [
        dcc.ConfirmDialog(
            id="btn-chat-session-remove-confirm",
            message="Are you sure you want to remove this chat?",
        )
    ] + [
        get_chat_item(session_id, chats_uuids[session_id])
        for session_id in chats_uuids.keys()
    ]
