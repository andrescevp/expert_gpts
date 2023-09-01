from expert_gpts.llms.chat_managers import get_history


def get_chats_list(config_key, session_id):
    chat_history = get_history(session_id, config_key)
    return chat_history.get_chats_sessions()
