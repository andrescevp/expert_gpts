from dataclasses import dataclass


@dataclass
class WebChatPageState:
    """
    {'current_user_prompt': user_prompt, 'current_expert': current_expert, 'answer': None}
    """

    current_user_prompt: str
    current_expert: str
    answer: str = None
