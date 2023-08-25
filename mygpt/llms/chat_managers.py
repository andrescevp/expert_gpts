import os
from typing import List, Optional

from langchain.agents import Tool
from langchain.memory import PostgresChatMessageHistory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate, SystemMessage

from mygpt.llms.base import BaseLLMManager
from mygpt.memory.base import MemoryBase
from shared.config import ExpertItem

DEFAULT_EXPERT_CONFIG = ExpertItem()


def get_history(session_id):
    return PostgresChatMessageHistory(
        connection_string=os.getenv("DATABASE_URL"),
        session_id=session_id,
    )


class SingleChatManager:
    def __init__(
        self,
        llm_manager: BaseLLMManager,
        expert_key: str,
        expert_config: ExpertItem = DEFAULT_EXPERT_CONFIG,
        session_id: str = "same-session",
        memory: Optional[MemoryBase] = None,
    ):
        self.memory = memory
        self.expert_config = expert_config
        self.expert_key = expert_key
        self.llm_manager = llm_manager
        self.history = get_history(session_id)

    def ask(self, question):
        context = None
        if self.memory:
            context = self.memory.search(question)
            self.history.add_ai_message("Memory: " + context)

        template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=self.expert_config.prompts.system,
                    name=f"{self.expert_key}ChatBot",
                ),
                HumanMessagePromptTemplate.from_template(
                    "History: {history}, Context: {context} Question: {question}"
                ),
            ]
        )
        self.history.add_user_message(question)
        answer = self.llm_manager.create_chat_completion(
            template.format_messages(
                question=question, context=context, history=self.history.messages
            ),
            temperature=self.expert_config.temperature,
            max_tokens=self.expert_config.max_tokens,
            model=self.expert_config.model,
        )

        self.history.add_ai_message(answer)
        return answer


class ChainChatManager:
    def __init__(
        self,
        llm_manager: BaseLLMManager,
        temperature: float = 0,
        max_tokens: int | None = None,
        tools: Optional[List[Tool]] = None,
        model: str | None = None,
        session_id: str = "same-session",
    ):
        self.model = model
        self.tools = tools
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm_manager = llm_manager
        self.history = get_history(session_id)

    def ask(self, question):
        self.history.add_user_message(question)
        answer = self.llm_manager.create_chat_completion_with_agent(
            question,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model=self.model,
            tools=self.tools,
            history=[x for x in self.history.messages],
        )

        self.history.add_ai_message(answer)
        return answer
