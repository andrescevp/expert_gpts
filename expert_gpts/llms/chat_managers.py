import os
from typing import List, Optional

from langchain.agents import Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import PostgresChatMessageHistory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage

from expert_gpts.memory.base import EmbeddingsHandlerBase
from shared.config import ExpertItem
from shared.llm_manager_base import BaseLLMManager
from shared.llms.system_prompts import CHAT_HUMAN_PROMPT_TEMPLATE

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
        embeddings: Optional[EmbeddingsHandlerBase] = None,
        query_memory_before_ask: bool = True,
    ):
        self.query_memory_before_ask = query_memory_before_ask
        self.embeddings = embeddings
        self.expert_config = expert_config
        self.expert_key = expert_key
        self.llm_manager = llm_manager
        self.history = get_history(session_id)

    def ask(self, question):
        context = None
        if self.embeddings and self.query_memory_before_ask:
            context = self.embeddings.search(question)
            self.history.add_ai_message("Memory: " + context.response)

        template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=self.expert_config.prompts.system,
                    name=f"{self.expert_key}ChatBot",
                    role="system",
                ),
                CHAT_HUMAN_PROMPT_TEMPLATE,
            ]
        )
        self.history.add_user_message(question)
        answer = self.llm_manager.create_chat_completion(
            template.format_messages(
                question=question,
                context=context.response,
                history=[x.content for x in self.history.messages][:5],
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
        agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory: Optional[EmbeddingsHandlerBase] = None,
        query_memory_before_ask: bool = True,
    ):
        self.query_memory_before_ask = query_memory_before_ask
        self.memory = memory
        self.agent_type = agent_type
        self.model = model
        self.tools = tools
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm_manager = llm_manager
        self.history = get_history(session_id)

    def ask(self, question):
        context = None
        history = [x.content for x in self.history.messages][:5]
        if self.memory and self.query_memory_before_ask:
            context = self.memory.search(question)
            self.history.add_ai_message("Memory: " + context.response)
            context = context.response
        template = ChatPromptTemplate.from_messages(
            [
                CHAT_HUMAN_PROMPT_TEMPLATE,
            ]
        )
        self.history.add_user_message(question)
        answer = self.llm_manager.create_chat_completion_with_agent(
            "\n".join(
                [
                    x.content
                    for x in template.format_messages(
                        question=question,
                        context=context,
                        history=history,
                    )
                ]
            ),
            agent_type=self.agent_type,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model=self.model,
            tools=self.tools,
        )

        self.history.add_ai_message(answer)
        return answer
