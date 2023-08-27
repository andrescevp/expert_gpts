import os
from typing import List, Optional

from langchain.agents import Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import PostgresChatMessageHistory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate, SystemMessage

from expert_gpts.memory.base import MemoryBase
from shared.config import ExpertItem
from shared.llm_manager_base import BaseLLMManager

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
            self.history.add_ai_message("Memory: " + context.response)

        template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=self.expert_config.prompts.system,
                    name=f"{self.expert_key}ChatBot",
                    role="system",
                ),
                HumanMessagePromptTemplate.from_template(
                    """
                        Use the History and Context to look for relevant information about the Question.
                        History: {history}
                        Context: {context}
                        Question: {question}
                        """,
                    role="user",
                ),
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
        memory: Optional[MemoryBase] = None,
    ):
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
        if self.memory:
            context = self.memory.search(question)
            self.history.add_ai_message("Memory: " + context.response)
        self.history.add_user_message(question)
        user_input = f"""
            Use the History and Context to look for relevant information about the Question.
            History: {[x.content for x in self.history.messages][:5]}
            Context: {context.response}
            Question: {question}
        """
        answer = self.llm_manager.create_chat_completion_with_agent(
            user_input,
            agent_type=self.agent_type,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model=self.model,
            tools=self.tools,
        )

        self.history.add_ai_message(answer)
        return answer
