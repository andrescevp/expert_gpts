from __future__ import annotations

import logging
from typing import List, Optional

import langchain
from langchain.agents import AgentExecutor, Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema.messages import BaseMessage

from mygpt.llms.base import BaseLLMManager, Cost
from shared.llms.openai import GPT_3_5_TURBO, GPT_4, TEXT_ADA_EMBEDDING

langchain.debug = True

logger = logging.getLogger(__name__)

COSTS = {
    GPT_3_5_TURBO: Cost(prompt=0.0015, completion=0.002),
    GPT_4: Cost(prompt=0.03, completion=0.05),
    TEXT_ADA_EMBEDDING: Cost(prompt=0.0001, completion=0.0001),
}


class OpenAIApiManager(BaseLLMManager):
    def __init__(self):
        super().__init__(COSTS)

    def get_agent_executor(
        self,
        llm,
        llm_key: AgentType = "zero-shot-react-description",
        memory: Optional[BaseChatMemory] = None,
        tools: Optional[List[Tool]] = None,
    ) -> AgentExecutor:
        return initialize_agent(
            tools=tools,
            llm=llm,
            agent=llm_key,
            memory=memory,
            handle_parsing_errors="Check your output and make sure it conforms!",
        )

    def create_chat_completion(
        self,
        messages: List[BaseMessage],  # type: ignore
        model: str | None = GPT_3_5_TURBO,
        temperature: float = 0,
        max_tokens: int | None = None,
        deployment_id=None,
        openai_api_key=None,
    ) -> str:
        llm = self.get_llm(max_tokens, model, temperature)
        with get_openai_callback() as cb:
            response = llm(messages)
        self.update_cost(cb)
        return response.content

    def create_chat_completion_with_agent(
        self,
        user_input: str,  # type: ignore
        agent_name: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        model: str | None = GPT_3_5_TURBO,
        temperature: float = 0,
        max_tokens: int | None = None,
        memory: Optional[BaseChatMemory] = None,
        tools: Optional[List[Tool]] = None,
        history: Optional[List[BaseMessage]] = None,
    ) -> str:
        llm = self.get_llm(max_tokens, model, temperature)
        agent = self.get_agent_executor(llm, agent_name, memory, tools)
        with get_openai_callback() as cb:
            response = agent.run(input=user_input, chat_history=history)
        self.update_cost(cb)
        return response

    def get_llm(self, max_tokens, model, temperature) -> BaseChatModel:
        llm = ChatOpenAI(
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return llm
