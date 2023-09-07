from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional

import langchain
from langchain.agents import AgentExecutor, Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import PromptTemplate
from langchain.schema.messages import BaseMessage
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)

from expert_gpts.llms.agent import HUMAN_SUFFIX, SYSTEM_PREFIX, ConvoOutputCustomParser
from shared.llm_manager_base import BaseLLMManager, Cost
from shared.llms.openai import GPT_3_5_TURBO, GPT_4, TEXT_ADA_EMBEDDING
from shared.llms.system_prompts import get_open_ai_prompt_template

langchain.debug = True

logger = logging.getLogger(__name__)

COSTS = {
    GPT_3_5_TURBO: Cost(prompt=0.0015, completion=0.002),
    GPT_4: Cost(prompt=0.03, completion=0.05),
    TEXT_ADA_EMBEDDING: Cost(prompt=0.0001, completion=0.0001),
}


PLANNER_SYSTEM_PROMPT = """
Let's first understand the problem and devise a plan to solve the problem.
Please output the plan starting with the header 'Plan:'
and then followed by a numbered list of steps.
Please make the plan the minimum number of steps required and try to make it with maximal 5 steps
to accurately complete the task. If the task is a question,
the final step should almost always be 'Given the above steps taken,
please respond to the users original question'.
At the end of your plan, say '<END_OF_PLAN>'
"""


class OpenAIApiManager(BaseLLMManager):
    _agents = {}

    def __init__(self):
        super().__init__(COSTS)

    def get_agent_executor(
        self,
        llm,
        agent_type: AgentType = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory: Optional[BaseChatMemory] = None,
        tools: Optional[List[Tool]] = None,
        system_message: Optional[str] = SYSTEM_PREFIX,
        human_message: Optional[str] = HUMAN_SUFFIX,
    ) -> AgentExecutor:
        agent_kwargs = {
            "output_parser": ConvoOutputCustomParser(),
        }
        if system_message:
            agent_kwargs["system_message"] = system_message
        if human_message:
            agent_kwargs["human_message"] = human_message
        return initialize_agent(
            tools=tools,
            llm=llm,
            agent=agent_type,
            memory=memory,
            agent_kwargs=agent_kwargs,
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
            response = llm(messages, callbacks=[self.callbacks_handler])
        self.update_cost(cb)
        return response.content

    def create_chat_completion_with_agent(
        self,
        user_input: str,  # type: ignore
        agent_type: AgentType = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        model: str | None = GPT_3_5_TURBO,
        agent_key: str = "default",
        temperature: float = 0,
        max_tokens: int | None = None,
        memory: Optional[BaseChatMemory] = None,
        tools: Optional[List[Tool]] = None,
    ) -> str:
        llm = self.get_llm(max_tokens, model, temperature)
        if agent_key not in self._agents:
            self._agents[agent_key] = self.get_agent_executor(
                llm, agent_type, memory, tools
            )
        agent = self._agents[agent_key]
        with get_openai_callback() as cb:
            response = agent.run(input=user_input, callbacks=[self.callbacks_handler])
        self.update_cost(cb)
        return response

    def execute_plan(
        self,
        user_input: str,  # type: ignore
        model: str | None = GPT_3_5_TURBO,
        agent_key: str = "default_plan",
        temperature: float = 0,
        max_tokens: int | None = None,
        tools: Optional[List[Tool]] = None,
    ) -> str:
        llm = self.get_llm(max_tokens, model, temperature)
        if agent_key not in self._agents:
            planner = load_chat_planner(llm, system_prompt=PLANNER_SYSTEM_PROMPT)
            executor = load_agent_executor(llm, tools, verbose=True)
            agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
            self._agents[agent_key] = agent
        agent = self._agents[agent_key]
        with get_openai_callback() as cb:
            response = agent.run(input=user_input, callbacks=[self.callbacks_handler])
        self.update_cost(cb)
        return response

    @lru_cache
    def get_llm(
        self, max_tokens, model, temperature, as_predictor: bool = False
    ) -> BaseChatModel:
        llm = ChatOpenAI(
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return llm

    def get_prompt_template(self) -> PromptTemplate:
        return get_open_ai_prompt_template()
