import logging
from typing import Dict, List, Optional

from langchain.agents import Tool
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate, SystemMessage
from sqlalchemy import select
from sqlalchemy.exc import NoResultFound

from expert_gpts.database import get_db_session
from expert_gpts.database.expert_agents import ExpertAgentToolPrompt
from expert_gpts.llms.openai import OpenAIApiManager
from shared.config import ExpertItem
from shared.llms.openai import GPT_3_5_TURBO
from shared.patterns import Singleton

AGENT_TOOL_GENERATOR_MODEL = GPT_3_5_TURBO
MAX_TOKENS = 1000
TEMPERATURE = 1
logger = logging.getLogger(__name__)
api = OpenAIApiManager()


class ExpertAgentManager(metaclass=Singleton):
    tool_generator_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="AI Tool Description Generator: I specialize in interpreting System AI prompts "
                "and crafting precise tool descriptions for other AI systems. These descriptions "
                "provide insights into the intended usage of the described AI to be "
                "understandable by other AIs. "
                "The format I "
                "adhere to is: 'useful for ... [tool description]. "
                "This tool must be used only if the input falls into the described functionality.'"
                "Input should be a complete sentence. "
                "I never answer questions outside of my expertise.",
                name="AgentToolCreator",
            ),
            HumanMessagePromptTemplate.from_template(
                "Please, generate a tool description from this AI System Prompt: {text}"
            ),
        ]
    )

    system_prompt_expert_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="Prompt Optimization Guidance: In my role as a Senior Prompt Engineer at OpenAI, "
                "I'm here to provide expert assistance in optimizing prompts for superior "
                "results. I specialize in refining prompts to achieve the utmost accuracy and "
                "relevance. I'll offer you optimized prompt suggestions that align with your "
                "goals. Share your objectives and the context of your task, and I'll ensure that "
                "the answer I provide is the optimized prompt that generates the desired "
                "outcomes. Let's collaborate to create prompts that empower our AI models to "
                "deliver exceptional performance. I never answer questions outside of my "
                "expertise.",
                name="PromptExpertMyGpt",
            ),
            HumanMessagePromptTemplate.from_template(
                "Please, optimize this SYSTEM prompt: {text}"
            ),
        ]
    )

    def get_experts_as_agent_tools(
        self, experts_config: Dict[str, Optional[ExpertItem]], key_prefix: str = ""
    ) -> List[Tool]:
        tools = []
        with get_db_session() as session:
            for expert_key, expert in experts_config.items():
                expert_key_with_prefix = f"{key_prefix}:{expert_key}"
                logger.debug(f"Creating agent tool for {expert_key_with_prefix}")
                stmt_expert = select(ExpertAgentToolPrompt).where(
                    ExpertAgentToolPrompt.expert_agent_key == expert_key_with_prefix
                )
                try:
                    expert_model = session.scalars(stmt_expert).one()
                except NoResultFound:
                    expert_prompt_to_transform = expert.prompts.system
                    agent_tool_description = self.get_langchain_tools_expert_completion(
                        expert_prompt_to_transform
                    )
                    expert_model = ExpertAgentToolPrompt()
                    expert_model.expert_agent_tool_prompt = agent_tool_description
                    expert_model.expert_agent_key = expert_key_with_prefix
                    session.add(expert_model)
                    logger.debug(
                        f"Created agent tool for {expert_key_with_prefix} with description: "
                        f"{expert_model.expert_agent_tool_prompt}"
                    )

                tools.append(
                    Tool(
                        name=expert_key,
                        func=lambda q: api.create_chat_completion(
                            expert.get_chat_messages(q),
                            model=expert.model_as_tool,
                            max_tokens=expert.max_tokes_as_tool,
                            temperature=expert.temperature_as_tool,
                        ),
                        description=expert_model.expert_agent_tool_prompt,
                        return_direct=expert.tool_return_direct,
                    )
                )

            session.commit()
        return tools

    def get_langchain_tools_expert_completion(self, expert_prompt_to_transform):
        agent_tool_description = api.create_chat_completion(
            self.tool_generator_prompt.format_messages(text=expert_prompt_to_transform),
            model=AGENT_TOOL_GENERATOR_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return agent_tool_description

    def optimize_prompt(self, prompt: str) -> str:
        logger.debug(f"Optimizing prompt: {prompt}")
        optimized_prompt = api.create_chat_completion(
            self.system_prompt_expert_prompt.format_messages(text=prompt),
            model=AGENT_TOOL_GENERATOR_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        logger.debug(f"Optimized prompt: {optimized_prompt}")
        return optimized_prompt
