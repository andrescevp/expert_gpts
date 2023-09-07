import abc
import logging
from functools import lru_cache

from expert_gpts.embeddings.factory import EmbeddingsHandlerFactory
from expert_gpts.llms.chat_managers import (
    ChainChatManager,
    PlannerManager,
    get_history,
    get_memory,
)
from expert_gpts.llms.expert_agents import ExpertAgentManager
from expert_gpts.llms.providers.openai import OpenAIApiManager
from expert_gpts.toolkit.modules import ModuleLoader
from shared.config import Config
from shared.llms.system_prompts import BASE_EXPERTS

logger = logging.getLogger(__name__)


class LLMConfigBuilderSingleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        cls_key = None
        if cls not in cls._instances:
            config_key = args[0].chain.chain_key
            cls_key = f"llm_config_{config_key}"
            cls._instances[cls_key] = super(LLMConfigBuilderSingleton, cls).__call__(
                *args, **kwargs
            )

        return cls._instances[cls_key]


class LLMConfigBuilder(metaclass=LLMConfigBuilderSingleton):
    _expert_tools = {}

    def __init__(self, config: Config):
        self.config = config
        self.module_loader = ModuleLoader()
        self.llm_manager = OpenAIApiManager()
        self.expert_agent_manager = ExpertAgentManager()
        self.embeddings_factory = EmbeddingsHandlerFactory()
        self.default_tools = {}
        if self.config.enabled_default_agent_tools:
            self.default_tools = {
                x.value: BASE_EXPERTS[x.value]
                for x in self.config.enabled_default_agent_tools
            }
        self.default_experts = {}
        if self.config.enabled_default_experts:
            self.default_experts = {
                x.value: BASE_EXPERTS[x.value]
                for x in self.config.enabled_default_experts
            }
        self.experts_map = {**self.default_experts, **self.config.experts.__root__}

        self.custom_tools = []
        if self.config.custom_tools:
            self.config.custom_tools = self.module_loader.build_module(
                self.config.custom_tools
            )
            self.custom_tools = self.config.custom_tools.get_attribute_built()

    @lru_cache
    def get_expert_chat(self, expert_key: str, session_id: str = "same-session"):
        return self.expert_agent_manager.get_expert_chat(
            self.experts_map,
            self.llm_manager,
            self.embeddings_factory,
            expert_key,
            session_id,
        )

    def get_expert_tools(
        self,
        prefix: str = "",
        session_id: str = "same-session",
        history=None,
        memory=None,
    ):
        expert_tools_key = f"{prefix}_{session_id}_expert_tools"
        if expert_tools_key not in self._expert_tools:
            experts_as_tools = {
                **{k: v for k, v in self.default_tools.items() if v.use_as_tool},
                **{
                    k: v
                    for k, v in self.config.experts.__root__.items()
                    if v.use_as_tool
                },
            }
            expert_tools = self.expert_agent_manager.get_experts_as_agent_tools(
                experts_as_tools,
                prefix,
                session_id=session_id,
                history=history,
                memory=memory,
            )
            self._expert_tools[expert_tools_key] = expert_tools

        return self._expert_tools[expert_tools_key]

    @lru_cache
    def get_planner(
        self, session_id: str = "same-session", memory_key: str = "chat_history"
    ):
        embeddings = self.embeddings_factory.get_chain_embeddings(
            self.llm_manager,
            embeddings=self.config.planner.embeddings.__root__
            if self.config.planner.embeddings
            else None,
            load_docs=False,
            index_name=self.config.planner.chain_key,
            index_prefix=f"{self.config.planner.chain_key}_",
        )

        embeddings_tools = []
        if self.config.chain.get_embeddings_as_tool:
            embeddings_tools.append(
                embeddings.get_embeddings_tool_get_memory(
                    tool_key=self.config.planner.chain_key
                )
            )
        if self.config.chain.save_embeddings_as_tool:
            embeddings_tools.append(
                embeddings.get_embeddings_tool_save_memory(
                    tool_key=self.config.planner.chain_key
                )
            )
        return PlannerManager(
            self.llm_manager,
            temperature=self.config.planner.temperature,
            max_tokens=self.config.planner.max_tokens,
            tools=embeddings_tools
            + self.get_expert_tools(
                prefix=self.config.planner.chain_key,
            )
            + self.custom_tools,
            model=self.config.planner.model,
            chain_key=self.config.planner.chain_key,
        )

    @lru_cache
    def get_chain_chat(
        self, session_id: str = "same-session", memory_key: str = "chat_history"
    ):
        history = get_history(session_id, self.config.chain.chain_key)
        memory = get_memory(
            self.llm_manager,
            chat_memory=history,
            memory_type=self.config.chain.memory_type,
            memory_key=memory_key,
        )
        embeddings = self.embeddings_factory.get_chain_embeddings(
            self.llm_manager,
            embeddings=self.config.chain.embeddings.__root__
            if self.config.chain.embeddings
            else None,
            load_docs=False,
            index_name=self.config.chain.chain_key,
            index_prefix=f"{self.config.chain.chain_key}_",
        )

        embeddings_tools = []
        if self.config.chain.get_embeddings_as_tool:
            embeddings_tools.append(
                embeddings.get_embeddings_tool_get_memory(
                    tool_key=self.config.chain.chain_key
                )
            )
        if self.config.chain.save_embeddings_as_tool:
            embeddings_tools.append(
                embeddings.get_embeddings_tool_save_memory(
                    tool_key=self.config.chain.chain_key
                )
            )

        return ChainChatManager(
            self.llm_manager,
            temperature=self.config.chain.temperature,
            max_tokens=self.config.chain.max_tokens,
            tools=embeddings_tools
            + self.get_expert_tools(
                prefix=self.config.chain.chain_key,
                session_id=session_id,
                history=history,
                memory=memory,
            )
            + self.custom_tools,
            model=self.config.chain.model,
            session_id=session_id,
            chain_key=self.config.chain.chain_key,
            memory_type=self.config.chain.memory_type,
            memory=memory,
            history=history,
        )

    def load_docs(self):
        # main chain embeddings load
        self.embeddings_factory.get_chain_embeddings(
            self.llm_manager,
            self.config.chain.embeddings.__root__
            if self.config.chain.embeddings
            else None,
            load_docs=True,
            index_name=self.config.chain.chain_key,
            index_prefix=f"{self.config.chain.chain_key}_",
        )

        for dict_expert_key, expert_config in self.config.experts.__root__.items():
            self.embeddings_factory.get_expert_embeddings(
                self.llm_manager,
                dict_expert_key,
                expert_config.embeddings.__root__,
                load_docs=True,
            )
