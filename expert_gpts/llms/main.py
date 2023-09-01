import abc
import logging
from functools import lru_cache

from expert_gpts.embeddings.factory import EmbeddingsHandlerFactory
from expert_gpts.llms.chat_managers import ChainChatManager, SingleChatManager
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
        for dict_expert_key, expert_config in self.experts_map.items():
            if dict_expert_key == expert_key:
                return SingleChatManager(
                    self.llm_manager,
                    expert_key,
                    expert_config=expert_config,
                    session_id=session_id,
                    embeddings=self.embeddings_factory.get_expert_embeddings(
                        self.llm_manager, expert_key, expert_config.embeddings.__root__
                    ),
                    query_memory_before_ask=expert_config.query_memory_before_ask,
                    enable_history_fuzzy_search=expert_config.enable_history_fuzzy_search,
                    fuzzy_search_distance=expert_config.fuzzy_search_distance,
                    fuzzy_search_limit=expert_config.fuzzy_search_limit,
                    enable_summary_memory=expert_config.enable_summary_memory,
                    create_standalone_question_to_search_context=expert_config.create_standalone_question_to_search_context,
                )

        raise Exception(f"Expert {expert_key} not found")

    @lru_cache
    def get_expert_tools(self, prefix: str = "", session_id: str = "same-session"):
        experts_as_tools = {
            **{k: v for k, v in self.default_tools.items() if v.use_as_tool},
            **{k: v for k, v in self.config.experts.__root__.items() if v.use_as_tool},
        }
        return self.expert_agent_manager.get_experts_as_agent_tools(
            experts_as_tools, prefix, session_id=session_id
        )

    @lru_cache
    def get_chain_chat(self, session_id: str = "same-session"):
        chain_memory = self.embeddings_factory.get_chain_embeddings(
            self.llm_manager,
            embeddings=self.config.chain.embeddings.__root__
            if self.config.chain.embeddings
            else None,
            load_docs=False,
            index_name=self.config.chain.chain_key,
            index_prefix=f"{self.config.chain.chain_key}_",
        )

        embeddings_tools = []
        if self.config.chain.get_from_memory_as_tool:
            embeddings_tools.append(
                chain_memory.get_embeddings_tool_get_memory(
                    tool_key=self.config.chain.chain_key
                )
            )
        if self.config.chain.save_in_memory_as_tool:
            embeddings_tools.append(
                chain_memory.get_embeddings_tool_save_memory(
                    tool_key=self.config.chain.chain_key
                )
            )

        return ChainChatManager(
            self.llm_manager,
            temperature=self.config.chain.temperature,
            max_tokens=self.config.chain.max_tokens,
            tools=embeddings_tools
            + self.get_expert_tools(
                prefix=self.config.chain.chain_key, session_id=session_id
            )
            + self.custom_tools,
            model=self.config.chain.model,
            session_id=session_id,
            embeddings=chain_memory,
            query_memory_before_ask=self.config.chain.query_memory_before_ask,
            enable_history_fuzzy_search=self.config.chain.enable_history_fuzzy_search,
            fuzzy_search_distance=self.config.chain.fuzzy_search_distance,
            fuzzy_search_limit=self.config.chain.fuzzy_search_limit,
            chain_key=self.config.chain.chain_key,
            enable_memory=self.config.chain.enable_memory,
            enable_summary_memory=self.config.chain.enable_summary_memory,
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
