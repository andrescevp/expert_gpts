import logging
from collections import defaultdict

from expert_gpts.embeddings.factory import EmbeddingsHandlerFactory
from expert_gpts.llms.chat_managers import ChainChatManager, SingleChatManager
from expert_gpts.llms.expert_agents import ExpertAgentManager
from expert_gpts.llms.providers.openai import OpenAIApiManager
from expert_gpts.toolkit.modules import ModuleLoader
from shared.config import Config
from shared.llms.system_prompts import BASE_EXPERTS

logger = logging.getLogger(__name__)


class LLMConfigBuilder:
    def __init__(self, config: Config):
        self.module_loader = ModuleLoader()
        self.config = config
        self.llm_manager = OpenAIApiManager()
        self.expert_agent_manager = ExpertAgentManager()
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

    def get_expert_chats(self, session_id: str = "same-session"):
        chats = defaultdict()
        for expert_key, expert_config in self.experts_map.items():
            chats[expert_key] = SingleChatManager(
                self.llm_manager,
                expert_key,
                expert_config=expert_config,
                session_id=session_id,
                embeddings=EmbeddingsHandlerFactory().get_expert_embeddings(
                    self.llm_manager, expert_key, expert_config.embeddings.__root__
                ),
            )

        return chats

    def get_expert_chat(self, expert_key: str, session_id: str = "same-session"):
        for dict_expert_key, expert_config in self.experts_map.items():
            if dict_expert_key == expert_key:
                return SingleChatManager(
                    self.llm_manager,
                    expert_key,
                    expert_config=expert_config,
                    session_id=session_id,
                    embeddings=EmbeddingsHandlerFactory().get_expert_embeddings(
                        self.llm_manager, expert_key, expert_config.embeddings.__root__
                    ),
                    query_memory_before_ask=expert_config.query_memory_before_ask,
                    enable_history_fuzzy_search=expert_config.enable_history_fuzzy_search,
                    fuzzy_search_distance=expert_config.fuzzy_search_distance,
                    fuzzy_search_limit=expert_config.fuzzy_search_limit,
                )

        raise Exception(f"Expert {expert_key} not found")

    def get_expert_tools(self, prefix: str = ""):
        experts_as_tools = {
            **{k: v for k, v in self.default_tools.items() if v.use_as_tool},
            **{k: v for k, v in self.config.experts.__root__.items() if v.use_as_tool},
        }
        return self.expert_agent_manager.get_experts_as_agent_tools(
            experts_as_tools, prefix
        )

    def get_chain_chat(self, session_id: str = "same-session"):
        chain_memory = EmbeddingsHandlerFactory().get_chain_embeddings(
            self.llm_manager,
            embeddings=self.config.chain.embeddings.__root__
            if self.config.chain.embeddings
            else None,
            load_docs=False,
            index_name=self.config.chain.chain_key,
            index_prefix=f"{self.config.chain.chain_key}_",
        )

        memory_tools = []
        if self.config.chain.get_from_memory_as_tool:
            memory_tools.append(
                chain_memory.get_memory_tool_get_memory(
                    tool_key=self.config.chain.chain_key
                )
            )
        if self.config.chain.save_in_memory_as_tool:
            memory_tools.append(
                chain_memory.get_memory_tool_save_memory(
                    tool_key=self.config.chain.chain_key
                )
            )

        return ChainChatManager(
            self.llm_manager,
            temperature=self.config.chain.temperature,
            max_tokens=self.config.chain.max_tokens,
            tools=memory_tools
            + self.get_expert_tools(prefix=self.config.chain.chain_key)
            + self.custom_tools,
            model=self.config.chain.model,
            session_id=session_id,
            memory=chain_memory,
            query_memory_before_ask=self.config.chain.query_memory_before_ask,
            enable_history_fuzzy_search=self.config.chain.enable_history_fuzzy_search,
            fuzzy_search_distance=self.config.chain.fuzzy_search_distance,
            fuzzy_search_limit=self.config.chain.fuzzy_search_limit,
        )

    def load_docs(self):
        # main chain embeddings load
        EmbeddingsHandlerFactory().get_chain_embeddings(
            self.llm_manager,
            self.config.chain.embeddings.__root__
            if self.config.chain.embeddings
            else None,
            load_docs=True,
            index_name=self.config.chain.chain_key,
            index_prefix=f"{self.config.chain.chain_key}_",
        )

        for dict_expert_key, expert_config in self.config.experts.__root__.items():
            EmbeddingsHandlerFactory().get_expert_embeddings(
                self.llm_manager,
                dict_expert_key,
                expert_config.embeddings.__root__,
                load_docs=True,
            )
