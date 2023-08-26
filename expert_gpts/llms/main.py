import logging
from collections import defaultdict

from expert_gpts.llms.chat_managers import ChainChatManager, SingleChatManager
from expert_gpts.llms.expert_agents import ExpertAgentManager
from expert_gpts.llms.openai import OpenAIApiManager
from expert_gpts.memory.factory import MemoryFactory
from expert_gpts.toolkit.modules import ModuleLoader
from shared.config import Config, DefaultAgentTools, ExpertItem, Prompts
from shared.llms.openai import GPT_3_5_TURBO

GENERAL_EXPERT = ExpertItem(
    model=GPT_3_5_TURBO,
    temperature=1,
    prompts=Prompts(
        system="I'm a helpful AI assistant for other AIs I can help in everything else other AIs tools can "
        "not do "
        "with "
        "tools available."
    ),
)
NO_CODE_FUNCTIONS = ExpertItem(
    model=GPT_3_5_TURBO,
    temperature=1,
    prompts=Prompts(
        system="I am a helpful AI assistant for other AIs. I am able to translate the input in to a python "
        "function magically and return the expected result."
    ),
)
THINKER_EXPERT = ExpertItem(
    model=GPT_3_5_TURBO,
    temperature=1,
    prompts=Prompts(
        system="I am a helpful AI assistant for other AIs. "
        "I am the first tool in any chain to decide what to do with the question of the user, "
        "even before any memory tool."
        "I am able to decide what tool to use to answer the question so the chain can be simple and avoid loops. "
        "I am able to decide if the answer is good enough to be saved in the memory. "
        "I am able to decide if the answer is good enough to be show the answer to the user. "
    ),
)
SUMMARIZER_EXPERT = ExpertItem(
    model=GPT_3_5_TURBO,
    temperature=1,
    prompts=Prompts(
        system="I am a helpful AI assistant for other AIs. I am the last tool in any chain to summarize the "
        "answer to max 1000 tokens. I am also the tool used before save any memory. I am able to "
        "validate the length of the answer using internally "
        "the tokenization of the model."
    ),
)

BASE_EXPERTS = {
    DefaultAgentTools.GENERALIST_EXPERT.value: GENERAL_EXPERT,
    DefaultAgentTools.NO_CODE_PYTHON_FUNCTIONS_EXPERT.value: NO_CODE_FUNCTIONS,
    DefaultAgentTools.THINKER_EXPERT.value: THINKER_EXPERT,
    DefaultAgentTools.SUMMARIZER_EXPERT.value: SUMMARIZER_EXPERT,
}


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
                memory=MemoryFactory().get_expert_embeddings_memory(
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
                    memory=MemoryFactory().get_expert_embeddings_memory(
                        self.llm_manager, expert_key, expert_config.embeddings.__root__
                    ),
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
        memory_tools = []
        if self.config.enable_memory_tools:
            memory_tools = (
                MemoryFactory()
                .get_chain_embeddings_memory(
                    self.llm_manager,
                    self.config.chain.embeddings.__root__
                    if self.config.chain.embeddings
                    else None,
                )
                .get_agent_tools()
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
        )

    def load_docs(self):
        # main chain memory load
        MemoryFactory().get_chain_embeddings_memory(
            self.llm_manager,
            self.config.chain.embeddings.__root__
            if self.config.chain.embeddings
            else None,
            load_docs=True,
        )

        for dict_expert_key, expert_config in self.config.experts.__root__.items():
            MemoryFactory().get_expert_embeddings_memory(
                self.llm_manager,
                dict_expert_key,
                expert_config.embeddings.__root__,
                load_docs=True,
            )
