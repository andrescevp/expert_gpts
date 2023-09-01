import abc
import logging
from functools import lru_cache
from typing import List, Optional

from langchain.agents import Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage

from expert_gpts.chat_history.mysql import MysqlChatMessageHistory
from expert_gpts.database import get_db_session
from expert_gpts.embeddings.base import EmbeddingsHandlerBase
from shared.config import ExpertItem
from shared.llm_manager_base import BaseLLMManager
from shared.llms.openai import GPT_3_5_TURBO
from shared.llms.system_prompts import (
    CHAT_HUMAN_PROMPT_TEMPLATE,
    CHAT_SYSTEM_PROMPT_STANDALONE_QUESTION,
)

DEFAULT_EXPERT_CONFIG = ExpertItem()

logger = logging.getLogger(__name__)


@lru_cache
def get_history(session_id, ai_key):
    with get_db_session() as session:
        return MysqlChatMessageHistory(
            session_id=session_id, session=session, ai_key=ai_key
        )


class ChatSingleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        cls_key = None
        if cls not in cls._instances:
            if "expert_key" in kwargs:
                cls_key = f"expert_key_{kwargs['expert_key']}"
            if "chain_key" in kwargs:
                cls_key = f"chain_key_{kwargs['chain_key']}"

            cls._instances[cls_key] = super(ChatSingleton, cls).__call__(
                *args, **kwargs
            )

        return cls._instances[cls_key]


class SingleChatManager(metaclass=ChatSingleton):
    def __init__(
        self,
        llm_manager: BaseLLMManager,
        expert_key: str,
        expert_config: ExpertItem = DEFAULT_EXPERT_CONFIG,
        session_id: str = "same-session",
        embeddings: Optional[EmbeddingsHandlerBase] = None,
        create_standalone_question_to_search_context: bool = True,
        query_memory_before_ask: bool = True,
        enable_history_fuzzy_search: bool = True,
        fuzzy_search_distance: int = 5,
        fuzzy_search_limit: int = 5,
        enable_summary_memory: bool = False,
    ):
        self.enable_summary_memory = enable_summary_memory
        self.create_standalone_question_to_search_context = (
            create_standalone_question_to_search_context
        )
        self.fuzzy_search_limit = fuzzy_search_limit
        self.fuzzy_search_distance = fuzzy_search_distance
        self.enable_history_fuzzy_search = enable_history_fuzzy_search
        self.query_memory_before_ask = query_memory_before_ask
        self.embeddings = embeddings
        self.expert_config = expert_config
        self.expert_key = expert_key
        self.llm_manager = llm_manager
        self.history = get_history(session_id, expert_key)
        self.session_id = session_id
        self.memory_summary = ConversationSummaryMemory.from_messages(
            llm=self.llm_manager.get_llm(
                temperature=0, max_tokens=None, model=GPT_3_5_TURBO
            ),
            chat_memory=self.history,
        )

    def ask(self, question):
        if self.enable_history_fuzzy_search:
            chat_history = self.history.fuzzy_search(
                question,
                limit=self.fuzzy_search_limit,
                distance=self.fuzzy_search_distance,
            )
        else:
            chat_history = self.history.messages[:5]

        if self.enable_summary_memory:
            chat_history = self.memory_summary.predict_new_summary(
                chat_history, self.memory_summary.buffer
            )

        search_context_question = question
        if self.create_standalone_question_to_search_context:
            search_context_question = get_standalone_question(
                question,
                self.history,
                chat_history,
                self.llm_manager,
                self.expert_config.temperature,
                self.expert_config.max_tokens,
                self.expert_config.model,
            )

        context = ""
        if self.embeddings and self.query_memory_before_ask:
            try:
                context = self.embeddings.search(search_context_question).response
            except Exception as e:
                logger.error("Could not query embeddings: %s", e)

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
                context=context,
                history=chat_history,
            ),
            temperature=self.expert_config.temperature,
            max_tokens=self.expert_config.max_tokens,
            model=self.expert_config.model,
        )

        self.history.add_ai_message(answer)
        return answer

    def get_log(self):
        return self.llm_manager.callbacks_handler.log


class ChainChatManager(metaclass=ChatSingleton):
    def __init__(
        self,
        llm_manager: BaseLLMManager,
        temperature: float = 0,
        max_tokens: int | None = None,
        chain_key: str = "default",
        tools: Optional[List[Tool]] = None,
        model: str | None = None,
        session_id: str = "same-session",
        agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        embeddings: Optional[EmbeddingsHandlerBase] = None,
        query_memory_before_ask: bool = True,
        enable_history_fuzzy_search: bool = True,
        fuzzy_search_distance: int = 5,
        fuzzy_search_limit: int = 5,
        create_standalone_question_to_search_context: bool = True,
        enable_summary_memory: bool = True,
        enable_memory: bool = True,
    ):
        self.enable_memory = enable_memory
        self.enable_summary_memory = enable_summary_memory
        self.chain_key = chain_key
        self.create_standalone_question_to_search_context = (
            create_standalone_question_to_search_context
        )
        self.fuzzy_search_limit = fuzzy_search_limit
        self.fuzzy_search_distance = fuzzy_search_distance
        self.enable_history_fuzzy_search = enable_history_fuzzy_search
        self.query_memory_before_ask = query_memory_before_ask
        self.embeddings = embeddings
        self.agent_type = agent_type
        self.model = model
        self.tools = tools
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm_manager = llm_manager
        self.history = get_history(session_id, self.chain_key)
        self.session_id = session_id
        self.memory_summary = ConversationSummaryMemory.from_messages(
            llm=self.llm_manager.get_llm(
                temperature=0, max_tokens=None, model=GPT_3_5_TURBO
            ),
            chat_memory=self.history,
        )
        self.memory = ConversationBufferMemory(
            llm=self.llm_manager.get_llm(
                temperature=0, max_tokens=None, model=GPT_3_5_TURBO
            )
        )
        for x in self.history.messages:
            self.memory.chat_memory.add_message(x)

    def ask(self, question):
        if self.enable_history_fuzzy_search:
            chat_history = self.history.fuzzy_search(
                question,
                limit=self.fuzzy_search_limit,
                distance=self.fuzzy_search_distance,
            )
        else:
            chat_history = self.history.messages[:5]

        if self.enable_summary_memory:
            chat_history = self.memory_summary.predict_new_summary(
                chat_history, self.memory_summary.buffer
            )

        search_context_question = question
        if self.create_standalone_question_to_search_context:
            search_context_question = get_standalone_question(
                question,
                self.history,
                chat_history,
                self.llm_manager,
                self.temperature,
                self.max_tokens,
                self.model,
            )
        context = ""
        if self.embeddings and self.query_memory_before_ask:
            try:
                context = self.embeddings.search(search_context_question)
            except Exception as e:
                logger.error("Could not query embeddings: %s", e)

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
                        history=None,
                    )
                ]
            ),
            agent_type=self.agent_type,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model=self.model,
            tools=self.tools,
            memory=self.memory if self.enable_memory else None,
            agent_key=self.chain_key,
        )

        self.history.add_ai_message(answer)
        return answer

    def get_log(self):
        return self.llm_manager.callbacks_handler.log


def get_standalone_question(
    question, history, chat_history, llm_manager, temperature, max_tokens, model
):
    template = ChatPromptTemplate.from_messages(
        [
            CHAT_SYSTEM_PROMPT_STANDALONE_QUESTION,
            CHAT_HUMAN_PROMPT_TEMPLATE,
        ]
    )
    return llm_manager.create_chat_completion(
        template.format_messages(
            question=question,
            context="",
            history=chat_history,
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )
