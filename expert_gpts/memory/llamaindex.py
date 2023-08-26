import logging
import os
from typing import List

from langchain.agents import Tool
from langchain.embeddings import OpenAIEmbeddings
from llama_index import (
    LLMPredictor,
    OpenAIEmbedding,
    PromptHelper,
    ServiceContext,
    SimpleDirectoryReader,
    StringIterableReader,
    VectorStoreIndex,
)
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import RedisVectorStore

from expert_gpts.llms.base import BaseLLMManager
from expert_gpts.memory.base import MemoryBase
from shared.config import EMBEDDINGS_TYPE
from shared.llms.openai import GPT_3_5_TURBO

logger = logging.getLogger(__name__)

EMBEDS_MODEL = OpenAIEmbedding()
embeddings = OpenAIEmbeddings()


# https://zeeshankhawar.medium.com/connecting-chatgpt-with-your-own-data-using-llama-index-and-langchain-74ba79fb7429
# https://betterprogramming.pub/llamaindex-how-to-use-index-correctly-6f928b8944c6
# https://medium.com/badal-io/exploring-langchain-and-llamaindex-to-achieve-standardization-and-interoperability-in-large-2b5f3fabc360
# https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/
# https://dev.to/iamadhee/combine-langchain-llama-index-1068
# https://cobusgreyling.medium.com/llamaindex-chat-engine-858311dfb8cb


class LlamaIndexMemory(MemoryBase):
    get_memories_tool_prompt = """
useful for remember or gather relevant information about a question,
this tool should be used in any chain to try to get information before actually
call other actions, this tool can be
used to get data provided by the user in previous conversations.
If there is not relevant information another action should follow to find the
right answer. Input should be a string.
    """

    save_memories_tool_prompt = """
useful to save relevant information about a question or information you believe is
relevant and will be useful to know and understand to reach your goals,
this tool should be used always to store relevant information and be able to
query the history of the conversation. Input should be a string.
    """

    def __init__(
        self,
        llm_manager: BaseLLMManager,
        embeddings: EMBEDDINGS_TYPE,
        index_name: str = "pg_essays",
        index_prefix: str = "llama",
        load_docs: bool = False,
    ):
        self.document_strings = []
        self.documents = []
        for key, value in embeddings.items():
            if value.folder_path:
                self.documents.extend(
                    SimpleDirectoryReader(value.folder_path).load_data()
                )

        for key, value in embeddings.items():
            if value.content:
                self.document_strings.append(value.content)
        self.documents.extend(StringIterableReader().load_data(self.document_strings))

        vector_store = RedisVectorStore(
            index_name=index_name,
            index_prefix=index_prefix,
            redis_url=os.getenv("REDIS_URL"),
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        llm_predictor = LLMPredictor(
            llm=llm_manager.get_llm(
                max_tokens=256, temperature=0.9, model=GPT_3_5_TURBO
            )
        )
        node_parser = SimpleNodeParser(
            text_splitter=TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
        )
        prompt_helper = PromptHelper(
            context_window=4096,
            num_output=256,
            chunk_overlap_ratio=0.1,
            chunk_size_limit=None,
        )
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            embed_model=EMBEDS_MODEL,
            node_parser=node_parser,
            prompt_helper=prompt_helper,
        )
        if not load_docs:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                service_context=service_context,
            )
        else:
            self.index = VectorStoreIndex.from_documents(
                documents=self.documents,
                storage_context=storage_context,
                service_context=service_context,
            )

    def search(self, query: str) -> RESPONSE_TYPE:
        logger.debug(f"query: {query}")
        return self.index.as_query_engine().query(query)

    def save(self, remember_this: List[str]):
        logger.debug(f"remember_this: {remember_this}")
        documents = StringIterableReader().load_data(remember_this)
        self.documents.extend(documents)
        for document in documents:
            self.index.insert(document)

    def get_agent_tools(
        self, with_save: bool = False, tool_key: str = "default"
    ) -> List[Tool]:
        get_memories = Tool(
            name=f"{tool_key}_get_memories",
            func=lambda q: str(self.search(q)),
            description=self.get_memories_tool_prompt,
        )

        if not with_save:
            return [get_memories]

        return [
            get_memories,
            Tool(
                name=f"{tool_key}_save_memory",
                func=lambda q: str(self.save([q])),
                description=self.save_memories_tool_prompt,
            ),
        ]
