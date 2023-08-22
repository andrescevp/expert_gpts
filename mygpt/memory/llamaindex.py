import logging
import os
from typing import List

from langchain.agents import Tool
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
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import RedisVectorStore

from mygpt.llms.base import BaseLLMManager
from mygpt.memory.base import MemoryBase
from shared.llms.openai import GPT_3_5_TURBO

logger = logging.getLogger(__name__)


class LlamaIndexMemory(MemoryBase):
    def __init__(
        self,
        llm_manager: BaseLLMManager,
        document_path: str,
        index_name: str = "pg_essays",
        index_prefix: str = "llama",
    ):
        self.documents = (
            SimpleDirectoryReader(document_path).load_data() if document_path else []
        )

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
        embed_model = OpenAIEmbedding()
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
            embed_model=embed_model,
            node_parser=node_parser,
            prompt_helper=prompt_helper,
        )
        self.index = VectorStoreIndex.from_documents(
            documents=self.documents,
            storage_context=storage_context,
            service_context=service_context,
        )

    def search(self, query: str):
        logger.debug(f"query: {query}")
        return self.index.as_query_engine().query(query)

    def save(self, remember_this: List[str]):
        logger.debug(f"remember_this: {remember_this}")
        documents = StringIterableReader().load_data(remember_this)
        self.documents.extend(documents)
        for document in documents:
            self.index.insert(document)

    def get_agent_tools(self):
        return [
            Tool(
                name="LlamaIndexGetMemories",
                func=lambda q: str(self.search(q)),
                description="useful for remember or gather relevant information about a question, this tool "
                "can be "
                "used to get data provided by the user in previous conversations. "
                "If there is not relevant information another action should follow to find the right answer."
                "Input should be a complete English sentence.",
            ),
            Tool(
                name="LlamaIndexSaveMemories",
                func=lambda q: str(self.save([q])),
                description="useful to save relevant information about a question or information you think is "
                "relevant and will be useful to know and understand to reach your goals, "
                "this tool should be used always to store relevant information and be able to "
                "query the history of the conversation. Input "
                "should be a string.",
            ),
        ]
