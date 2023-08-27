from expert_gpts.memory.llamaindex import LlamaIndexMemory
from shared.config import EMBEDDINGS_TYPE
from shared.llm_manager_base import BaseLLMManager
from shared.patterns import Singleton


class MemoryFactory(metaclass=Singleton):
    def get_chain_embeddings_memory(
        self,
        llm_manager: BaseLLMManager,
        embeddings: EMBEDDINGS_TYPE = None,
        load_docs: bool = False,
        index_name: str = "main_chain_memory",
        index_prefix: str = "main_chain_memory_",
    ) -> LlamaIndexMemory:
        return LlamaIndexMemory(
            llm_manager,
            embeddings=embeddings,
            index_name=index_name,
            index_prefix=index_prefix,
            load_docs=load_docs,
        )

    def get_expert_embeddings_memory(
        self,
        llm_manager: BaseLLMManager,
        expert_key: str,
        embeddings: EMBEDDINGS_TYPE = None,
        load_docs: bool = False,
    ) -> LlamaIndexMemory:
        return LlamaIndexMemory(
            llm_manager,
            embeddings=embeddings,
            index_name=f"{expert_key}_memory",
            index_prefix=f"{expert_key}_memory_",
            load_docs=load_docs,
        )
