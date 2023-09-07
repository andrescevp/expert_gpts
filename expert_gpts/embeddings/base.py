from typing import List

from langchain.agents import Tool


class EmbeddingsHandlerBase:
    def __init__(self, *args, **kwargs):
        pass

    def search(self, query: str):
        raise NotImplementedError

    def save(self, remember_this: List[str]):
        raise NotImplementedError

    def get_embeddings_tool_get_memory(self) -> List[Tool]:
        raise NotImplementedError
