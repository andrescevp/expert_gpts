from typing import List

from langchain.agents import Tool


class MemoryBase:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def search(self, query: str):
        raise NotImplementedError

    def save(self, remember_this: List[str]):
        raise NotImplementedError

    def get_agent_tools(self) -> List[Tool]:
        raise NotImplementedError
