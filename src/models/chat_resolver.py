from abc import ABC, abstractmethod
from typing import List, Dict


class AIChatResolverBase(ABC):
    def __init__(self, llm_model: str, base_api: str):
        """ Initialize the resolver with a specific model. """
        self.llm_model = llm_model
        self.base_api = base_api

    @abstractmethod
    async def query_llm(self, messages: List[Dict[str, str]]) -> List[str]:
        """ Abstract method to query the AI model. """
        pass

