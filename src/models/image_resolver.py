from abc import ABC, abstractmethod
from PIL import Image


class AIImageResolverBase(ABC):
    def __init__(self, tti_model: str, base_api: str):
        """ Initialize the resolver with a specific model. """
        self.tti_model = tti_model
        self.base_api = base_api

    @abstractmethod
    async def query_tti(self, prompt: str) -> Image:
        """ Abstract method to query the AI model. """
        pass

