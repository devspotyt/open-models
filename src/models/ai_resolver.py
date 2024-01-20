from abc import abstractmethod
from typing import List, Dict
from PIL.Image import Image
from pydub import AudioSegment
from src.models.audio_resolver import AIAudioResolverBase
from src.models.chat_resolver import AIChatResolverBase
from src.models.image_resolver import AIImageResolverBase


class BaseAIResolver(AIChatResolverBase, AIImageResolverBase,
                     AIAudioResolverBase):
    def __init__(self,
                 llm_base_api: str, llm_model: str,
                 tti_base_api: str, tti_model: str,
                 tts_base_api: str, tts_model: str):
        """ Initialize the resolver with a specific "model" combination. """
        AIChatResolverBase.__init__(self, llm_model, llm_base_api)
        AIImageResolverBase.__init__(self, tti_model, tti_base_api)
        AIAudioResolverBase.__init__(self, tts_model, tts_base_api)

    @abstractmethod
    async def query_llm(self, messages: List[Dict[str, str]]) -> List[str]:
        """ Abstract method to query the AI model. """
        pass

    @abstractmethod
    async def query_tti(self, prompt: str) -> Image:
        """ Abstract method to query the AI model. """
        pass

    @abstractmethod
    async def query_tts(self, prompt: str) -> AudioSegment:
        """ Abstract method to query the AI model. """
        pass
