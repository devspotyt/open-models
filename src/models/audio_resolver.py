from abc import ABC, abstractmethod
from pydub import AudioSegment


class AIAudioResolverBase(ABC):
    def __init__(self, tts_model: str, base_api: str):
        """ Initialize the resolver with a specific model. """
        self.tts_model = tts_model
        self.base_api = base_api

    @abstractmethod
    async def query_tts(self, prompt: str) -> AudioSegment:
        """ Abstract method to query the AI model. """
        pass

