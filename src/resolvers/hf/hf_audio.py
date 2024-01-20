from src.handlers.inference import InferenceHandler
from pydub import AudioSegment
from io import BytesIO
from tenacity import stop_after_attempt, wait_exponential, retry
from src.models.audio_resolver import AIAudioResolverBase


class HuggingFaceTTSResolver(AIAudioResolverBase):
    def __init__(self, base_api: str, token: str, tts_model: str):
        AIAudioResolverBase.__init__(self, tts_model=tts_model,
                                     base_api=base_api)
        self.inference = InferenceHandler(base_url=self.base_api, token=token)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=5, max=5))
    async def query_tts(self, prompt: str) -> AudioSegment:
        audio_bytes = await self.inference.query_endpoint(self.tts_model,
                                                          prompt)
        # Convert the audio bytes directly to an AudioSegment
        return AudioSegment.from_file(BytesIO(audio_bytes), format="flac")
