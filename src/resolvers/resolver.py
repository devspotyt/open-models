from typing import List, Dict
from PIL.Image import Image
from pydub import AudioSegment
from src.models.ai_resolver import BaseAIResolver
from src.models.audio_resolver import AIAudioResolverBase
from src.models.chat_resolver import AIChatResolverBase
from src.models.image_resolver import AIImageResolverBase


class AIResolver(BaseAIResolver):
    def __init__(self, llm_resolver: AIChatResolverBase,
                 tti_resolver: AIImageResolverBase,
                 tts_resolver: AIAudioResolverBase):
        super().__init__(
            llm_base_api=llm_resolver.base_api,
            llm_model=llm_resolver.llm_model,
            tti_base_api=tti_resolver.base_api,
            tti_model=tti_resolver.tti_model,
            tts_base_api=tts_resolver.base_api,
            tts_model=tts_resolver.tts_model
        )
        self.llm_resolver = llm_resolver
        self.tti_resolver = tti_resolver
        self.tts_resolver = tts_resolver

    async def query_llm(self, messages: List[Dict[str, str]]) -> List[str]:
        return await self.llm_resolver.query_llm(messages=messages)

    async def query_tti(self, prompt: str) -> Image:
        return await self.tti_resolver.query_tti(prompt=prompt)

    async def query_tts(self, prompt: str) -> AudioSegment:
        return await self.tts_resolver.query_tts(prompt=prompt)
