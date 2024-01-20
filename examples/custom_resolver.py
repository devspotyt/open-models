"""
Purpose: Serve as an example for how one can create a custom AI resolver by
combining LLM, TTI & TTS resolvers of various types.
"""

import asyncio
from src.resolvers.hf.hf_audio import HuggingFaceTTSResolver
from src.resolvers.hf.hf_image import HuggingFaceTTIResolver
from src.resolvers.ollama.ollama_llm import OllamaLLMResolver
from src.resolvers.resolver import AIResolver
from src.settings import Settings


def prepare_ai_resolver(base_hf_api: str, settings: Settings) -> AIResolver:
    """ Prepares an AI resolver based on the provided params. """
    llm_resolver = OllamaLLMResolver(base_api='http://localhost:11434',
                                     llm_model='ollama/mistral')
    tti_resolver = HuggingFaceTTIResolver(
        base_api=base_hf_api, token=settings.hf_api_token,
        tti_model='dataautogpt3/OpenDalleV1.1')
    tts_resolver = HuggingFaceTTSResolver(
        base_api=base_hf_api, token=settings.hf_api_token,
        tts_model='suno/bark-small')
    return AIResolver(
        llm_resolver=llm_resolver,
        tti_resolver=tti_resolver,
        tts_resolver=tts_resolver
    )


async def main():
    base_hf_url = "https://api-inference.huggingface.co/models/"
    settings = Settings()
    ai_resolver = prepare_ai_resolver(base_hf_url, settings)

    res = await ai_resolver.llm_resolver.query_llm(messages=[{
        "content": "how much is 4 + 4?",
        "role": "user"
    }])

    print(res)


if __name__ == '__main__':
    asyncio.run(main())
