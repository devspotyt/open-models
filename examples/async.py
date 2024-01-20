"""
Purpose: Serve as an example for parallel usage of ALL AI resolver
functionalities via HF inference.
"""

import asyncio
from src.facade import Facade
from src.models.methods import ModelType
from src.resolvers.hf.hf_resolver import HuggingFaceResolver
from src.settings import Settings


async def main():
    settings = Settings()
    resolver = HuggingFaceResolver(
        base_api="https://api-inference.huggingface.co/models/",
        token=settings.hf_api_token,
        llm_model="mistralai/Mistral-7B-Instruct-v0.1",
        tti_model="dataautogpt3/OpenDalleV1.1",
        tts_model="suno/bark-small")
    facade = Facade(resolver=resolver)
    # Run tasks concurrently
    text, image, audio = await asyncio.gather(
        facade.execute("how much is 2 + 4", method=ModelType.LLM),
        facade.execute("Astronaut riding a horse", method=ModelType.TTI),
        facade.execute("Subscribe to Devspot on YouTube!",
                       method=ModelType.TTS)
    )

    print(f"text: {text}")
    print(f"image stored at: {image}")
    print(f"audio stored at: {audio}")
    print("done!")


if __name__ == '__main__':
    asyncio.run(main())
