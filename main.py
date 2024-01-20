import asyncio
from src.handlers.ui import ChatbotInterface
from src.resolvers.hf.hf_llm import HuggingFaceLLMResolver
from src.resolvers.hf.hf_resolver import HuggingFaceResolver
from src.facade import Facade
from src.settings import Settings


async def main():
    settings = Settings()
    base_hf_url = "https://api-inference.huggingface.co/models/"
    # "AI" resolver that utilizes Mistral, Open Dalle & Bark:
    resolver = HuggingFaceResolver(
        base_api=base_hf_url,
        token=settings.hf_api_token,
        llm_model="mistralai/Mistral-7B-Instruct-v0.1",
        tti_model="dataautogpt3/OpenDalleV1.1",
        tts_model="suno/bark-small")
    # LLM Engine that utilizes zephyr from HF inference:
    llm_engine = HuggingFaceLLMResolver(
        base_api=base_hf_url,
        token=settings.hf_api_token,
        llm_model="HuggingFaceH4/zephyr-7b-beta")
    facade = Facade(resolver=resolver)
    chatbot_ui = ChatbotInterface(facade, llm_engine=llm_engine)
    chatbot_ui.launch()


if __name__ == "__main__":
    asyncio.run(main())
