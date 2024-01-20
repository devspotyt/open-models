from urllib.parse import urljoin
from litellm import completion
from typing import List, Dict
from tenacity import stop_after_attempt, wait_exponential, retry
from src.handlers.inference import InferenceHandler
from src.models.chat_resolver import AIChatResolverBase


class HuggingFaceLLMResolver(AIChatResolverBase):
    RESOLVER_PREFIX = "huggingface"

    def __init__(self, base_api: str, token: str, llm_model: str):
        AIChatResolverBase.__init__(self, llm_model=llm_model,
                                    base_api=base_api)
        self.inference = InferenceHandler(base_url=self.base_api, token=token)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=5, max=5))
    async def query_llm(self, messages: List[Dict[str, str]]) -> List[str]:
        response = completion(
            model=f"{self.RESOLVER_PREFIX}/{self.llm_model}",
            messages=messages,
            api_base=urljoin(self.base_api, self.llm_model),
            max_new_tokens=512
        )
        return [x.message.content for x in response.choices]

    async def execute(self, prompt: str) -> str:
        """ Receives a prompt to execute and returns the LLM result. """
        res = await self.query_llm(messages=[{
            "content": prompt,
            "role": "user"
        }])
        return next(iter(res), 'Failed to receive a response.')
