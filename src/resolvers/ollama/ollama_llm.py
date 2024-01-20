from litellm import completion
from typing import List, Dict
from tenacity import stop_after_attempt, wait_exponential, retry
from src.models.chat_resolver import AIChatResolverBase


class OllamaLLMResolver(AIChatResolverBase):

    def __init__(self, base_api: str, llm_model: str):
        AIChatResolverBase.__init__(self, llm_model=llm_model,
                                    base_api=base_api)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=5, max=5))
    async def query_llm(self, messages: List[Dict[str, str]],
                        **kwargs) -> List[str]:
        response = completion(
            model=self.llm_model,
            messages=messages,
            api_base=self.base_api,
            **kwargs
        )
        return [x.message.content for x in response.choices]

    async def execute(self, prompt: str) -> str:
        """ Receives a prompt to execute and returns the LLM result. """
        res = await self.query_llm(messages=[{
            "content": prompt,
            "role": "user"
        }])
        return next(iter(res), 'Failed to receive a response.')
