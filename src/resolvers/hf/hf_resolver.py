from src.handlers.inference import InferenceHandler
from src.models.ai_resolver import BaseAIResolver
from src.resolvers.hf.hf_audio import HuggingFaceTTSResolver
from src.resolvers.hf.hf_image import HuggingFaceTTIResolver
from src.resolvers.hf.hf_llm import HuggingFaceLLMResolver


class HuggingFaceResolver(HuggingFaceLLMResolver, HuggingFaceTTIResolver,
                          HuggingFaceTTSResolver, BaseAIResolver):
    def __init__(self, base_api: str, token: str, llm_model: str,
                 tti_model: str, tts_model: str):
        HuggingFaceLLMResolver.__init__(self, llm_model=llm_model,
                                        base_api=base_api, token=token)
        HuggingFaceTTIResolver.__init__(self, tti_model=tti_model,
                                        base_api=base_api, token=token)
        HuggingFaceTTSResolver.__init__(self, tts_model=tts_model,
                                        base_api=base_api, token=token)
        BaseAIResolver.__init__(self,
                                llm_model=llm_model, llm_base_api=base_api,
                                tti_model=tti_model, tti_base_api=base_api,
                                tts_base_api=base_api, tts_model=tts_model)
        self.inference = InferenceHandler(base_url=self.base_api, token=token)
