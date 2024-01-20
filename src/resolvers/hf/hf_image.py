from PIL.Image import Image
from src.handlers.inference import InferenceHandler
from PIL import Image
from io import BytesIO
from tenacity import stop_after_attempt, wait_exponential, retry
from src.models.image_resolver import AIImageResolverBase


class HuggingFaceTTIResolver(AIImageResolverBase):
    def __init__(self, base_api: str, token: str, tti_model: str):
        AIImageResolverBase.__init__(self, tti_model=tti_model,
                                     base_api=base_api)
        self.inference = InferenceHandler(base_url=self.base_api, token=token)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=5, max=5))
    async def query_tti(self, prompt: str) -> Image:
        image_bytes = await self.inference.query_endpoint(self.tti_model,
                                                          prompt)
        # Convert the image bytes directly to a PIL Image
        return Image.open(BytesIO(image_bytes))
