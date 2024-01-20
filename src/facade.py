import tempfile
from os import path
from src.models.methods import ModelType
from src.models.ai_resolver import BaseAIResolver


class Facade:
    def __init__(self, resolver: BaseAIResolver):
        self.resolver = resolver
        self.temp_dir = tempfile.TemporaryDirectory()

    async def _execute_image(self, prompt: str) -> str:
        """ Receives a prompt to execute and returns a path at which the
        result image is stored. """
        image = await self.resolver.query_tti(prompt)
        temp_image_path = path.join(self.temp_dir.name, 'temp_image.png')
        image.save(temp_image_path)
        print(f"Image saved as temporary file: {temp_image_path}")
        return temp_image_path

    async def _execute_audio(self, prompt: str) -> str:
        """ Receives a prompt to execute and returns a path at which the
        result audio is stored. """
        audio = await self.resolver.query_tts(prompt)
        temp_audio_path = path.join(self.temp_dir.name, 'temp_audio.mp3')
        audio.export(temp_audio_path, format="mp3")
        print(f"Audio saved as temporary file: {temp_audio_path}")
        return temp_audio_path

    async def _execute_llm(self, prompt: str) -> str:
        """ Receives a prompt to execute and returns the LLM result. """
        res = await self.resolver.query_llm(messages=[{
            "content": prompt,
            "role": "user"
        }])
        return next(iter(res), 'Failed to receive a response.')

    async def execute(self, prompt: str, method: ModelType):
        if method == ModelType.LLM:
            return await self._execute_llm(prompt=prompt)
        elif method == ModelType.TTI:
            return await self._execute_image(prompt=prompt)
        elif method == ModelType.TTS:
            return await self._execute_audio(prompt=prompt)
        else:
            raise ValueError("Unknown method type")

    def __del__(self):
        # Clean up the temporary directory when the class instance is destroyed
        self.temp_dir.cleanup()
