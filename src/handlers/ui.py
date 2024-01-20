from json import loads, JSONDecodeError

import gradio as gr

from src.facade import Facade
from src.models.methods import ModelType
from src.resolvers.hf.hf_llm import HuggingFaceLLMResolver
from loguru import logger


def parse_llm_response(llm_response):
    # Default to LLM in case of parsing failure
    model_type = ModelType.LLM
    prompt = ""

    try:
        # Convert the JSON string to a Python dictionary
        response_dict = loads(llm_response)

        # Extract type and content
        response_type = response_dict.get("type", "")
        prompt = response_dict.get("content", "")

        # Map the response type to the ModelType enum
        if response_type == "TTI":
            model_type = ModelType.TTI
        elif response_type == "TTS":
            model_type = ModelType.TTS
        elif response_type == "LLM":
            model_type = ModelType.LLM

    except JSONDecodeError as e:
        print(f"Error parsing LLM JSON response: {e}")
        # In case of exception, use the empty prompt and default to LLM

    return model_type, prompt


class ChatbotInterface:
    def __init__(self, facade: Facade, llm_engine: HuggingFaceLLMResolver):
        self.facade = facade
        self.llm_engine = llm_engine

    async def respond_to_query(self, query, history):
        # Logic to determine the response type
        action, prompt = await self.determine_response_type(query)

        if action == ModelType.TTI:
            image_path = await self.facade.execute(prompt,
                                                   method=ModelType.TTI)
            return (image_path,)
        elif action == ModelType.TTS:
            audio_path = await self.facade.execute(prompt,
                                                   method=ModelType.TTS)
            return (audio_path,)
        else:  # ModelType.LLM
            response = await self.facade.execute(query, method=ModelType.LLM)
            return response

    async def determine_response_type(self, user_query):
        logger.info(f"received query: '{user_query}'")

        # Structured query for the LLM
        llm_query = f"""
            Analyze the following prompt and categorize it accurately. 
            Return a response in JSON format with two keys: 'type' and 'content'. 
            The 'type' key should specify the category of the request: 
                - 'TTI' for image generation requests (e.g., 'Create an image of a sunset over the mountains').
                - 'TTS' for audio generation requests (e.g., 'Convert this text into spoken words').
                - 'LLM' for general chat or information queries (e.g., 'Explain the theory of relativity').
            The 'content' key should contain the main essence of the user prompt.

            Based on the user prompt provided, classify it as one of the above types and extract its main content.
            Here is the prompt: '{user_query}'.

            Ensure the response follows this format: 
            {{'type': '<TTI/TTS/LLM>', 'content': '<MAIN USER PROMPT CONTENT>'}}. 
            The response must strictly align with these guidelines, avoiding any misinterpretation or hallucination.
        """

        # Send the query to the LLM and get the response
        llm_response = await self.llm_engine.execute(prompt=llm_query)

        # Parse the LLM response assuming it follows the specified format
        action, prompt = parse_llm_response(llm_response)

        logger.info(f"action: '{str(action)}', prompt: '{prompt}'")

        return action, prompt

    def launch(self):
        gr.ChatInterface(fn=self.respond_to_query).launch()
