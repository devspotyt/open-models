from src.handlers.endpoint import EndpointHandler


class InferenceHandler:
    def __init__(self, base_url: str, token: str):
        self.base_endpoint = EndpointHandler(base_url=base_url)
        self.token = token

    async def query_endpoint(self, model: str, prompt: str) -> bytes:
        """ Receives a model, a prompt & a format. Performs HTTP request
        utilizing the base endpoint and returns the result of the request in
        the requested format."""
        headers = self.base_endpoint.construct_bearer_auth(self.token)
        payload = {"inputs": prompt}
        response = await self.base_endpoint.request("POST", model,
                                                    headers=headers,
                                                    json=payload)
        return response.content
