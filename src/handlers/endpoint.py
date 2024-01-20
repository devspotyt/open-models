from typing import Dict
from urllib.parse import urljoin

from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from requests import Session, Response, RequestException


class EndpointHandler:
    def __init__(self, base_url: str):
        self.session = Session()
        self.base_url = base_url

    @staticmethod
    def construct_bearer_auth(token: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=5, max=5))
    async def request(self, method: str, endpoint: str, **kwargs) -> Response:
        url = urljoin(self.base_url, endpoint)
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except RequestException as e:
            logger.error(f"Request to '{endpoint}' failed: {e}")
            raise
