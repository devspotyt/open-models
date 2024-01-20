from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    hf_api_token: str = Field(alias='HF_API_TOKEN')

    class Config:
        # P.S - This is relative to where you're running the code from:
        env_file = ".env"


settings = Settings()
