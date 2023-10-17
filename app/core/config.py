from pathlib import Path
from pydantic_settings import BaseSettings
import logging


class Settings(BaseSettings):
    app_name: str = 'FastAPi Service template'
    api_version: str = 'v1'

    class Config:
        env_file = ".env"


PROJECT_ROOT = Path(__file__).resolve().parents[3]
settings = Settings()

local_settings = {"model_path": "/app/SSL_Anti-spoofing/Best_LA_model_for_DF.pth",
                  "log_level": logging.WARNING,
                  "threshold": 0.0}