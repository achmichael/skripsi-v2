from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    model_dir: str = "."
    model_version: str = "v1.0-20261031"
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://skripsi.mchl.my.id",
        "https://www.skripsi.mchl.my.id",
        "https://*.vercel.app"
    ]

    class Config:
        env_file = ".env"

settings = Settings()
