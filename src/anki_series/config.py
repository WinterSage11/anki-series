"""Configuración leída de variables de entorno / archivo .env."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    pixabay_api_key: SecretStr | None = None
    data_dir: Path = Path("data")
    # Campo del mazo conocido con la palabra en inglés: índice ("1") o nombre ("Palabra").
    known_deck_field: str = "0"
