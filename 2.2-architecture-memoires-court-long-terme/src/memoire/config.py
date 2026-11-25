"""Configuration centralisée mémoire."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MEM_",
    )

    # Stockage
    data_dir: Path = Field(
        default=Path("data/memory"),
        description="Dossier bases SQLite",
    )
    sessions_db: Path = Field(
        default=Path("data/memory/sessions.db"),
        description="DB sessions (court terme)",
    )
    memory_db: Path = Field(
        default=Path("data/memory/memory.db"),
        description="DB utilisateurs (long terme)",
    )

    # Court terme
    hot_window_size: int = Field(
        default=10,
        description="Nb messages gardés en brut (fenêtre glissante)",
    )
    hot_summary_trigger: int = Field(
        default=20,
        description="Nb messages avant résumé récursif",
    )
    hot_max_summary_length: int = Field(
        default=500,
        description="Longueur max résumé (tokens approx)",
    )

    # Long terme
    cold_max_facts_per_user: int = Field(
        default=100,
        description="Max faits stockés par utilisateur",
    )
    cold_max_history_per_user: int = Field(
        default=500,
        description="Max interactions historiques par user",
    )

    # Device / LLM pour résumé
    summarizer_model: str = Field(
        default="qwen2.5:3b-instruct-q4_k_m",
        description="Modèle pour résumé (Ollama)",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="URL Ollama OpenAI-compatible",
    )


settings = Settings()

settings.data_dir.mkdir(parents=True, exist_ok=True)