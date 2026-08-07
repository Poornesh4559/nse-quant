"""Configuration loading — single source of truth is the repo-root .env file.

All modules import settings from here so secrets and connection details are
never hardcoded. Falls back to sensible defaults when env vars are missing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """App settings pulled from environment variables."""

    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "nse"))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "nse_quant"))
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "127.0.0.1"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))

    fyers_app_id: str = field(default_factory=lambda: os.getenv("FYERS_APP_ID", ""))
    fyers_app_type: str = field(default_factory=lambda: os.getenv("FYERS_APP_TYPE", "100"))
    fyers_app_secret: str = field(default_factory=lambda: os.getenv("FYERS_APP_SECRET", ""))
    fyers_redirect_uri: str = field(default_factory=lambda: os.getenv("FYERS_REDIRECT_URI", ""))
    fyers_id: str = field(default_factory=lambda: os.getenv("FYERS_ID", ""))
    fyers_totp_key: str = field(default_factory=lambda: os.getenv("FYERS_TOTP_KEY", ""))
    fyers_pin: str = field(default_factory=lambda: os.getenv("FYERS_PIN", ""))
    fyers_access_token: str = field(default_factory=lambda: os.getenv("FYERS_ACCESS_TOKEN", ""))

    @property
    def fyers_client_id(self) -> str:
        """FyersModel client_id is '<app_id>-<app_type>', e.g. 'XXXX-100'."""
        return f"{self.fyers_app_id}-{self.fyers_app_type}"

    @property
    def db_url(self) -> str:
        """psycopg2 connection string for the TimescaleDB container."""
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"dbname={self.postgres_db} user={self.postgres_user} "
            f"password={self.postgres_password}"
        )

    @property
    def token_cache_path(self) -> Path:
        """Where the 1-day access token is cached between runs."""
        return DATA_DIR / "fyers_token.json"


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


settings = get_settings()
