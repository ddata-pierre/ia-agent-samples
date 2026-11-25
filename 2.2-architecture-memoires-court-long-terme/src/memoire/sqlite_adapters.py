"""Adaptateurs SQLite pour les dates et heures."""

import sqlite3
from datetime import datetime


def adapt_datetime(value: datetime) -> str:
    """Convertit datetime en chaîne ISO 8601."""
    return value.isoformat()


def convert_datetime(value: bytes) -> datetime:
    """Convertit une chaîne ISO 8601 en datetime."""
    return datetime.fromisoformat(value.decode())


def register_sqlite_adapters():
    """Enregistre les adaptateurs et convertisseurs SQLite."""
    sqlite3.register_adapter(datetime, adapt_datetime)
    sqlite3.register_converter("DATETIME", convert_datetime)
    sqlite3.register_converter("TIMESTAMP", convert_datetime)


register_sqlite_adapters()