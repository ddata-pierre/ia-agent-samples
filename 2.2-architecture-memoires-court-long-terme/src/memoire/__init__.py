"""Package mémoire hybride court/long terme."""

from .config import settings
from .models import (
    Message, RoleMessage, SessionSummary, HotMemoryState,
    ProfilUtilisateur, FaitUtilisateur, TypeFait, InteractionHistorique
)
from .hot_memory import HotMemoryStore, HotMemoryManager
from .cold_memory import ColdMemoryStore, ColdMemoryManager
from .memory_manager import MemoryManager, ContexteInjection

from memoire.sqlite_adapters import register_sqlite_adapters

register_sqlite_adapters()

__all__ = [
    "settings",
    "Message", "RoleMessage", "SessionSummary", "HotMemoryState",
    "ProfilUtilisateur", "FaitUtilisateur", "TypeFait", "InteractionHistorique",
    "HotMemoryStore", "HotMemoryManager",
    "ColdMemoryStore", "ColdMemoryManager",
    "MemoryManager", "ContexteInjection",
]