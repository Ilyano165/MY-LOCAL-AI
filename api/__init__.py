"""NOVA API: lokale HTTP-Schicht zwischen Benutzeroberfläche und Core.

Die UI spricht ausschließlich mit dieser API – nie direkt mit Modell-Runtimes.
"""

from api.app import create_app
from api.config import ApiConfig

__all__ = ["ApiConfig", "create_app"]
