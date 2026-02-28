"""Configuration package — re-exports for convenient importing.

Usage::

    from src.config import settings, get_settings
"""

from src.config.settings import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]
