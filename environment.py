"""
Backward-compatible environment module.

Grid2D now lives in ``core.grid``; import from here or from ``core`` directly.
"""

from core.grid import Grid2D, default_env

__all__ = ["Grid2D", "default_env"]
