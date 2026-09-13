"""Centralized config for experiments.

Configuration is loaded from YAML files (see ``configs/defaults/default_train.yaml``) and
exposed here as a module-level singleton.

Usage::

    from qorix.utils import config
    config.cfg.model           # -> "Qwen/Qwen2.5-3B"
    config.cfg.learning_rate   # -> 1e-6
"""
from __future__ import annotations

from qorix.utils.config_schema import QorixConfig

# ---------------------------------------------------------------------------
# Singleton — set by train.py before any other import touches config
# ---------------------------------------------------------------------------

_cfg: QorixConfig | None = None


def get_config() -> QorixConfig:
    """Return the active config, lazily loading defaults if needed."""
    global _cfg
    if _cfg is None:
        from qorix.utils.config_loader import load_config
        _cfg = load_config()
    return _cfg


def install_config(data: dict) -> QorixConfig:
    """Install a config from a pre-serialized dict (used by Ray actors).

    Returns the installed config instance.
    """
    global _cfg
    _cfg = QorixConfig(**data)
    return _cfg


class _CfgProxy:
    """Descriptor that always returns the live singleton."""
    def __repr__(self):
        return repr(get_config())

    def __getattr__(self, name: str):
        return getattr(get_config(), name)


cfg = _CfgProxy()
