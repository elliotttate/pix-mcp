"""Configuration: PIX install discovery, environment, default paths."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


PIX_DEFAULT_ROOT = Path(r"C:\Program Files\Microsoft PIX")
PIX_ENV_OVERRIDE = "PIX_MCP_PIXTOOL"
PIX_INSTALL_ENV_OVERRIDE = "PIX_MCP_INSTALL_ROOT"
CACHE_DIR_ENV = "PIX_MCP_CACHE_DIR"


def _version_key(name: str) -> tuple[int, ...]:
    """Sort PIX version directories like '2603.25' numerically."""
    parts = re.findall(r"\d+", name)
    return tuple(int(p) for p in parts) if parts else (0,)


def find_pixtool() -> Path | None:
    """Locate pixtool.exe.

    Resolution order:
      1. PIX_MCP_PIXTOOL env var (absolute path to pixtool.exe).
      2. PIX_MCP_INSTALL_ROOT env var (folder containing pixtool.exe).
      3. Newest version folder under C:\\Program Files\\Microsoft PIX.
      4. pixtool.exe on PATH.
    """
    env_path = os.environ.get(PIX_ENV_OVERRIDE)
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return candidate

    env_root = os.environ.get(PIX_INSTALL_ENV_OVERRIDE)
    if env_root:
        candidate = Path(env_root) / "pixtool.exe"
        if candidate.is_file():
            return candidate

    if PIX_DEFAULT_ROOT.is_dir():
        versions = sorted(
            (p for p in PIX_DEFAULT_ROOT.iterdir() if p.is_dir()),
            key=lambda p: _version_key(p.name),
            reverse=True,
        )
        for v in versions:
            candidate = v / "pixtool.exe"
            if candidate.is_file():
                return candidate

    from shutil import which

    found = which("pixtool")
    if found:
        return Path(found)
    return None


def default_cache_dir() -> Path:
    """Where to store per-capture indexes and parser caches."""
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override)
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    return base / "pix-mcp" / "cache"


def default_pix_captures_dir() -> Path:
    """PIX writes captures here when triggered via F11 / in-game."""
    return Path.home() / "Documents" / "PIX" / "Captures"


@dataclass
class Settings:
    pixtool_path: Path | None = field(default_factory=find_pixtool)
    cache_dir: Path = field(default_factory=default_cache_dir)
    captures_dir: Path = field(default_factory=default_pix_captures_dir)
    default_remote: str | None = None  # e.g. "localhost" or a remote machine name
    pixtool_timeout_sec: int = 600  # generous default for big captures

    def require_pixtool(self) -> Path:
        if self.pixtool_path is None or not self.pixtool_path.is_file():
            raise FileNotFoundError(
                "pixtool.exe not found. Set PIX_MCP_PIXTOOL or install Microsoft PIX "
                f"under {PIX_DEFAULT_ROOT}."
            )
        return self.pixtool_path

    def ensure_cache_dir(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir


# Module-level singleton, refreshable via reload_settings().
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings
