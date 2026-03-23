"""Prompt loader for GOAP protocol."""

from pathlib import Path

_DIR = Path(__file__).parent


def load_prompt(name: str, setting: str | None = None) -> str:
    """Load a prompt template, with optional per-setting override.

    If *setting* is provided, checks ``prompts/{setting}/{name}`` first.
    Falls back to the shared ``prompts/{name}``.
    """
    if setting:
        p = _DIR / setting / name
        if p.exists():
            return p.read_text()
    return (_DIR / name).read_text()
