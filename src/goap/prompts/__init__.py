"""Prompt loader for GOAP protocol."""

from pathlib import Path

_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    return (_DIR / name).read_text()
