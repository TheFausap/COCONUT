"""Tiny COCONUT language model package."""

from coconut_lm.config import CoconutConfig
from coconut_lm.tokenizer import CharTokenizer

__all__ = [
    "CharTokenizer",
    "CoconutConfig",
    "CoconutOutput",
    "TinyCoconutLM",
]


def __getattr__(name: str):
    if name in {"CoconutOutput", "TinyCoconutLM"}:
        from coconut_lm.model import CoconutOutput, TinyCoconutLM

        return {"CoconutOutput": CoconutOutput, "TinyCoconutLM": TinyCoconutLM}[name]
    raise AttributeError(name)
