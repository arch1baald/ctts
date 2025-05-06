"""Universal Interface to Test and Compare Text-to-Speech models (UTTS)."""

from utts import cartesia, config, elevenlabs, hume, kokoro, openai, orpheus, replicate, utils, zyphra

# Expose key modules directly at the package level
# This allows imports like `from utts import openai, elevenlabs`
__all__ = [
    "cartesia",
    "config",
    "elevenlabs",
    "hume",
    "kokoro",
    "openai",
    "orpheus",
    "replicate",
    "utils",
    "zyphra",
]
