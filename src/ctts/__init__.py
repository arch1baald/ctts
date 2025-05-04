"""Compare Text-to-Speech (CTTS) package.

This package provides interfaces to various text-to-speech APIs.
"""

from ctts import cartesia, config, elevenlabs, hume, kokoro, openai, orpheus, replicate, utils, zyphra

# Expose key modules directly at the package level
# This allows imports like `from ctts import openai, elevenlabs`
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
