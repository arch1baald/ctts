"""Compare Text-to-Speech (CTTS) package.

This package provides interfaces to various text-to-speech APIs.
"""

from ctts import config, elevenlabs, kokoro, openai, orpheus, replicate, utils

# Expose key modules directly at the package level
# This allows imports like `from ctts import openai, elevenlabs`
__all__ = [
    "config",
    "elevenlabs",
    "kokoro",
    "openai",
    "orpheus",
    "replicate",
    "utils",
]
