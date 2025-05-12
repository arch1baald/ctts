from enum import Enum
from functools import lru_cache
from io import BytesIO

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from timeout_function_decorator import timeout

from utts.config import MAXHITS, TIMEOUT, get_settings
from utts.utils import convert_to_enum


class Voice(str, Enum):
    """Available voices for ElevenLabs TTS API.

    These values are voice IDs, not voice names.
    Check https://api.elevenlabs.io/v1/voices for the current list of available voices.
    """

    # Basic voices
    ADAM = "pNInz6obpgDQGcFmaJgB"
    ANTONI = "ErXwobaYiN019PkySvjV"
    ARNOLD = "VR6AewLTigWG4xSOukaG"
    BELLA = "EXAVITQu4vr4xnSDxMaL"
    DOMI = "AZnzlk1XvdvUeBnXmlld"
    ELLI = "MF3mGyEYCl7XYWbV9V6O"
    JOSH = "TxGEqnHWrfWFTfGW9XjX"
    RACHEL = "21m00Tcm4TlvDq8ikWAM"
    SAM = "yoZ06aMxZJJ28mfd3POQ"

    # Additional voices
    CHARLIE = "IKne3meq5aSn9XLyUdCD"
    EMILY = "LcfcDJNUP1GQjkzn1xUU"
    GIGI = "jBpfuIE2acCO8z3wKNLl"
    GRACE = "oWAxZDx7w5VEj9dCyTzz"
    DREW = "29vD33N1CtxCmqQRPOHJ"
    THOMAS = "GBv7mTt0atIp3Br8iCZE"


class Model(str, Enum):
    """Available models for ElevenLabs TTS API."""

    ELEVEN_MULTILINGUAL_V2 = "eleven_multilingual_v2"
    ELEVEN_MONOLINGUAL_V1 = "eleven_monolingual_v1"
    ELEVEN_TURBO_V2 = "eleven_turbo_v2"


@lru_cache(MAXHITS)
def get_client() -> ElevenLabs:
    """Returns an ElevenLabs client."""
    settings = get_settings().elevenlabs
    assert settings is not None, "ElevenLabs settings are not configured"
    return ElevenLabs(api_key=settings.api_key)


@timeout(TIMEOUT)
def generate(
    text: str,
    voice: Voice | str = Voice.RACHEL,
    model: Model | str = Model.ELEVEN_MULTILINGUAL_V2,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0.0,
    use_speaker_boost: bool = True,
) -> bytes:
    """
    Generates audio from text using ElevenLabs TTS API (synchronous version).

    Args:
        text: Text to convert to speech
        voice: Voice to use
        model: TTS model to use
        stability: Voice stability (0-1)
        similarity_boost: Similarity boost factor (0-1)
        style: Speaking style factor (0-1)
        use_speaker_boost: Enable speaker boost

    Returns:
        Audio data as bytes
    """
    voice_str = convert_to_enum(Voice, voice).value
    model_str = convert_to_enum(Model, model).value

    client = get_client()

    voice_settings = VoiceSettings(
        stability=stability, similarity_boost=similarity_boost, style=style, use_speaker_boost=use_speaker_boost
    )

    audio_stream = client.text_to_speech.convert(
        text=text, voice_id=voice_str, model_id=model_str, voice_settings=voice_settings
    )

    audio_data = BytesIO()
    for chunk in audio_stream:
        if chunk:
            audio_data.write(chunk)

    audio_data.seek(0)
    return audio_data.read()


@timeout(TIMEOUT)
async def agenerate(
    text: str,
    voice: Voice | str = Voice.RACHEL,
    model: Model | str = Model.ELEVEN_MULTILINGUAL_V2,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0.0,
    use_speaker_boost: bool = True,
) -> bytes:
    """
    Generates audio from text using ElevenLabs TTS API asynchronously.

    Args:
        text: Text to convert to speech
        voice: Voice to use
        model: TTS model to use
        stability: Voice stability (0-1)
        similarity_boost: Similarity boost factor (0-1)
        style: Speaking style factor (0-1)
        use_speaker_boost: Enable speaker boost

    Returns:
        Audio data as bytes
    """
    # Since the official ElevenLabs client does not have an async API for text_to_speech
    # we use the synchronous function. This can be updated in the future when the API adds support.

    return generate(
        text=text,
        voice=voice,
        model=model,
        stability=stability,
        similarity_boost=similarity_boost,
        style=style,
        use_speaker_boost=use_speaker_boost,
    )
