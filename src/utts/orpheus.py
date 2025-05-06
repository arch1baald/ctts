from enum import Enum

from timeout_function_decorator import timeout

from utts.config import TIMEOUT
from utts.replicate import run
from utts.utils import convert_to_enum


class Voice(str, Enum):
    """Available speakers for Orpheus TTS model."""

    TARA = "tara"
    LEAH = "leah"
    JESS = "jess"
    LEO = "leo"
    DAN = "dan"
    MIA = "mia"
    ZAC = "zac"
    ZOE = "zoe"


class Model(str, Enum):
    """Available models for Orpheus TTS."""

    ORPHEUS_3B = "lucataco/orpheus-3b-0.1-ft:79f2a473e6a9720716a473d9b2f2951437dbf91dc02ccb7079fb3d89b881207f"


class EmotiveTags(str, Enum):
    LAUGH = "`<laugh>`"
    CHUCKLE = "`<chuckle>`"
    SIGH = "`<sigh>`"
    COUGH = "`<cough>`"
    SNIFFLE = "`<sniffle>`"
    GROAN = "`<groan>`"
    YAWN = "`<yawn>`"
    GASP = "`<gasp>`"


@timeout(TIMEOUT)
def generate(
    text: str,
    speaker: Voice | str = Voice.TARA,
    model: Model | str = Model.ORPHEUS_3B,
    language: str = "en",
    top_p: float = 0.95,
    top_k: int = 50,
    temperature: float = 0.6,
    max_new_tokens: int = 1200,
    repetition_penalty: float = 1.1,
    duration_scale: float = 1.0,
    output_format: str = "mp3",
) -> bytes:
    """
    Generates audio from text using the Orpheus TTS model via Replicate.

    Args:
        text: Text to convert to speech
        speaker: Speaker type (male, female, or tara)
        model: TTS model to use
        language: Language code
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        temperature: Temperature for sampling
        max_new_tokens: Maximum number of new tokens to generate
        repetition_penalty: Penalty for repetition
        duration_scale: Speech duration scaling factor
        output_format: Output format (mp3 or wav)

    Returns:
        Audio data as bytes
    """
    speaker_str = convert_to_enum(Voice, speaker).value
    model_str = convert_to_enum(Model, model).value
    input_data = {
        "text": text,
        "speaker": speaker_str,
        "language": language,
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "duration_scale": duration_scale,
        "output_format": output_format,
    }
    return run(model_str, input_data)


@timeout(TIMEOUT)
async def agenerate(
    text: str,
    speaker: Voice | str = Voice.TARA,
    model: Model | str = Model.ORPHEUS_3B,
    language: str = "en",
    top_p: float = 0.95,
    top_k: int = 50,
    temperature: float = 0.6,
    max_new_tokens: int = 1200,
    repetition_penalty: float = 1.1,
    duration_scale: float = 1.0,
    output_format: str = "mp3",
) -> bytes:
    """
    Asynchronous version of generate (currently implemented synchronously).

    See generate() for parameter details.
    """
    # For now, this is just a wrapper around the synchronous function
    # In the future, this could be updated to use async features
    return generate(
        text=text,
        speaker=speaker,
        model=model,
        language=language,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        duration_scale=duration_scale,
        output_format=output_format,
    )
