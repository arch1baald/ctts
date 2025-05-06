from enum import Enum
from typing import Optional

from timeout_function_decorator import timeout

from utts.config import TIMEOUT
from utts.replicate import run
from utts.utils import convert_to_enum


class Voice(str, Enum):
    """Available voices for Kokoro TTS model."""

    # American English female voices
    AF_ALLOY = "af_alloy"
    AF_AOEDE = "af_aoede"
    AF_BELLA = "af_bella"
    AF_JESSICA = "af_jessica"
    AF_KORE = "af_kore"
    AF_NICOLE = "af_nicole"
    AF_NOVA = "af_nova"
    AF_RIVER = "af_river"
    AF_SARAH = "af_sarah"
    AF_SKY = "af_sky"

    # American English male voices
    AM_ADAM = "am_adam"
    AM_ECHO = "am_echo"
    AM_ERIC = "am_eric"
    AM_FENRIR = "am_fenrir"
    AM_LIAM = "am_liam"
    AM_MICHAEL = "am_michael"
    AM_ONYX = "am_onyx"
    AM_PUCK = "am_puck"

    # British English female voices
    BF_EMMA = "bf_emma"
    BF_ISABELLA = "bf_isabella"
    BF_ALICE = "bf_alice"
    BF_LILY = "bf_lily"

    # British English male voices
    BM_FABLE = "bm_fable"
    BM_DANIEL = "bm_daniel"
    BM_GEORGE = "bm_george"
    BM_LEWIS = "bm_lewis"

    # French female voice
    FF_SIWIS = "ff_siwis"

    # Hindi voices
    HF_ALPHA = "hf_alpha"
    HF_BETA = "hf_beta"
    HM_OMEGA = "hm_omega"
    HM_PSI = "hm_psi"

    # Italian voices
    IF_SARA = "if_sara"
    IM_NICOLA = "im_nicola"

    # Japanese voices
    JF_ALPHA = "jf_alpha"
    JF_GONGITSUNE = "jf_gongitsune"
    JF_NEZUMI = "jf_nezumi"
    JF_TEBUKURO = "jf_tebukuro"
    JM_KUMO = "jm_kumo"

    # Chinese voices
    ZF_XIAOBEI = "zf_xiaobei"
    ZF_XIAONI = "zf_xiaoni"
    ZF_XIAOXIAO = "zf_xiaoxiao"
    ZF_XIAOYI = "zf_xiaoyi"
    ZM_YUNJIAN = "zm_yunjian"
    ZM_YUNXI = "zm_yunxi"
    ZM_YUNXIA = "zm_yunxia"
    ZM_YUNYANG = "zm_yunyang"


class Model(str, Enum):
    """Available models for Kokoro TTS."""

    KOKORO_82M = "jaaari/kokoro-82m:f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13"


@timeout(TIMEOUT)
def generate(
    text: str,
    voice: Voice | str = Voice.AF_NICOLE,
    model: Model | str = Model.KOKORO_82M,
    seed: Optional[int] = None,
    speed: float = 1.0,
    denoising_strength: float = 0.5,
    output_format: str = "mp3",
) -> bytes:
    """
    Generates audio from text using the Kokoro TTS model via Replicate.

    Args:
        text: Text to convert to speech
        voice: Voice to use
        model: TTS model to use
        seed: Random seed for generation
        speed: Speech speed (default: 1.0)
        denoising_strength: Denoising strength (0-1)
        output_format: Output format (mp3 or wav)

    Returns:
        Audio data as bytes
    """
    voice_str = convert_to_enum(Voice, voice).value
    model_str = convert_to_enum(Model, model).value

    input_data = {
        "text": text,
        "voice": voice_str,
        "speed": speed,
        "denoising_strength": denoising_strength,
        "output_format": output_format,
    }
    if seed is not None:
        input_data["seed"] = seed

    return run(model_str, input_data)


@timeout(TIMEOUT)
async def agenerate(
    text: str,
    voice: Voice | str = Voice.AF_NICOLE,
    model: Model | str = Model.KOKORO_82M,
    seed: Optional[int] = None,
    speed: float = 1.0,
    denoising_strength: float = 0.5,
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
        voice=voice,
        model=model,
        seed=seed,
        speed=speed,
        denoising_strength=denoising_strength,
        output_format=output_format,
    )
