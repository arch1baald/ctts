from .cartesia import CartesiaClient
from .elevenlabs import ElevenLabsClient
from .hume import HumeProviderClient
from .kokoro import KokoroClient
from .openai import OpenAIClient
from .orpheus import OrpheusClient
from .zyphra import ZyphraClient

DEFAULT_TIMEOUT = 10


class UTTSClient:
    """
    Universal text-to-speech client.

    Args:
        openai_api_key: OpenAI API key
        elevenlabs_api_key: ElevenLabs API key
        replicate_api_key: Replicate API key
        zyphra_api_key: Zyphra API key
        hume_api_key: Hume API key
        cartesia_api_key: Cartesia API key
        timeout: Timeout for API calls

    Obtain API keys for the services you want to use:
    - [OpenAI](https://platform.openai.com/settings/api-keys)
    - [ElevenLabs](https://elevenlabs.io/app/settings/api-keys)
    - [Replicate](https://replicate.com/account/api-tokens) (for Kokoro and Orpheus)
    - [Zyphra/Zonos](https://playground.zyphra.com/settings/api-keys)
    - [Hume AI](https://platform.hume.ai/settings/keys)
    - [Cartesia](https://play.cartesia.ai/keys)

    Example:
    ```python
    from utts import UTTSClient

    client = UTTSClient(
        openai_api_key="your_openai_api_key",
    )
    audio = client.elevenlabs.generate('Hello, world!')
    ```
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        elevenlabs_api_key: str | None = None,
        replicate_api_key: str | None = None,
        zyphra_api_key: str | None = None,
        hume_api_key: str | None = None,
        cartesia_api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._openai = OpenAIClient(api_key=openai_api_key, timeout=timeout) if openai_api_key else None
        self._elevenlabs = ElevenLabsClient(api_key=elevenlabs_api_key, timeout=timeout) if elevenlabs_api_key else None
        self._kokoro = KokoroClient(api_key=replicate_api_key, timeout=timeout) if replicate_api_key else None
        self._orpheus = OrpheusClient(api_key=replicate_api_key, timeout=timeout) if replicate_api_key else None
        self._zyphra = ZyphraClient(api_key=zyphra_api_key, timeout=timeout) if zyphra_api_key else None
        self._hume = HumeProviderClient(api_key=hume_api_key, timeout=timeout) if hume_api_key else None
        self._cartesia = CartesiaClient(api_key=cartesia_api_key, timeout=timeout) if cartesia_api_key else None

    @property
    def openai(self) -> OpenAIClient:
        if self._openai is None:
            raise ValueError("OpenAI API key is not set")
        return self._openai

    @property
    def elevenlabs(self) -> ElevenLabsClient:
        if self._elevenlabs is None:
            raise ValueError("ElevenLabs API key is not set")
        return self._elevenlabs

    @property
    def kokoro(self) -> KokoroClient:
        if self._kokoro is None:
            raise ValueError("Replicate API key is not set")
        return self._kokoro

    @property
    def orpheus(self) -> OrpheusClient:
        if self._orpheus is None:
            raise ValueError("Replicate API key is not set")
        return self._orpheus

    @property
    def zyphra(self) -> ZyphraClient:
        if self._zyphra is None:
            raise ValueError("Zyphra API key is not set")
        return self._zyphra

    @property
    def hume(self) -> HumeProviderClient:
        if self._hume is None:
            raise ValueError("Hume API key is not set")
        return self._hume

    @property
    def cartesia(self) -> CartesiaClient:
        if self._cartesia is None:
            raise ValueError("Cartesia API key is not set")
        return self._cartesia
