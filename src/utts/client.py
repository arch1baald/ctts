from .cartesia import CartesiaClient
from .elevenlabs import ElevenLabsClient
from .hume import HumeProviderClient
from .kokoro import KokoroClient
from .openai import OpenAIClient
from .orpheus import OrpheusClient
from .zyphra import ZyphraClient

DEFAULT_TIMEOUT = 10


class UTTSClient:
    """Universal text-to-speech client."""

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
        self.openai = OpenAIClient(api_key=openai_api_key, timeout=timeout) if openai_api_key else None
        self.elevenlabs = ElevenLabsClient(api_key=elevenlabs_api_key, timeout=timeout) if elevenlabs_api_key else None
        self.kokoro = KokoroClient(api_key=replicate_api_key, timeout=timeout) if replicate_api_key else None
        self.orpheus = OrpheusClient(api_key=replicate_api_key, timeout=timeout) if replicate_api_key else None
        self.zyphra = ZyphraClient(api_key=zyphra_api_key, timeout=timeout) if zyphra_api_key else None
        self.hume = HumeProviderClient(api_key=hume_api_key, timeout=timeout) if hume_api_key else None
        self.cartesia = CartesiaClient(api_key=cartesia_api_key, timeout=timeout) if cartesia_api_key else None
