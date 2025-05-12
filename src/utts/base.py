from abc import ABC, abstractmethod
from typing import Any


class ProviderClient(ABC):
    """Base class for all provider clients."""

    @abstractmethod
    def get_client(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def generate(self, text: str, *args: Any, **kwargs: Any) -> bytes:
        pass
