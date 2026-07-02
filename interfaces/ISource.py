from abc import ABC, abstractmethod
import numpy as np


class ISource(ABC):
    """Base lifecycle contract for all audio sources."""

    @abstractmethod
    def open(self) -> None: pass

    @abstractmethod
    def is_active(self) -> bool: pass

    @abstractmethod
    def close(self) -> None: pass


class ISyncSource(ISource):
    """Synchronous (blocking) audio source — e.g. microphone."""

    @abstractmethod
    def get_chunk(self) -> np.ndarray | None: pass


class IAsyncSource(ISource):
    """Asynchronous audio source — e.g. WebSocket."""

    @abstractmethod
    async def get_chunk(self) -> np.ndarray | None: pass