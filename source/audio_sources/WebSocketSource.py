import numpy as np
from starlette.websockets import WebSocket, WebSocketDisconnect

from interfaces.ISource import IAsyncSource

class WebsocketEndpointSource(IAsyncSource):

    def __init__(self, websocket:WebSocket):
        self._websocket:WebSocket = websocket
        self._active:bool = False

    def open(self) -> None:
        self._active = True

    def is_active(self) -> bool:
        return self._active

    async def get_chunk(self) -> np.ndarray|None:
        try:
            raw_bytes = await self._websocket.receive_bytes()
        except WebSocketDisconnect:
            self._active = False
            return None
        return np.frombuffer(raw_bytes, dtype=np.float32)

    def close(self) -> None:
        self._active = False