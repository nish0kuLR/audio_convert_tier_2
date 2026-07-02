import pytest
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from source.audio_sources.WebSocketSource import WebsocketEndpointSource


@pytest.fixture
def fastapi_app() -> FastAPI:
    app = FastAPI()

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        source = WebsocketEndpointSource(websocket)
        source.open()
        while source.is_active():
            chunk = await source.get_chunk()
            if chunk is None:
                break
        source.close()

    return app


def test_on_open_accepts_connection(fastapi_app):
    client = TestClient(fastapi_app)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_bytes(np.array([1, 2, 3], dtype=np.float32).tobytes())
        websocket.close()


def test_on_get_chunk_returns_audio_bytes(fastapi_app):
    client = TestClient(fastapi_app)
    audio_chunk:np.ndarray = np.array([1, 2, 3], dtype=np.float32)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_bytes(audio_chunk.tobytes())
        websocket.close()


@pytest.fixture
def mock_websocket():
    class MockWebSocket:
        def __init__(self, chunks:list[bytes]):
            self._chunks = chunks
            self._index = 0
            self.accepted = False

        async def accept(self):
            self.accepted = True

        async def receive_bytes(self):
            if self._index >= len(self._chunks):
                from starlette.websockets import WebSocketDisconnect
                raise WebSocketDisconnect()
            chunk = self._chunks[self._index]
            self._index += 1
            return chunk

    return MockWebSocket


@pytest.mark.asyncio
async def test_on_is_active_before_open(mock_websocket):
    websocket = mock_websocket([])
    source = WebsocketEndpointSource(websocket)
    assert source.is_active() is False


@pytest.mark.asyncio
async def test_on_is_active_after_open(mock_websocket):
    websocket = mock_websocket([])
    source = WebsocketEndpointSource(websocket)
    source.open()
    assert source.is_active() is True


@pytest.mark.asyncio
async def test_on_get_chunk_returns_ndarray(mock_websocket):
    audio_chunk:np.ndarray = np.array([1, 2, 3], dtype=np.float32)
    websocket = mock_websocket([audio_chunk.tobytes()])
    source = WebsocketEndpointSource(websocket)
    source.open()
    result:np.ndarray = await source.get_chunk()
    assert np.array_equal(result, audio_chunk)


@pytest.mark.asyncio
async def test_on_get_chunk_returns_none_on_disconnect(mock_websocket):
    websocket = mock_websocket([])
    source = WebsocketEndpointSource(websocket)
    source.open()
    result = await source.get_chunk()
    assert result is None


@pytest.mark.asyncio
async def test_on_is_active_false_after_disconnect(mock_websocket):
    websocket = mock_websocket([])
    source = WebsocketEndpointSource(websocket)
    source.open()
    await source.get_chunk()
    assert source.is_active() is False


@pytest.mark.asyncio
async def test_on_close(mock_websocket):
    websocket = mock_websocket([])
    source = WebsocketEndpointSource(websocket)
    source.open()
    source.close()
    assert source.is_active() is False