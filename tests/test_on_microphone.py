import pytest
from interfaces.ISource import ISource, ISyncSource, IAsyncSource


def test_on_base_source_has_open():
    assert hasattr(ISource, "open")

def test_on_base_source_has_is_active():
    assert hasattr(ISource, "is_active")

def test_on_base_source_has_close():
    assert hasattr(ISource, "close")



def test_on_sync_source_inherits_base():
    assert issubclass(ISyncSource, ISource)

def test_on_sync_source_has_get_chunk():
    assert hasattr(ISyncSource, "get_chunk")


def test_on_async_source_inherits_base():
    assert issubclass(IAsyncSource, ISource)

def test_on_async_source_has_get_chunk():
    assert hasattr(IAsyncSource, "get_chunk")