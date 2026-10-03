import pytest

from aqua_drift import storage
from aqua_drift.storage import EventStore


class _Connection:
    async def execute(self, _sql: str) -> None:
        return None


class _Acquire:
    async def __aenter__(self) -> _Connection:
        return _Connection()

    async def __aexit__(self, *_args: object) -> None:
        return None


class _Pool:
    def acquire(self) -> _Acquire:
        return _Acquire()

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_connect_retries_while_database_initializes(monkeypatch) -> None:
    attempts = {"count": 0}

    async def create_pool(*_args: object, **_kwargs: object) -> _Pool:
        attempts["count"] += 1
        if attempts["count"] < 3:  # temporary init server: TCP refused
            raise ConnectionRefusedError(111, "Connection refused")
        return _Pool()

    monkeypatch.setattr(storage.asyncpg, "create_pool", create_pool)
    store = EventStore("postgresql://x", connect_timeout_s=5, retry_delay_s=0)
    await store.connect()
    assert attempts["count"] == 3
    assert store.pool is not None


@pytest.mark.asyncio
async def test_connect_gives_up_after_timeout(monkeypatch) -> None:
    async def create_pool(*_args: object, **_kwargs: object) -> _Pool:
        raise ConnectionRefusedError(111, "Connection refused")

    monkeypatch.setattr(storage.asyncpg, "create_pool", create_pool)
    store = EventStore("postgresql://x", connect_timeout_s=0, retry_delay_s=0)
    with pytest.raises(ConnectionRefusedError):
        await store.connect()
