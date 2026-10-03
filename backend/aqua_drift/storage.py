from __future__ import annotations

import json
from typing import Any

import asyncpg

from aqua_drift.models import ScenarioConfig

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE TABLE IF NOT EXISTS scenario_config (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS simulation_events (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    entity_id TEXT,
    tick BIGINT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS simulation_events_tick_idx
    ON simulation_events (tick, event_type);
CREATE TABLE IF NOT EXISTS active_observers (
    observer_id TEXT PRIMARY KEY,
    registered_tick BIGINT NOT NULL,
    last_tick BIGINT NOT NULL,
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS target_track (
    tick BIGINT PRIMARY KEY,
    position geometry(PointZ, 4326) NOT NULL,
    depth_ft DOUBLE PRECISION NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS observer_track (
    observer_id TEXT NOT NULL,
    tick BIGINT NOT NULL,
    position geometry(PointZ, 4326) NOT NULL,
    depth_ft DOUBLE PRECISION NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (observer_id, tick)
);
CREATE INDEX IF NOT EXISTS observer_track_position_gix
    ON observer_track USING GIST (position);
"""


class EventStore:
    def __init__(self, database_url: str | None) -> None:
        self.database_url = database_url
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if not self.database_url:
            return
        self.pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)
        async with self.pool.acquire() as connection:
            await connection.execute(SCHEMA)

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def save_config(self, config: ScenarioConfig) -> None:
        if not self.pool:
            return
        payload = json.dumps(config.model_dump(mode="json"))
        await self.pool.execute(
            """
            INSERT INTO scenario_config (id, payload, updated_at)
            VALUES (1, $1::jsonb, now())
            ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload, updated_at = now()
            """,
            payload,
        )

    async def append_event(
        self, event_type: str, tick: int, payload: dict[str, Any], entity_id: str | None = None
    ) -> None:
        if not self.pool:
            return
        await self.pool.execute(
            """
            INSERT INTO simulation_events (event_type, entity_id, tick, payload)
            VALUES ($1, $2, $3, $4::jsonb)
            """,
            event_type,
            entity_id,
            tick,
            json.dumps(payload),
        )

    async def upsert_observer(
        self, observer_id: str, registered_tick: int, last_tick: int, payload: dict[str, Any]
    ) -> None:
        if not self.pool:
            return
        await self.pool.execute(
            """
            INSERT INTO active_observers
                (observer_id, registered_tick, last_tick, payload, updated_at)
            VALUES ($1, $2, $3, $4::jsonb, now())
            ON CONFLICT (observer_id) DO UPDATE SET
                last_tick = EXCLUDED.last_tick,
                payload = EXCLUDED.payload,
                updated_at = now()
            """,
            observer_id,
            registered_tick,
            last_tick,
            json.dumps(payload),
        )

    async def archive_observer(self, observer_id: str) -> None:
        if self.pool:
            await self.pool.execute(
                "DELETE FROM active_observers WHERE observer_id = $1", observer_id
            )

    async def save_target_position(self, tick: int, payload: dict[str, Any]) -> None:
        if not self.pool:
            return
        position = payload["position"]
        await self.pool.execute(
            """
            INSERT INTO target_track (tick, position, depth_ft, payload)
            VALUES ($1, ST_SetSRID(ST_MakePoint($2, $3, $4), 4326), $5, $6::jsonb)
            ON CONFLICT (tick) DO UPDATE SET
                position = EXCLUDED.position,
                depth_ft = EXCLUDED.depth_ft,
                payload = EXCLUDED.payload
            """,
            tick,
            position["longitude"],
            position["latitude"],
            -position["depth_ft"] * 0.3048,
            position["depth_ft"],
            json.dumps(payload),
        )

    async def save_observer_position(
        self, observer_id: str, tick: int, payload: dict[str, Any]
    ) -> None:
        if not self.pool:
            return
        position = payload["position"]
        await self.pool.execute(
            """
            INSERT INTO observer_track (observer_id, tick, position, depth_ft, payload)
            VALUES ($1, $2, ST_SetSRID(ST_MakePoint($3, $4, $5), 4326), $6, $7::jsonb)
            ON CONFLICT (observer_id, tick) DO UPDATE SET
                position = EXCLUDED.position,
                depth_ft = EXCLUDED.depth_ft,
                payload = EXCLUDED.payload
            """,
            observer_id,
            tick,
            position["longitude"],
            position["latitude"],
            -position["depth_ft"] * 0.3048,
            position["depth_ft"],
            json.dumps(payload),
        )
