"""Orchestrator container: starts and stops observer containers on demand.

* reads /internal/orchestrator-feed (active observers, queued placements, initial observers)
* starts one container per needed observer slot (obs-01 .. obs-99, lowest free number first)
  from the same backend image and network as itself, with AutoRemove and a memory limit
* containers exit by themselves when their observer ends (3 h limit / evicted) or when a
  standby timeout passes; idle surplus containers are stopped
* on shutdown (SIGTERM, e.g. `docker compose down`) all managed observer containers are stopped

Requires the Docker Engine socket (/var/run/docker.sock) mounted into this container. Without
it the orchestrator stays idle and observers can be run statically
(`docker compose --profile static-observers up --scale observer=N`).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import time

import httpx

from aqua_drift.orchestration import ManagedContainer, plan, slot_number
from aqua_drift.services.common import API_URL, CURRENT_FIELD_URL, wait_for_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s orchestrator %(message)s")
log = logging.getLogger(__name__)

DOCKER_SOCKET = os.getenv("DOCKER_SOCKET", "/var/run/docker.sock")
LABEL = "aqua-drift.managed"
INTERVAL_S = float(os.getenv("ORCHESTRATOR_INTERVAL_S", "2"))
WARM_STANDBY = int(os.getenv("OBSERVER_WARM_STANDBY", "0"))
MEMORY_MB = int(os.getenv("OBSERVER_MEMORY_MB", "192"))
STANDBY_TIMEOUT_S = int(os.getenv("OBSERVER_STANDBY_TIMEOUT_S", "120"))


class Docker:
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=DOCKER_SOCKET), base_url="http://docker", timeout=30
        )
        self.image = ""
        self.network = ""
        self.project = ""

    async def discover_self(self) -> None:
        """Use this container's own image (same backend code) and compose network."""
        me = (await self.client.get(f"/containers/{socket.gethostname()}/json")).raise_for_status().json()
        self.image = me["Config"]["Image"]
        self.network = next(iter(me["NetworkSettings"]["Networks"]))
        self.project = me["Config"]["Labels"].get("com.docker.compose.project", "aqua-drift-simulator")

    async def managed(self) -> list[ManagedContainer]:
        filters = json.dumps({"label": [f"{LABEL}=observer", f"com.docker.compose.project={self.project}"]})
        rows = (await self.client.get("/containers/json", params={"all": "1", "filters": filters})).raise_for_status().json()
        now = time.time()
        result = []
        for row in rows:
            slot = row["Labels"].get("aqua-drift.slot", "")
            if slot_number(slot) is None:
                continue
            result.append(ManagedContainer(
                slot=slot, container_id=row["Id"], running=row["State"] in ("created", "running", "restarting"),
                age_s=now - row["Created"],
            ))
        return result

    async def start(self, slot: str) -> bool:
        name = f"{self.project}-{slot}"
        body = {
            "Image": self.image,
            "Cmd": ["python", "-m", "aqua_drift.services.observer"],
            "Env": [
                f"API_URL={API_URL}",
                f"CURRENT_FIELD_URL={CURRENT_FIELD_URL}",
                f"OBSERVER_ID={slot}",
                f"OBSERVER_STANDBY_TIMEOUT_S={STANDBY_TIMEOUT_S}",
            ],
            "Labels": {
                LABEL: "observer",
                "aqua-drift.slot": slot,
                # lets `docker compose down --remove-orphans` clean up as well
                "com.docker.compose.project": self.project,
            },
            "HostConfig": {
                "AutoRemove": True,
                "NetworkMode": self.network,
                "Memory": MEMORY_MB * 1024 * 1024,
            },
        }
        created = await self.client.post("/containers/create", params={"name": name}, json=body)
        if created.status_code == 409:
            return False  # previous container with this name is still being removed
        created.raise_for_status()
        container_id = created.json()["Id"]
        (await self.client.post(f"/containers/{container_id}/start")).raise_for_status()
        return True

    async def stop(self, container_id: str) -> None:
        response = await self.client.post(f"/containers/{container_id}/stop", params={"t": "5"})
        if response.status_code not in (204, 304, 404):
            response.raise_for_status()


async def stop_all(docker: Docker) -> None:
    for container in await docker.managed():
        await docker.stop(container.container_id)


async def run() -> None:
    if not os.path.exists(DOCKER_SOCKET):
        log.warning("Docker socket %s not mounted: dynamic observer start-up disabled", DOCKER_SOCKET)
        while True:
            await asyncio.sleep(3600)
    docker = Docker()
    await docker.discover_self()
    log.info("image=%s network=%s project=%s memory=%d MB", docker.image, docker.network, docker.project, MEMORY_MB)
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopping.set)
    await stop_all(docker)  # leftovers from a previous run
    async with httpx.AsyncClient(trust_env=False) as api:
        await wait_for_api(api)
        while not stopping.is_set():
            try:
                feed = (await api.get(f"{API_URL}/internal/orchestrator-feed", timeout=10)).raise_for_status().json()
                containers = await docker.managed()
                decision = plan(
                    limit=feed["limit"],
                    active_ids=feed["active_ids"],
                    pending_placements=feed["pending_placements"],
                    initial_remaining=feed["initial_remaining"],
                    containers=containers,
                    warm_standby=WARM_STANDBY,
                )
                if decision.evict_oldest:
                    await api.post(f"{API_URL}/internal/observers/evict-oldest", params={"count": decision.evict_oldest})
                    log.info("all slots busy: evicted %d oldest observers", decision.evict_oldest)
                for slot in decision.start:
                    if await docker.start(slot):
                        log.info("started observer container %s", slot)
                for container_id in decision.stop:
                    await docker.stop(container_id)
                    log.info("stopped idle observer container %s", container_id[:12])
            except (httpx.HTTPError, KeyError) as error:
                log.warning("orchestration step failed: %s", error)
            try:
                await asyncio.wait_for(stopping.wait(), timeout=INTERVAL_S)
            except TimeoutError:
                pass
    log.info("shutting down: stopping observer containers")
    await stop_all(docker)


if __name__ == "__main__":
    asyncio.run(run())
