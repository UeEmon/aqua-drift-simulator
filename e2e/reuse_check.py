"""Observer slot reuse and on-demand containers (runs on the CI host, needs the docker CLI).

1. evict the oldest observer (frees its slot, history kept)
2. wait until its container has exited (AutoRemove)
3. queue a manual placement -> the orchestrator starts a container for the lowest free slot,
   i.e. the freed number, with a new session
4. report observer containers and memory use
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request

API = "http://localhost:8091"
PROJECT = "aqua-drift-simulator"
failures = []


def get(path: str):
    with urllib.request.urlopen(f"{API}{path}", timeout=10) as response:
        return json.load(response)


def post(path: str, body: dict | None = None):
    data = json.dumps(body or {}).encode()
    request = urllib.request.Request(f"{API}{path}", data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def observer_containers() -> list[str]:
    out = subprocess.run(["docker", "ps", "--filter", "label=aqua-drift.managed=observer",
                          "--format", "{{.Names}}"], capture_output=True, text=True, check=True)
    return sorted(out.stdout.split())


def fail(message: str) -> None:
    failures.append(message)
    print(f"::error::{message}", flush=True)


snap = get("/api/snapshot")
active = [r["state"]["observer_id"] for r in snap["observers"]]
print("active before:", active, "containers:", observer_containers())
evicted = post("/internal/observers/evict-oldest?count=1")["evicted"]
if not evicted:
    fail("nothing evicted")
    sys.exit(1)
slot = evicted[0]
deadline = time.time() + 60
while time.time() < deadline and f"{PROJECT}-{slot}" in observer_containers():
    time.sleep(1)
if f"{PROJECT}-{slot}" in observer_containers():
    fail(f"container of evicted {slot} did not exit")
position = snap["observers"][-1]["state"]["position"]
post("/api/observers/placements", {"position": position})
deadline = time.time() + 120
reused = None
while time.time() < deadline:
    snap = get("/api/snapshot")
    for record in snap["observers"]:
        if record["state"]["observer_id"] == slot:
            reused = record
    if reused:
        break
    time.sleep(2)
if not reused:
    fail(f"freed slot {slot} was not reused")
else:
    session = reused["state"]["session"]
    archived = snap["archived_observer_ids"]
    print(f"::notice::slot reuse: {slot} evicted -> archived {archived} -> restarted as session {session}")
    if session < 1 or f"{slot}#0" not in archived:
        fail(f"reuse bookkeeping wrong: session {session}, archived {archived}")
stats = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}"],
                       capture_output=True, text=True, check=True).stdout.strip().splitlines()
units = {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "kB": 1 / 1000, "MB": 1, "GB": 1000, "B": 1 / 2**20}
total = 0.0
for line in stats:
    name, usage = line.split("\t")
    used = usage.split("/")[0].strip()
    for unit, factor in units.items():
        if used.endswith(unit):
            total += float(used[: -len(unit)]) * factor
            break
containers = observer_containers()
print("\n".join(stats))
print(f"::notice::containers: {len(stats)} running ({len(containers)} observer containers {containers}); "
      f"total memory {total:.0f} MiB")
sys.exit(1 if failures else 0)
