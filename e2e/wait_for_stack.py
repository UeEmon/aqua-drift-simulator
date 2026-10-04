"""Wait until the API is healthy and the estimator publishes a position estimate."""
from __future__ import annotations

import json
import sys
import time
import urllib.request

API = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8091"
deadline = time.time() + 600
last = ""
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"{API}/api/snapshot", timeout=5) as response:
            snap = json.load(response)
        estimates = [e for e in snap["estimates"] if e.get("current_position")]
        last = f"tick={snap['tick']} observers={len(snap['observers'])} estimates={len(estimates)}"
        if estimates and len(snap["observers"]) >= 4:
            print("ready:", last, estimates[0]["observability_status"])
            sys.exit(0)
    except Exception as error:  # noqa: BLE001 - keep polling while containers start
        last = repr(error)
    time.sleep(5)
print(f"::error::stack not ready within 10 min ({last})")
sys.exit(1)
