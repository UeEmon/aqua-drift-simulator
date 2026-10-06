"""Long-run (soak) check: does anything stop after a while?

Runs the GIS in headless Chromium with the default settings (follow on, truth) for
SOAK_MINUTES and samples every SOAK_INTERVAL_S seconds:

* simulation  : API tick, target tick, observers, doppler tick
* estimator   : latest estimate tick, estimation running
* browser     : tick shown on screen, frames rendered, Cesium render-loop state / error panel,
                render errors recovered, camera altitude, WebSocket status, message line
* containers  : docker compose state of every service (exited / restarting)

A stall is reported as soon as one part stops advancing while the others go on, with the
recent container logs of the suspicious service. Every sample is written to the job summary
and to soak-artifacts/soak.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

API = os.environ.get("API_URL", "http://localhost:8091")
WEB = os.environ.get("WEB_URL", "http://localhost:8090")
MINUTES = float(os.environ.get("SOAK_MINUTES", "25"))
INTERVAL = float(os.environ.get("SOAK_INTERVAL_S", "30"))
OUT = Path(os.environ.get("SOAK_OUT", "soak-artifacts"))

failures: list[str] = []


def esc(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def fail(message: str) -> None:
    failures.append(message)
    print(f"::error::{esc(message)}", flush=True)


def api_sample() -> dict:
    try:
        with urllib.request.urlopen(f"{API}/api/snapshot", timeout=10) as response:
            snap = json.load(response)
    except Exception as error:  # noqa: BLE001
        return {"error": repr(error)}
    estimates = [e for e in snap.get("estimates", []) if e.get("current_position")]
    return {
        "tick": snap["tick"],
        "target_tick": (snap.get("target") or {}).get("tick"),
        "observers": len(snap.get("observers", [])),
        "doppler_tick": (snap.get("doppler") or {}).get("tick"),
        "estimate_tick": max((e["tick"] for e in estimates), default=None),
        "estimation_running": (snap.get("estimation") or {}).get("running"),
    }


def containers() -> dict:
    try:
        out = subprocess.run(
            ["docker", "compose", "ps", "-a", "--format", "json"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception as error:  # noqa: BLE001
        return {"error": repr(error)}
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        rows.extend(data if isinstance(data, list) else [data])
    return {r.get("Service") or r.get("Name"): r.get("State", "?") for r in rows}


def logs_tail(service: str, lines: int = 40) -> str:
    try:
        return subprocess.run(
            ["docker", "compose", "logs", "--no-color", "--tail", str(lines), service],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception as error:  # noqa: BLE001
        return repr(error)


BROWSER_JS = """() => {
  const a = window.aquaDrift; const v = a.viewer; const c = v.camera;
  const panel = document.querySelector('.cesium-widget-errorPanel');
  return {
    tick: Number(document.getElementById('tick').textContent),
    frames: a.telemetry.frames, renderErrors: a.renderFaults ? a.renderFaults.count : 0, frameMs: Math.round(a.telemetry.frameIntervalMs || 0),
    lastRenderError: a.renderFaults ? a.renderFaults.last : '',
    renderLoop: v.useDefaultRenderLoop, errorPanel: panel ? panel.innerText.slice(0, 300) : '',
    connection: document.getElementById('connection-status').textContent,
    message: document.getElementById('message').textContent.slice(0, 160),
    altitudeFt: Math.round(c.positionCartographic.height / 0.3048),
    follow: document.getElementById('follow-on').checked,
    updates: a.telemetry.updates, quality: a.telemetry.quality, latencyMs: Math.round(a.telemetry.updateLatencyMs || 0), smooth: a.telemetry.smooth,
    heapMB: performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1048576) : null,
  };
}"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    page_errors: list[str] = []
    stalls: set[str] = set()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.on("pageerror", lambda e: page_errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: page_errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        page.goto(WEB, wait_until="load")
        page.wait_for_function("() => window.aquaDrift && window.aquaDrift.state.latestSnapshot", timeout=180_000)
        start = time.time()
        prev: dict | None = None
        while time.time() - start < MINUTES * 60:
            time.sleep(INTERVAL)
            api = api_sample()
            try:
                ui = page.evaluate(BROWSER_JS)
            except Exception as error:  # noqa: BLE001
                ui = {"error": repr(error)}
            sample = {"t": round(time.time() - start), "api": api, "ui": ui, "containers": containers()}
            samples.append(sample)
            line = (f"t={sample['t']}s api tick {api.get('tick')} est {api.get('estimate_tick')} obs {api.get('observers')} | "
                    f"ui tick {ui.get('tick')} frames {ui.get('frames')} loop {ui.get('renderLoop')} "
                    f"renderErrors {ui.get('renderErrors')} alt {ui.get('altitudeFt')} ft {ui.get('connection')} "
                    f"heap {ui.get('heapMB')} MB q {ui.get('quality')} latency {ui.get('latencyMs')} ms frame {ui.get('frameMs')} ms smooth {ui.get('smooth')}")
            print(line, flush=True)
            bad = {k: v for k, v in sample["containers"].items() if isinstance(v, str) and v not in ("running",)}
            if bad:
                print(f"  containers not running: {bad}", flush=True)
            if prev:
                checks = {
                    "simulation clock (API tick)": (prev["api"].get("tick"), api.get("tick"), ["clock", "api"]),
                    "target": (prev["api"].get("target_tick"), api.get("target_tick"), ["target"]),
                    "doppler": (prev["api"].get("doppler_tick"), api.get("doppler_tick"), ["doppler"]),
                    "estimator": (prev["api"].get("estimate_tick"), api.get("estimate_tick"), ["estimator"]),
                    "screen tick": (prev["ui"].get("tick"), ui.get("tick"), ["api"]),
                    "screen frames": (prev["ui"].get("frames"), ui.get("frames"), []),
                }
                for name, (before, after, services) in checks.items():
                    if name == "estimator" and not api.get("estimation_running"):
                        continue
                    if before is not None and after is not None and after <= before and name not in stalls:
                        stalls.add(name)
                        fail(f"STALL at t={sample['t']}s: {name} stopped at {after} (sample: {json.dumps(sample, ensure_ascii=False)[:900]})")
                        for service in services:
                            print(f"--- logs {service} ---\n{logs_tail(service)}", flush=True)
                if ui.get("errorPanel"):
                    fail(f"Cesium render loop stopped at t={sample['t']}s: {ui['errorPanel']}")
                if ui.get("renderErrors", 0) > prev["ui"].get("renderErrors", 0):
                    fail(f"render error recovered at t={sample['t']}s: {ui.get('lastRenderError')}")
                lag = (api.get("tick") or 0) - (ui.get("tick") or 0)
                if api.get("tick") and ui.get("tick") is not None and lag > 15 and "screen lag" not in stalls:
                    stalls.add("screen lag")
                    fail(f"screen lags the simulation by {lag} s at t={sample['t']}s")
            prev = sample
        page.screenshot(path=str(OUT / "soak-final.png"))
        browser.close()
    unique = list(dict.fromkeys(page_errors))
    for error in unique[:6]:
        fail(error[:600])
    (OUT / "soak.json").write_text(json.dumps({"samples": samples, "failures": failures, "errors": unique},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    last = samples[-1] if samples else {}
    verdict = "no stall" if not failures else f"{len(failures)} problem(s)"
    print(f"::notice title=soak::{esc(f'{MINUTES:.0f} min soak: {verdict}; last sample {json.dumps(last, ensure_ascii=False)[:700]}')}")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"## Soak ({MINUTES:.0f} min): {verdict}\n\n| t (s) | API tick | estimate tick | screen tick | frames | render loop | render errors | altitude ft | heap MB |\n|---|---|---|---|---|---|---|---|---|\n")
            for s in samples:
                a, u = s["api"], s["ui"]
                handle.write(f"| {s['t']} | {a.get('tick')} | {a.get('estimate_tick')} | {u.get('tick')} | {u.get('frames')} | "
                             f"{u.get('renderLoop')} | {u.get('renderErrors')} | {u.get('altitudeFt')} | {u.get('heapMB')} |\n")
            for line in failures:
                handle.write(f"\n- ❌ {line[:500]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
