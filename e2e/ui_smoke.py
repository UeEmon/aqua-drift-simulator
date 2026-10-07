"""Smoke test of the web UI without the backend: does the Cesium 3D view start and draw?

Serves web/ (with Cesium from web/node_modules) on a local port, opens it in headless Chromium
(real WebGL via SwiftShader) and checks: no page errors, the Cesium viewer is created with a
WebGL context and a visible canvas, frames are rendered and the canvas shows the globe. The API
and WebSocket are not running, so the page stays in "再接続中"; that is expected here.

Usage: python e2e/ui_smoke.py [--out ui-smoke-artifacts]   (run `npm ci` in web/ first)
"""
from __future__ import annotations

import argparse
import functools
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parent.parent / "web"
CESIUM = WEB / "node_modules" / "cesium" / "Build" / "Cesium"


class Handler(SimpleHTTPRequestHandler):
    # same layout as the nginx image: web/ at /, Cesium's build at /cesium/
    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/cesium/"):
            return str(CESIUM / clean[len("/cesium/"):])
        return super().translate_path(path)

    def log_message(self, *args) -> None:  # keep the CI log short
        pass


# the stack is not running: connection failures to /ws (and /api) are expected
EXPECTED = ("WebSocket", "/ws", "/api/", "ERR_CONNECTION_REFUSED")

FRAME_JS = """() => new Promise((resolve) => {
  const viewer = window.aquaDrift.viewer; const scene = viewer.scene;
  const off = scene.postRender.addEventListener(() => {
    off();
    // read the drawing buffer in the same task as the frame (preserveDrawingBuffer is off)
    const src = scene.canvas; const c = document.createElement('canvas');
    c.width = src.width; c.height = src.height;
    const g = c.getContext('2d'); g.drawImage(src, 0, 0);
    const px = (fx, fy) => Array.from(g.getImageData(Math.floor(src.width * fx), Math.floor(src.height * fy), 1, 1).data);
    resolve({ center: px(0.5, 0.5), corner: px(0.01, 0.01) });
  });
  scene.requestRender();
})"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ui-smoke-artifacts")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (CESIUM / "Cesium.js").exists():
        print(f"::error::Cesium build not found at {CESIUM}; run `npm ci` in web/", flush=True)
        return 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"

    failures: list[str] = []
    errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on(
            "console",
            lambda m: errors.append(f"console.error: {m.text}")
            if m.type == "error" and not any(s in m.text for s in EXPECTED)
            else None,
        )
        try:
            page.goto(url, wait_until="load")
            page.wait_for_function("() => window.aquaDrift && window.aquaDrift.viewer", timeout=60_000)
            info = page.evaluate(
                """() => { const s = window.aquaDrift.viewer.scene; const c = s.canvas;
                  return { webgl2: !!s.context.webgl2, gl: !!s.context._gl,
                           width: c.clientWidth, height: c.clientHeight,
                           message: document.getElementById('message').innerText }; }"""
            )
            print(f"viewer: {info}", flush=True)
            if not info["gl"]:
                failures.append("Cesium viewer has no WebGL context")
            if info["width"] < 200 or info["height"] < 200:
                failures.append(f"Cesium canvas too small: {info['width']}x{info['height']}")
            if info["message"].startswith("描画エラー"):
                failures.append(f"render error shown: {info['message']}")

            # the sea surface (globe) is off by default and there is no data: turn it on through
            # the control panel so there is something to draw. The initial camera looks down on
            # the whole globe, so the centre is the globe and the corner is the background
            page.check("#show-sea")
            page.wait_for_function(
                """() => { const s = window.aquaDrift.viewer.scene; s.requestRender();
                  return s.globe.show && s.globe.tilesLoaded; }""",
                timeout=60_000,
                polling=500,
            )
            frame = page.evaluate(FRAME_JS)
            print(f"pixels: {frame}", flush=True)
            if frame["center"] == frame["corner"]:
                failures.append(f"canvas looks blank: centre {frame['center']} == corner {frame['corner']}")
            if sum(frame["center"][:3]) == 0:
                failures.append(f"globe not drawn: centre pixel {frame['center']}")
            page.wait_for_timeout(1000)
            render_faults = page.evaluate("() => window.aquaDrift.renderFaults.count")
            if render_faults:
                failures.append(f"Cesium render loop errors: {render_faults}")
        except Exception as error:  # noqa: BLE001 - report as a failure with the screenshot
            failures.append(f"{type(error).__name__}: {error}"[:800])
        finally:
            page.screenshot(path=str(out / "ui-smoke.png"))
            browser.close()
            server.shutdown()

    failures.extend(errors)
    for message in failures:
        print(f"::error::{message}", flush=True)
    if failures:
        return 1
    print("UI smoke test passed: Cesium viewer started and drew the globe", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
