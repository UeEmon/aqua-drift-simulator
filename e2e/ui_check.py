"""Headless-browser check of the Cesium GIS (real WebGL via SwiftShader).

Checks: no page/console errors, WebGL2 + MSAA active, comparison cards rendered, no horizontal
overflow in the control panel, view buttons (oblique / top / side) set the camera, centre-on-
estimate puts the estimate at the screen centre. Writes screenshots and GitHub annotations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

failures: list[str] = []
notes: list[str] = []


def _escape(message: str) -> str:
    # GitHub workflow-command escaping so multi-line messages still become annotations
    return message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def fail(message: str) -> None:
    failures.append(message)
    print(f"::error::{_escape(message)}", flush=True)


def note(message: str) -> None:
    notes.append(message)
    print(f"::notice::{_escape(message)}", flush=True)


def stage(name: str, func) -> None:
    try:
        func()
    except Exception as error:  # noqa: BLE001 - report and continue with the other checks
        fail(f"{name}: {type(error).__name__}: {error}"[:800])


OVERFLOW_JS = """() => {
  const out = [];
  const check = (el, name) => { if (el && el.scrollWidth > el.clientWidth + 1) out.push(`${name}: ${el.scrollWidth}>${el.clientWidth}`); };
  check(document.getElementById('control-panel'), 'control-panel');
  document.querySelectorAll('.table-wrap').forEach((el, i) => check(el, 'table-wrap#' + i));
  check(document.getElementById('compare-cards'), 'compare-cards');
  document.querySelectorAll('.card').forEach((el, i) => check(el, 'card#' + i));
  return out;
}"""

CENTER_JS = """() => {
  const a = window.aquaDrift; const s = a.state.latestSnapshot;
  const est = s.estimates.find(e => e.current_position);
  if (!est) return null;
  const p = est.current_position;
  const ex = Number(document.getElementById('depth-exaggeration').value) || 1;
  const c = Cesium.Cartesian3.fromDegrees(p.longitude, p.latitude, -p.depth_ft * 0.3048 * ex);
  const st = Cesium.SceneTransforms;
  const fn = st.worldToWindowCoordinates || st.wgs84ToWindowCoordinates;
  const w = fn(a.viewer.scene, c);
  const canvas = a.viewer.scene.canvas;
  return w ? { x: w.x, y: w.y, cw: canvas.clientWidth, ch: canvas.clientHeight } : null;
}"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8090")
    parser.add_argument("--out", default="e2e-artifacts")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.add_init_script(
            """window.__longTasks = { count: 0, total: 0 };
            try { new PerformanceObserver((list) => { for (const e of list.getEntries()) {
              window.__longTasks.count++; window.__longTasks.total += e.duration; } })
              .observe({ entryTypes: ['longtask'] }); } catch (e) {}"""
        )
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        page.goto(args.url, wait_until="load")
        page.wait_for_function("() => window.aquaDrift && window.aquaDrift.state.latestSnapshot", timeout=120_000)
        try:
            page.wait_for_function(
                "() => document.querySelectorAll('#compare-cards .card').length === 8", timeout=240_000
            )
        except Exception:  # noqa: BLE001
            fail("comparison cards did not appear within 4 min")
        page.wait_for_timeout(3000)
        gl = page.evaluate(
            """() => { const s = window.aquaDrift.viewer.scene; return {
              webgl2: !!s.context.webgl2, msaa: s.msaaSamples, requestRenderMode: s.requestRenderMode,
              resolutionScale: window.aquaDrift.viewer.resolutionScale,
              primitives: s.primitives.length }; }"""
        )
        note(f"GPU: {json.dumps(gl)}")
        if not gl["webgl2"]:
            fail("WebGL2 context not active")
        if gl["msaa"] < 2:
            fail(f"MSAA not active (msaaSamples={gl['msaa']})")
        if not gl["requestRenderMode"]:
            fail("requestRenderMode is off")
        message = page.inner_text("#message")
        if message.startswith("描画エラー"):
            fail(f"render error shown: {message}")
        page.screenshot(path=str(out / "01-initial.png"))

        for width in (1600, 1280):
            page.set_viewport_size({"width": width, "height": 1000})
            page.wait_for_timeout(800)
            overflow = page.evaluate(OVERFLOW_JS)
            if overflow:
                fail(f"horizontal overflow at {width}px: {overflow}")
            else:
                note(f"no horizontal overflow at viewport {width}px")
        page.set_viewport_size({"width": 1600, "height": 1000})

        expected_pitch = {"top": -90.0, "side": 0.0, "oblique": None}
        for view in ("top", "side", "oblique"):
            page.click(f".vt[data-view='{view}']")
            page.wait_for_timeout(2500)
            pitch = page.evaluate("() => Cesium.Math.toDegrees(window.aquaDrift.viewer.camera.pitch)")
            note(f"view {view}: camera pitch {pitch:.1f} deg")
            target = expected_pitch[view]
            if target is not None and abs(pitch - target) > 3:
                fail(f"view '{view}' pitch {pitch:.1f} deg, expected {target}")
            if view == "oblique" and not (-80 < pitch < -10):
                fail(f"oblique pitch {pitch:.1f} deg out of range")
            page.screenshot(path=str(out / f"02-view-{view}.png"))

        page.click("#center-estimate")
        page.wait_for_timeout(2000)
        pos = page.evaluate(CENTER_JS)
        if pos is None:
            fail("could not project the estimate to the screen")
        else:
            dx = pos["x"] - pos["cw"] / 2
            dy = pos["y"] - pos["ch"] / 2
            note(f"centre-on-estimate offset from screen centre: {dx:.0f}, {dy:.0f} px")
            if math.hypot(dx, dy) > 0.05 * max(pos["cw"], pos["ch"]):
                fail(f"estimate not centred after button: offset {dx:.0f},{dy:.0f} px")
        page.screenshot(path=str(out / "03-centered.png"))

        def telemetry(label: str) -> None:
            t0 = page.evaluate("() => window.aquaDrift.telemetry.frames")
            page.wait_for_timeout(3000)
            data = page.evaluate(
                """() => ({ t: window.aquaDrift.telemetry, long: window.__longTasks,
                  pending: ['region','regionOutline','voxels'].map(k => !!window.aquaDrift.gpu[k].pending) })"""
            )
            fps = (data["t"]["frames"] - t0) / 3.0
            note(f"telemetry {label}: {fps:.1f} frames/s, long tasks {data['long']}, "
                 f"pending {data['pending']}, swaps {data['t']['swaps']}, dropped {data['t']['pendingDropped']}")
            if fps > 20:
                fail(f"render loop not idle ({label}): {fps:.1f} frames/s with requestRenderMode")

        def follow_and_capture() -> None:
            telemetry("before follow")
            page.evaluate("() => document.getElementById('follow-estimate').click()")
            page.wait_for_function("() => document.getElementById('follow-estimate').checked", timeout=10_000)
            page.wait_for_timeout(5000)
            telemetry("following")
            pos = page.evaluate(CENTER_JS)
            if pos:
                dx = pos["x"] - pos["cw"] / 2
                dy = pos["y"] - pos["ch"] / 2
                note(f"follow mode: estimate offset from centre {dx:.0f}, {dy:.0f} px")
                if math.hypot(dx, dy) > 0.08 * max(pos["cw"], pos["ch"]):
                    fail(f"follow mode lost the estimate: offset {dx:.0f},{dy:.0f} px")
            page.click(".tab[data-tab='tab-compare']")
            page.screenshot(path=str(out / "04-follow.png"))
            page.screenshot(path=str(out / "05-full-page.png"), full_page=True)

        stage("follow / final screenshots", follow_and_capture)
        browser.close()

    unique = list(dict.fromkeys(errors))
    if unique:
        note(f"{len(errors)} console/page errors ({len(unique)} distinct)")
    for error in unique[:8]:
        fail(error[:600])
    (out / "result.json").write_text(
        json.dumps({"failures": failures, "notes": notes, "errors": unique}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write("## GIS E2E\n\n")
            handle.writelines(f"- {line}\n" for line in notes)
            for line in failures:
                handle.write(f"- ❌ {line}\n")
    print("FAILURES:", len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
