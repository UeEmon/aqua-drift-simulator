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


KEY_NOTES = ("perf:", "GPU:", "forward deployment:", "follow mode", "default view centre",
             "centre-on-estimate", "background map:", "view oblique", "telemetry following")


def note(message: str) -> None:
    # GitHub shows at most 10 annotations of a kind per step: annotate only the key results
    notes.append(message)
    if message.startswith(KEY_NOTES):
        print(f"::notice::{_escape(message)}", flush=True)
    else:
        print(message, flush=True)


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

CENTER_JS = """(which) => {
  const a = window.aquaDrift; const s = a.state.latestSnapshot;
  let p = null;
  if (which === 'truth') { p = s.target && s.target.position; }
  else { const est = s.estimates.find(e => e.current_position); p = est && est.current_position; }
  if (!p) return null;
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
        def gpu_info() -> None:
            gl = page.evaluate(
                """() => { const s = window.aquaDrift.viewer.scene; return {
                  webgl2: !!s.context.webgl2, msaa: s.msaaSamples, requestRenderMode: s.requestRenderMode,
                  resolutionScale: window.aquaDrift.viewer.resolutionScale, primitives: s.primitives.length,
                  renderer: (() => { try { const g = s.context._gl; const d = g.getExtension('WEBGL_debug_renderer_info');
                    return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : g.getParameter(g.RENDERER); } catch (e) { return '?'; } })() }; }"""
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

        def telemetry(label: str) -> None:
            t0 = page.evaluate("() => window.aquaDrift.telemetry.frames")
            page.wait_for_timeout(3000)
            data = page.evaluate(
                """() => ({ t: window.aquaDrift.telemetry, long: window.__longTasks,
                  pending: ['region','regionOutline','voxels'].map(k => !!window.aquaDrift.gpu[k].pending) })"""
            )
            fps = (data["t"]["frames"] - t0) / 3.0
            t = data["t"]
            note(f"telemetry {label}: {fps:.1f} frames/s, render {t['renderMs']:.0f} ms avg / "
                 f"{t['maxRenderMs']:.0f} ms max, long tasks {data['long']}, pending {data['pending']}, "
                 f"swaps {t['swaps']}, dropped {t['pendingDropped']}")
            # with smooth display on, markers glide (continuous frames by design); the idle-loop
            # check applies when it is off (it switches off by itself on slow software WebGL)
            if not t.get("smooth") and fps > 20:
                fail(f"render loop not idle ({label}): {fps:.1f} frames/s with requestRenderMode")

        def overflow() -> None:
            for width in (1600, 1280):
                page.set_viewport_size({"width": width, "height": 1000})
                page.wait_for_timeout(800)
                found = page.evaluate(OVERFLOW_JS)
                if found:
                    fail(f"horizontal overflow at {width}px: {found}")
                else:
                    note(f"no horizontal overflow at viewport {width}px")
            page.set_viewport_size({"width": 1600, "height": 1000})

        def default_centre() -> None:
            selected = page.evaluate("() => document.getElementById('center-target').value")
            if selected != "truth":
                fail(f"default centre target is '{selected}', expected 'truth'")
            page.evaluate("() => document.querySelector(\".vt[data-view='oblique']\").click()")
            page.wait_for_timeout(3000)
            check_centred("default view centre", 0.05, "truth")
            page.evaluate("() => document.getElementById('center-truth').click()")
            page.wait_for_timeout(3000)
            check_centred("centre-on-truth button", 0.05, "truth")

        def views() -> None:
            # wait for the camera flight to finish (software WebGL frames can take seconds)
            ranges = {"top": (-93.0, -87.0), "side": (-3.0, 3.0), "oblique": (-80.0, -10.0)}
            for view in ("top", "side", "oblique"):
                low, high = ranges[view]
                page.evaluate(f"() => document.querySelector(\".vt[data-view='{view}']\").click()")
                try:
                    page.wait_for_function(
                        f"""() => {{ const c = window.aquaDrift.viewer.camera;
                            const p = Cesium.Math.toDegrees(c.pitch);
                            return !c._currentFlight && p >= {low} && p <= {high}; }}""",
                        timeout=20_000,
                    )
                except Exception:  # noqa: BLE001 - reported below with the measured pitch
                    pass
                page.wait_for_timeout(500)
                pitch = page.evaluate("() => Cesium.Math.toDegrees(window.aquaDrift.viewer.camera.pitch)")
                note(f"view {view}: camera pitch {pitch:.1f} deg")
                if not (low <= pitch <= high):
                    fail(f"view '{view}' pitch {pitch:.1f} deg, expected {low}..{high}")
                page.screenshot(path=str(out / f"02-view-{view}.png"))

        def check_centred(label: str, limit: float, which: str = "estimate") -> None:
            try:
                page.wait_for_function("() => !window.aquaDrift.viewer.camera._currentFlight", timeout=20_000)
            except Exception:  # noqa: BLE001
                pass
            pos = page.evaluate(CENTER_JS, which)
            name = "truth" if which == "truth" else "estimate"
            if pos is None:
                fail(f"{label}: could not project the {name} to the screen")
                return
            dx = pos["x"] - pos["cw"] / 2
            dy = pos["y"] - pos["ch"] / 2
            note(f"{label}: {name} offset from screen centre {dx:.0f}, {dy:.0f} px")
            if math.hypot(dx, dy) > limit * max(pos["cw"], pos["ch"]):
                fail(f"{label}: {name} not centred, offset {dx:.0f},{dy:.0f} px")

        def centre() -> None:
            page.evaluate("() => document.getElementById('center-estimate').click()")
            page.wait_for_timeout(3000)
            check_centred("centre-on-estimate", 0.05)
            page.screenshot(path=str(out / "03-centered.png"))

        def follow() -> None:
            page.evaluate("() => document.getElementById('follow-estimate').click()")
            page.wait_for_function("() => document.getElementById('follow-estimate').checked", timeout=10_000)
            page.wait_for_timeout(8000)
            check_centred("follow mode (default: truth)", 0.08, "truth")

        def final_screens() -> None:
            page.evaluate("() => document.querySelector(\".tab[data-tab='tab-compare']\").click()")
            page.screenshot(path=str(out / "04-follow.png"))
            page.screenshot(path=str(out / "05-full-page.png"), full_page=True)

        def base_map() -> None:
            info = page.evaluate(
                "() => ({ layers: window.aquaDrift.viewer.imageryLayers.length, "
                "loaded: !!window.aquaDrift.baseMap.layer, checked: document.getElementById('show-basemap').checked })"
            )
            note(f"base map at start: {info}")
            if info["layers"] != 0 or info["loaded"] or info["checked"]:
                fail(f"background map should be off and not loaded at start: {info}")
            page.evaluate("() => document.getElementById('show-basemap').click()")
            page.wait_for_function(
                "() => window.aquaDrift.baseMap.layer && window.aquaDrift.baseMap.layer.show", timeout=30_000
            )
            page.wait_for_timeout(2000)
            page.screenshot(path=str(out / "06-basemap-on.png"))
            page.evaluate("() => document.getElementById('show-basemap').click()")
            page.wait_for_timeout(500)
            if page.evaluate("() => window.aquaDrift.baseMap.layer.show"):
                fail("background map did not switch off")
            note("background map: off at start, loads on demand, switches off again")

        def forward_deployment() -> None:
            snap = page.evaluate("() => window.aquaDrift.state.latestSnapshot")
            before = len(snap["observers"])
            standby = snap["deployment"]["standby_count"]
            note(f"forward deployment: {before} active observers, {standby} standby before request")
            if standby < 2:
                fail(f"expected standby observers for forward deployment, got {standby}")
                return
            page.evaluate("() => document.querySelector(\".tab[data-tab='tab-display']\").click()")
            page.evaluate("() => document.getElementById('deploy-now').click()")
            page.wait_for_function(
                f"() => window.aquaDrift.state.latestSnapshot.observers.length >= {before + 2}", timeout=30_000
            )
            snap = page.evaluate("() => window.aquaDrift.state.latestSnapshot")
            note(f"forward deployment: {len(snap['observers'])} active observers after 'deploy now', "
                 f"history {len(snap['deployment']['history'])}")
            page.wait_for_timeout(2000)
            page.screenshot(path=str(out / "07-forward-deployment.png"))

        def perf() -> None:
            full = page.evaluate("() => fetch('/api/snapshot').then(r => r.text()).then(t => t.length)")
            t = page.evaluate("() => window.aquaDrift.telemetry")
            truth_points = page.evaluate(
                "() => (window.aquaDrift.tracks.get('truth') || { points: [] }).points.length"
            )
            ratio = full / max(t["bytes"], 1)
            note(
                f"perf: stream {t['bytes'] / 1024:.1f} KB/update vs full snapshot {full / 1024:.1f} KB "
                f"({ratio:.1f}x smaller, before deflate); update {t['updateMs']:.1f} ms (max {t['maxUpdateMs']:.0f}); "
                f"decode {t['parseMs']:.1f} ms in {'Web Worker' if t['worker'] else 'main thread'}; "
                f"quality {t['quality']}; smooth {t['smooth']}; frame interval {t['frameIntervalMs']:.0f} ms; "
                f"truth track {truth_points} pts"
            )
            if not t["worker"]:
                fail("stream decoding is not running in the Web Worker")
            if t["bytes"] > 0.35 * full:
                fail(f"delta stream not smaller than the full snapshot: {t['bytes']:.0f} vs {full} bytes")
            if t["updateMs"] > 100:
                fail(f"main-thread update too slow: {t['updateMs']:.1f} ms")
            if truth_points < 10:
                fail(f"truth track not rendered from the stream ({truth_points} points)")

        stage("gpu info", gpu_info)
        stage("background map", base_map)
        stage("telemetry idle", lambda: telemetry("idle"))
        stage("overflow", overflow)
        stage("default centre is truth", default_centre)
        stage("views", views)
        stage("telemetry after views", lambda: telemetry("after views"))
        stage("stream performance", perf)
        stage("centre on estimate", centre)
        stage("follow", follow)
        stage("telemetry following", lambda: telemetry("following"))
        stage("forward deployment", forward_deployment)
        stage("final screenshots", final_screens)
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
