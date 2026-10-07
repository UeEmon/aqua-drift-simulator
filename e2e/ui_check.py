"""Headless-browser check of the Cesium GIS (real WebGL via SwiftShader).

Checks: no page/console errors, WebGL2 + MSAA active, comparison cards rendered, no horizontal
overflow in the control panel, view buttons (oblique / top / side) set the camera, centre-on-
estimate puts the estimate at the screen centre. Writes screenshots and GitHub annotations.
"""
from __future__ import annotations

import argparse
import contextlib
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


KEY_NOTES = ("initial view", "perf:", "GPU:", "layer:", "Lloyd")


def note(message: str) -> None:
    # GitHub shows at most 10 annotations of a kind per step: annotate only the key results
    notes.append(message)
    if message.startswith(KEY_NOTES):
        print(f"::notice::{_escape(message)}", flush=True)
    else:
        print(message, flush=True)


def stage(name: str, func) -> None:
    print(f"--- stage: {name}", flush=True)
    notes.append(f"stage {name}")
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
            # MSAA follows the quality level (auto lowers it on slow software WebGL): check that
            # the level's setting is applied and that 高 gives 4x
            expected = {"高": 4, "中": 2, "低": 1, "最低": 1}
            level = page.evaluate("() => window.aquaDrift.telemetry.quality")
            if gl["msaa"] != expected.get(level, -1):
                fail(f"MSAA {gl['msaa']} does not match quality '{level}'")
            high = page.evaluate("""() => { const q = document.getElementById('quality'); q.value = '0';
                q.dispatchEvent(new Event('change')); const m = window.aquaDrift.viewer.scene.msaaSamples;
                q.value = 'auto'; q.dispatchEvent(new Event('change')); return m; }""")
            note(f"MSAA: {gl['msaa']}x at quality {level}, {high}x at 高")
            if high < 4:
                fail(f"MSAA not 4x at quality 高 (msaaSamples={high})")
            if not gl["requestRenderMode"]:
                fail("requestRenderMode is off")
            message = page.inner_text("#message")
            if message.startswith("描画エラー"):
                fail(f"render error shown: {message}")
            page.screenshot(path=str(out / "01-initial.png"))

        def telemetry(label: str) -> None:
            # count who requests frames during the window (diagnostics for the idle check)
            page.evaluate("""() => { const s = window.aquaDrift.viewer.scene; const d = { calls: 0, by: {}, cam: 0 };
                window.__renderDiag = d; if (!s.__origRequestRender) s.__origRequestRender = s.requestRender.bind(s);
                s.requestRender = () => { d.calls++; const line = (new Error().stack || '').split(String.fromCharCode(10))[2] || '?';
                  const key = line.trim().replace(/^at /, '').replace(/https?:[^ )]*[/]/, '').slice(0, 60);
                  d.by[key] = (d.by[key] || 0) + 1; s.__origRequestRender(); };
                const c = window.aquaDrift.viewer.camera; let last = c.positionWC.clone();
                d.sizes = {}; const w = window.aquaDrift.viewer.cesiumWidget || window.aquaDrift.viewer;
                d.off = s.postRender.addEventListener(() => { if (!Cesium.Cartesian3.equalsEpsilon(last, c.positionWC, 0, 1e-6)) d.cam++;
                  last = c.positionWC.clone(); const cv = s.canvas; const doc = document.documentElement;
                  const k = cv.clientWidth + 'x' + cv.clientHeight + ' buf ' + cv.width + 'x' + cv.height + ' dpr ' + window.devicePixelRatio
                    + ' res ' + window.aquaDrift.viewer.resolutionScale + ' doc ' + doc.scrollWidth + 'x' + doc.scrollHeight
                    + ' force ' + !!w._forceResize;
                  d.sizes[k] = (d.sizes[k] || 0) + 1; }); }""")
            t0 = page.evaluate("() => window.aquaDrift.telemetry.frames")
            page.wait_for_timeout(3000)
            data = page.evaluate(
                """() => { const s = window.aquaDrift.viewer.scene; const d = window.__renderDiag;
                  s.requestRender = s.__origRequestRender; d.off();
                  const by = Object.entries(d.by).sort((a, b) => b[1] - a[1]).slice(0, 4);
                  return { t: window.aquaDrift.telemetry, long: window.__longTasks,
                  diag: { requests: d.calls, cameraMoves: d.cam, by, sizes: Object.entries(d.sizes).slice(0, 4), globe: s.globe.show,
                          anim: window.aquaDrift.anim.active.size },
                  pending: ['region','regionOutline','voxels'].map(k => !!window.aquaDrift.gpu[k].pending) }; }"""
            )
            fps = (data["t"]["frames"] - t0) / 3.0
            t = data["t"]
            note(f"telemetry {label}: {fps:.1f} frames/s, render {t['renderMs']:.0f} ms avg / "
                 f"{t['maxRenderMs']:.0f} ms max, long tasks {data['long']}, pending {data['pending']}, "
                 f"swaps {t['swaps']}, dropped {t['pendingDropped']}")
            # with smooth display on, markers glide (continuous frames by design); the idle-loop
            # check applies when it is off (it switches off by itself on slow software WebGL)
            if not t.get("smooth") and fps > 20:
                fail(f"render loop not idle ({label}): {fps:.1f} frames/s with requestRenderMode; {data['diag']}")

        def map_fits(width: int, height: int) -> None:
            # the map (canvas, status strip, camera readout) must lie within the window height
            box = page.evaluate("""() => { const r = (el) => el.getBoundingClientRect();
                const s = window.aquaDrift.viewer.scene; const panel = document.getElementById('control-panel');
                return { win: window.innerHeight, canvas: r(s.canvas).bottom, strip: r(document.getElementById('status-strip')).bottom,
                         hud: r(document.getElementById('camera-hud')).bottom, doc: document.documentElement.scrollHeight,
                         panelScrolls: panel.scrollHeight > panel.clientHeight }; }""")
            note(f"map fits {width}x{height}: canvas bottom {box['canvas']:.0f} / window {box['win']}, "
                 f"status strip {box['strip']:.0f}, page height {box['doc']}, side panel scrolls {box['panelScrolls']}")
            worst = max(box["canvas"], box["strip"], box["hud"], box["doc"])
            if worst > box["win"] + 1:
                fail(f"map exceeds the window height at {width}x{height}: {box}")

        def overflow() -> None:
            for width, height in ((1600, 1000), (1280, 720), (700, 900)):
                page.set_viewport_size({"width": width, "height": height})
                page.wait_for_timeout(1200)
                map_fits(width, height)
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
                # a timeout is reported below with the measured pitch
                with contextlib.suppress(Exception):
                    page.wait_for_function(
                        f"""() => {{ const c = window.aquaDrift.viewer.camera;
                            const p = Cesium.Math.toDegrees(c.pitch);
                            return !c._currentFlight && p >= {low} && p <= {high}; }}""",
                        timeout=20_000,
                    )
                page.wait_for_timeout(500)
                pitch = page.evaluate("() => Cesium.Math.toDegrees(window.aquaDrift.viewer.camera.pitch)")
                note(f"view {view}: camera pitch {pitch:.1f} deg")
                if not (low <= pitch <= high):
                    fail(f"view '{view}' pitch {pitch:.1f} deg, expected {low}..{high}")
                page.screenshot(path=str(out / f"02-view-{view}.png"))

        def check_centred(label: str, limit: float, which: str = "estimate") -> None:
            with contextlib.suppress(Exception):
                page.wait_for_function("() => !window.aquaDrift.viewer.camera._currentFlight", timeout=20_000)
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

        def camera_info() -> dict:
            return page.evaluate("""() => { const a = window.aquaDrift; const s = a.state.latestSnapshot;
                const est = s.estimates.find(e => e.current_position);
                return { info: a.cameraInfo(), truth: s.target && s.target.position,
                         estimate: est && est.current_position, hud: document.getElementById('camera-hud').textContent,
                         follow: document.getElementById('follow-on').checked,
                         target: document.getElementById('center-target').value,
                         flying: !!a.viewer.camera._currentFlight, initial: a.state.initialCamera }; }""")

        def ground_yd(a: dict, b: dict) -> float:
            lat = math.radians((a["latitude"] + b["latitude"]) / 2)
            dn = math.radians(a["latitude"] - b["latitude"]) * 6_371_000
            de = math.radians(a["longitude"] - b["longitude"]) * 6_371_000 * math.cos(lat)
            return math.hypot(dn, de) / 0.9144

        def initial_view() -> None:
            # start-up: follow on (truth), camera placed once (no flight) at 10000 ft above the sea surface;
            # it has not moved since (minutes of target motion have passed by now)
            c = camera_info()
            info = c["info"]
            note(f"initial view: follow {'on' if c['follow'] else 'off'}, camera altitude {info['altitudeFt']:.0f} ft, "
                 f"depression {info['depressionDeg']:.1f} deg, HUD: {' / '.join(c['hud'].splitlines()[:4])}")
            if not c["follow"] or c["target"] != "truth":
                fail(f"follow should be on with the truth target at start (on={c['follow']}, target={c['target']})")
            if "追従中（真値）" not in c["hud"]:
                fail(f"camera HUD does not show following the truth: {c['hud']!r}")
            if not c["initial"]:
                fail("initial view was not applied")
            if abs(info["altitudeFt"] - 10000) > 3 or c["flying"]:
                fail(f"initial camera altitude {info['altitudeFt']:.1f} ft, expected 10000 ft (flying {c['flying']})")
            for label in ("カメラ", "高度", "俯角", "視点中心"):
                if label not in c["hud"]:
                    fail(f"camera HUD lacks '{label}': {c['hud']!r}")
            if "10,000 ft" not in c["hud"]:
                fail(f"camera HUD altitude not 10,000 ft: {c['hud']!r}")
            if not info["centre"]:
                fail("camera HUD: no view centre at the initial oblique view")
            # minutes after start the truth has moved; the view centre must still be on it
            check_centred("initial view follows the truth (default)", 0.03, "truth")
            page.wait_for_timeout(5000)
            later = camera_info()["info"]["altitudeFt"]
            if abs(later - info["altitudeFt"]) > 3:
                fail(f"camera altitude changed without user action (no automatic zoom): {info['altitudeFt']:.0f} -> {later:.0f} ft")
            overlap = page.evaluate("""() => { const r = (id) => document.getElementById(id).getBoundingClientRect();
                const hud = r('camera-hud'); const hit = [];
                for (const id of ['legend', 'view-toolbar', 'status-strip']) { const o = r(id);
                  if (hud.left < o.right && hud.right > o.left && hud.top < o.bottom && hud.bottom > o.top) hit.push(id); }
                return hit; }""")
            if overlap:
                fail(f"camera HUD overlaps {overlap}")

        def follow() -> None:
            # follow on (truth): the view centre is kept on the truth every frame (the centre
            # buttons in earlier stages may have switched the follow target)
            page.evaluate("""() => { const t = document.getElementById('center-target');
                t.value = 'truth'; t.dispatchEvent(new Event('change'));
                const el = document.getElementById('follow-on');
                el.checked = true; el.dispatchEvent(new Event('change')); }""")
            page.wait_for_timeout(10_000)
            check_centred("follow mode truth (after 10 s)", 0.02, "truth")
            c = camera_info()
            if not c["info"]["centre"]:
                fail("follow truth: camera HUD shows no view centre")
            else:
                gap = ground_yd(c["info"]["centre"], c["truth"])
                note(f"follow mode truth: HUD view centre {gap:.0f} yd from the truth, altitude {c['info']['altitudeFt']:.0f} ft")
                if gap > 60:
                    fail(f"follow truth: view centre {gap:.0f} yd from the truth")
            # switch the follow target to the estimate
            page.evaluate("""() => { const el = document.getElementById('center-target');
                el.value = 'estimate'; el.dispatchEvent(new Event('change')); }""")
            page.wait_for_timeout(6000)
            check_centred("follow mode estimate (after 6 s)", 0.05, "estimate")
            # off: the camera stays where it is while the target moves on
            page.evaluate("""() => { const el = document.getElementById('follow-on');
                el.checked = false; el.dispatchEvent(new Event('change'));
                const t = document.getElementById('center-target'); t.value = 'truth'; }""")
            page.wait_for_timeout(500)
            before = camera_info()["info"]
            page.wait_for_timeout(4000)
            after = camera_info()["info"]
            moved = ground_yd(before, after)
            note(f"follow off: camera moved {moved:.1f} yd in 4 s")
            if moved > 1:
                fail(f"follow off but the camera moved {moved:.1f} yd")

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

        def lloyd() -> None:
            # optional Lloyd's mirror depth: off by default, switched on from the compare tab
            page.evaluate("() => document.querySelector(\".tab[data-tab='tab-compare']\").click()")
            start = page.evaluate("""() => ({ checked: document.getElementById('lloyd-enabled').checked,
                summary: document.getElementById('lloyd-summary').textContent,
                cfg: window.aquaDrift.state.latestSnapshot.config.lloyd.enabled })""")
            if start["checked"] or start["cfg"]:
                fail(f"Lloyd's mirror should be off by default: {start}")
            page.evaluate("() => document.getElementById('lloyd-enabled').click()")
            # a timeout is reported below
            with contextlib.suppress(Exception):
                page.wait_for_function(
                    """() => { const l = window.aquaDrift.state.latestSnapshot.lloyd;
                        return l && l.enabled && l.status === 'OK'; }""", timeout=360_000, polling=2000)
            info = page.evaluate("""() => { const s = window.aquaDrift.state.latestSnapshot;
                return { lloyd: s.lloyd, truth: s.target && s.target.position.depth_ft,
                         summary: document.getElementById('lloyd-summary').textContent,
                         rows: document.querySelectorAll('#lloyd-table tbody tr').length,
                         online: (s.estimates.find(e => e.mode === 'ONLINE') || {}) }; }""")
            lloyd_result = info["lloyd"] or {}
            note(f"Lloyd's mirror: {info['summary']} | observers {[(o['observer_id'], o['status'], o['fringes']) for o in lloyd_result.get('observers', [])]}")
            if lloyd_result.get("status") != "OK":
                fail(f"Lloyd's mirror depth not obtained within 6 min: {lloyd_result.get('status')} {info['summary']}")
            else:
                error = lloyd_result["depth_ft"] - info["truth"]
                online = info["online"]
                online_err = (online.get("depth_ft") or 0) - info["truth"] if online.get("depth_ft") is not None else None
                note(f"Lloyd's mirror depth error {error:+.0f} ft (sigma {lloyd_result['sigma_ft']:.0f}), fit "
                     f"{lloyd_result['fit_ms']:.0f} ms, PF depth error {online_err if online_err is None else round(online_err)} ft")
                if abs(error) > 100:
                    fail(f"Lloyd's mirror depth error {error:.0f} ft")
                if info["rows"] < 1:
                    fail("Lloyd table empty")
            page.screenshot(path=str(out / "07-lloyd.png"))
            page.evaluate("() => document.getElementById('lloyd-enabled').click()")
            try:
                page.wait_for_function(
                    "() => { const l = window.aquaDrift.state.latestSnapshot.lloyd; return l && !l.enabled; }",
                    timeout=60_000)
                note("Lloyd's mirror switched off again: " + page.inner_text("#lloyd-summary"))
            except Exception:  # noqa: BLE001
                fail("Lloyd's mirror did not switch off")

        def forward_deployment() -> None:
            snap = page.evaluate("() => window.aquaDrift.state.latestSnapshot")
            ids = sorted(r["state"]["observer_id"] for r in snap["observers"])
            before = len(ids)
            note(f"forward deployment: initial observers {ids}")
            if ids[:4] != ["obs-01", "obs-02", "obs-03", "obs-04"]:
                fail(f"initial observers should be obs-01..obs-04, got {ids}")
            # the layer (設標者) has its own control tab; left turn is the standard
            page.evaluate("() => document.querySelector(\".tab[data-tab='tab-layer']\").click()")
            if page.evaluate("() => document.getElementById('layer-turn').value") != "left":
                fail("layer standard turn should be left")
            try:
                page.wait_for_function(
                    "() => document.querySelector('#layer-detail tbody').innerText.includes('基準旋回')", timeout=15_000)
                note("layer tab: " + page.inner_text("#layer-status") + " / "
                     + page.inner_text("#layer-detail tbody").replace("\n", " "))
            except Exception:  # noqa: BLE001
                fail("layer tab does not show the layer state")
            # the layer (設標者) circles the estimated target while idle
            layer0 = page.evaluate("() => window.aquaDrift.state.latestSnapshot.deployment.layer")
            if not layer0:
                fail("layer state missing")
            else:
                note(f"layer: idle mode {layer0['mode']} at {layer0['speed_kt']:.0f} kt, bank {layer0['bank_deg']:.1f} deg")
            # manual approval: the plan is proposed to the operator first
            page.evaluate("""() => { const el = document.getElementById('drop-approval');
                el.value = 'manual'; el.dispatchEvent(new Event('change')); }""")
            page.wait_for_function(
                "() => window.aquaDrift.state.latestSnapshot.config.layer.approval === 'manual'", timeout=15_000)
            page.evaluate("() => document.getElementById('deploy-now').click()")
            page.wait_for_function("() => !document.getElementById('drop-alert').hidden", timeout=30_000)
            proposed = page.evaluate("""() => window.aquaDrift.state.latestSnapshot.deployment.tasks
                .filter(t => t.status === 'PROPOSED').map(t => t.task_id)""")
            page.wait_for_timeout(3000)
            still = page.evaluate("""() => window.aquaDrift.state.latestSnapshot.deployment.tasks
                .filter(t => t.status === 'PROPOSED').length""")
            if not proposed or still < len(proposed):
                fail(f"proposals should wait for the operator: {proposed}, still {still}")
            page.evaluate("() => document.getElementById('drop-approve-all').click()")
            # timed drops: the layer keeps circling until it is time to leave (path-based timing),
            # so right after approval it may still be in ORBIT
            page.wait_for_function("""() => { const d = window.aquaDrift.state.latestSnapshot.deployment;
                return d.layer && d.tasks.every(t => t.status !== 'PROPOSED'); }""",
                timeout=30_000)
            mode = page.evaluate("() => window.aquaDrift.state.latestSnapshot.deployment.layer.mode")
            note(f"layer: {len(proposed)} proposals approved by the operator, layer {mode}")
            # the planned flight path (飛行予定経路) is sent with the layer state and drawn dashed
            page.wait_for_function("""() => { const l = window.aquaDrift.state.latestSnapshot.deployment.layer;
                return l && Array.isArray(l.planned_path) && l.planned_path.length > 1; }""", timeout=15_000)
            path_points = page.evaluate(
                "() => window.aquaDrift.state.latestSnapshot.deployment.layer.planned_path.length")
            note(f"layer: planned flight path {path_points} points (dashed)")
            banks = []
            for _ in range(10):
                page.wait_for_timeout(1000)
                banks.append(abs(page.evaluate("() => window.aquaDrift.state.latestSnapshot.deployment.layer.bank_deg")))
            if max(banks) > 15.0 + 1e-6:
                fail(f"layer bank exceeded 15 deg: {max(banks):.2f}")
            # the observer is in the water only when the layer reaches the drop point
            planned = page.evaluate("""() => window.aquaDrift.state.latestSnapshot.deployment.tasks
                .filter(t => t.status === 'APPROVED').map(t => [t.task_id, t.planned_tick])""")
            note(f"layer: approved drops with optimal drop times {planned} (task, planned tick)")
            if planned and any(p is None for _, p in planned):
                fail(f"approved drops without a planned drop time: {planned}")
            page.wait_for_function(
                f"() => window.aquaDrift.state.latestSnapshot.observers.length >= {before + 1}", timeout=900_000
            )
            done = page.evaluate("""() => window.aquaDrift.state.latestSnapshot.deployment.tasks
                .filter(t => t.status === 'DONE').map(t => [t.task_id, t.approved_tick, t.planned_tick, t.done_tick])""")
            note(f"layer: drops laid {done} (task, approved tick, planned drop tick, laid tick)")
            if not done:
                fail("no drop task completed by the layer")
            for task_id, approved, planned, laid in done:
                # a drop planned far enough ahead must be laid at the planned time
                if planned is not None and approved is not None and planned - approved >= 120 and abs(laid - planned) > 60:
                    fail(f"drop {task_id} laid at {laid}, planned {planned}")
            page.screenshot(path=str(out / "07a-layer-tab.png"))
            page.evaluate("""() => { const el = document.getElementById('drop-approval');
                el.value = 'auto'; el.dispatchEvent(new Event('change')); }""")
            page.evaluate("() => document.querySelector(\".tab[data-tab='tab-display']\").click()")
            page.wait_for_timeout(1500)
            snap = page.evaluate("() => window.aquaDrift.state.latestSnapshot")
            ids = sorted(r["state"]["observer_id"] for r in snap["observers"])
            note(f"forward deployment: containers started on demand, observers now {ids}")
            # optimal planner (default): positions, number and depths from the tracking information
            history = snap["deployment"]["history"]
            latest = history[-1] if history else {}
            depths = [round(p["depth_ft"]) for p in latest.get("positions", [])]
            table = page.inner_text("#deploy-table tbody")
            note(f"forward deployment (optimal): {len(history)} drops, latest {len(depths)} observers at depths {depths} Ft: "
                 f"{latest.get('reason', '')}")
            if not latest.get("reason", "").startswith("optimal"):
                fail(f"deployment not planned by the optimal planner: {latest.get('reason')}")
            if "最適配置" not in table:
                fail(f"deployment table does not show the optimal plan: {table[:200]}")
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

        stage("initial view", initial_view)
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
        stage("lloyd mirror", lloyd)
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
            handle.writelines(f"- ❌ {line}\n" for line in failures)
    print("FAILURES:", len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    import traceback

    try:
        code = main()
    except BaseException as error:  # noqa: BLE001 - report anything that escaped the stages
        trace = traceback.format_exc()[-1500:]
        fail(f"ui_check crashed: {type(error).__name__}: {error} | {trace}")
        code = 1
    if failures:
        summary = " || ".join(f[:300] for f in failures[:6])
        print(f"::error title=ui_check summary::{_escape(summary)}", flush=True)
        print(f"::error title=last notes::{_escape(' || '.join(n[:200] for n in notes[-6:]))}", flush=True)
    sys.exit(code)
