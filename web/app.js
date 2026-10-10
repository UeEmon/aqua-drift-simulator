/* global Cesium */
"use strict";

const FT_TO_M = 0.3048;
const YD_TO_M = 0.9144;
const EARTH_R = 6371008.8;
const REGION_REBUILD_MS = 3000; // presence-region geometry rebuild throttle
const MAX_HISTORY = 3600;

const $ = (id) => document.getElementById(id);
const checked = (id) => Boolean($(id) && $(id).checked);

// ================================================================== viewer (GPU settings)
const viewer = new Cesium.Viewer("cesiumContainer", {
  animation: false,
  timeline: false,
  baseLayerPicker: false,
  geocoder: false,
  homeButton: true,
  sceneModePicker: true,
  navigationHelpButton: false,
  infoBox: false,
  selectionIndicator: false,
  terrainProvider: new Cesium.EllipsoidTerrainProvider(),
  baseLayer: false,
  // render only when something changed (data update, camera move, tile load):
  // the GPU stays idle between the 1 Hz updates instead of redrawing at 60 fps
  requestRenderMode: true,
  maximumRenderTimeChange: Infinity,
  msaaSamples: 2, // hardware multisample anti-aliasing (WebGL2); set per quality level below
  showRenderLoopErrors: false, // recovered in scene.renderError below
  // plain sorted alpha blending instead of order-independent translucency: OIT needs extra
  // full-screen accumulation buffers (x MSAA x device pixels) for the translucent presence region
  orderIndependentTranslucency: false,
  contextOptions: { webgl: { powerPreference: "high-performance", alpha: false } },
});
const scene = viewer.scene;
// an exception inside Cesium's frame stops its render loop for good ("Rendering has stopped");
// record it and restart the loop instead of leaving a frozen map
const renderFaults = { count: 0, last: "", lastAt: 0 };
scene.renderError.addEventListener((_scene, error) => {
  renderFaults.count += 1;
  renderFaults.last = String((error && error.message) || error);
  if (window.console) console.error("render error", error);
  const now = performance.now();
  const soon = now - renderFaults.lastAt < 5000;
  renderFaults.lastAt = now;
  setTimeout(() => {
    viewer.useDefaultRenderLoop = true;
    scene.requestRender();
  }, soon ? 2000 : 200);
  setMessage(`描画エラーから復帰しました（${renderFaults.count} 回目）: ${renderFaults.last}`);
});
viewer.useBrowserRecommendedResolution = false; // full device-pixel resolution on HiDPI
viewer.resolutionScale = Math.min(window.devicePixelRatio || 1, 2);
if (scene.postProcessStages && scene.postProcessStages.fxaa) scene.postProcessStages.fxaa.enabled = true;
scene.globe.depthTestAgainstTerrain = false;
scene.globe.showGroundAtmosphere = false;
if (scene.globe.translucency) {
  scene.globe.translucency.enabled = true;
  scene.globe.translucency.frontFaceAlpha = 0.55;
  scene.globe.translucency.backFaceAlpha = 0.55;
}
scene.screenSpaceCameraController.enableCollisionDetection = false; // allow camera under the sea surface

// Rendering priority: no sky box, atmosphere, sun, moon or fog; plain dark sea-surface globe.
if (scene.skyBox) scene.skyBox.show = false;
if (scene.skyAtmosphere) scene.skyAtmosphere.show = false;
if (scene.sun) scene.sun.show = false;
if (scene.moon) scene.moon.show = false;
if (scene.fog) scene.fog.enabled = false;
scene.backgroundColor = Cesium.Color.fromCssColorString("#03080f");
scene.globe.baseColor = Cesium.Color.fromCssColorString("#0b2238");
scene.globe.show = false; // sea surface off by default (translucent globe = extra render passes)

function applyGlobe() {
  scene.globe.show = checked("show-sea") || Boolean(baseMap.layer && baseMap.layer.show);
  scene.requestRender();
}

// Background map (Natural Earth II): off by default, loaded only when first switched on.
const baseMap = { layer: null, loading: null };
async function setBaseMap(on) {
  for (const id of ["show-basemap", "show-basemap-2"]) if ($(id)) $(id).checked = on;
  if (on && !baseMap.layer) {
    baseMap.loading = baseMap.loading || Cesium.TileMapServiceImageryProvider.fromUrl(
      Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII"),
    );
    try {
      const provider = await baseMap.loading;
      baseMap.layer = baseMap.layer || viewer.imageryLayers.addImageryProvider(provider);
    } catch (error) {
      setMessage("背景地図（Natural Earth II）の読込みに失敗しました。");
      baseMap.loading = null;
      return;
    }
  }
  if (baseMap.layer) baseMap.layer.show = on;
  applyGlobe();
}
for (const id of ["show-basemap", "show-basemap-2"]) {
  if ($(id)) $(id).addEventListener("change", () => setBaseMap($(id).checked));
}

const COLORS = {
  truth: Cesium.Color.CYAN,
  online: Cesium.Color.ORANGE,
  smoothed: Cesium.Color.LIME,
  region: Cesium.Color.fromCssColorString("#ff4fd8"),
  observer: Cesium.Color.GOLD,
  detecting: Cesium.Color.fromCssColorString("#5dff9d"),
  inactive: Cesium.Color.GRAY,
  bearing: Cesium.Color.fromCssColorString("#ffe680"),
  drop: Cesium.Color.fromCssColorString("#ff9f43"),
  layer: Cesium.Color.fromCssColorString("#e9e4ff"),
  proposed: Cesium.Color.fromCssColorString("#ffd479"),
  cancel: Cesium.Color.fromCssColorString("#ff6b5e"), // drops the replanner proposes to cancel
  error: Cesium.Color.WHITE,
  flightWind: Cesium.Color.fromCssColorString("#a0e7ff"), // wind at the layer's flight altitude
  dropWind: Cesium.Color.fromCssColorString("#d59bff"), // mean wind used to correct the release points
  current: Cesium.Color.fromCssColorString("#3d8bff"), // external force (current) from observer drift
};

// GPU-batched primitive collections: one draw call per collection instead of one per entity
const gpu = {
  lines: scene.primitives.add(new Cesium.PolylineCollection()),
  points: scene.primitives.add(new Cesium.PointPrimitiveCollection()),
  labels: scene.primitives.add(new Cesium.LabelCollection()),
  regionLabels: scene.primitives.add(new Cesium.LabelCollection()),
  drops: scene.primitives.add(new Cesium.PointPrimitiveCollection()),
  voxelPoints: scene.primitives.add(new Cesium.PointPrimitiveCollection()),
  dropLabels: scene.primitives.add(new Cesium.LabelCollection()),
  layerLines: scene.primitives.add(new Cesium.PolylineCollection()),
  taskPoints: scene.primitives.add(new Cesium.PointPrimitiveCollection()),
  taskLabels: scene.primitives.add(new Cesium.LabelCollection()),
  releaseLines: scene.primitives.add(new Cesium.PolylineCollection()), // release point -> drop / entry point
  vectors: scene.primitives.add(new Cesium.PolylineCollection()), // wind and external-force arrows
  vectorLabels: scene.primitives.add(new Cesium.LabelCollection()),
  region: { current: null, pending: null },
  regionOutline: { current: null, pending: null },
  voxels: { current: null, pending: null },
};

function colorMaterial(color) {
  return Cesium.Material.fromType("Color", { color });
}
function dashMaterial(color, dashLength = 12) {
  return Cesium.Material.fromType("PolylineDash", { color, dashLength });
}

const state = {
  region: null,
  observers: new Map(), // id -> {point, label, track, range}
  bearingLines: new Map(),
  latestConfig: null,
  latestSnapshot: null,
  formsLoaded: false,
  firstFix: true,
  lastRegionKey: "",
  runKey: "",
  generation: null,
  history: [],
  trueCpa: new Map(),
  cancelKept: new Set(), // cancel proposals the operator chose to keep flying (継続)
};
// a cancel proposal is identified by its task and when it was made, so a later re-proposal shows again
const cancelKey = (t) => `${t.task_id}:${t.cancel_suggested_tick ?? ""}`;

// ================================================================== helpers
function exaggeration() {
  const value = Number($("depth-exaggeration").value);
  return Number.isFinite(value) && value >= 1 ? value : 1;
}
function heightOf(depthFt) {
  return -depthFt * FT_TO_M * exaggeration();
}
function cart(longitude, latitude, depthFt) {
  return Cesium.Cartesian3.fromDegrees(longitude, latitude, heightOf(depthFt));
}
function cartOf(position) {
  return cart(position.longitude, position.latitude, position.depth_ft);
}
function fmt(value, digits = 1) {
  return value == null || !Number.isFinite(Number(value)) ? "--" : Number(value).toFixed(digits);
}
function signed(value, digits = 1) {
  if (value == null || !Number.isFinite(Number(value))) return "--";
  const v = Number(value);
  return `${v >= 0 ? "+" : ""}${v.toFixed(digits)}`;
}
function setText(id, text) {
  const element = $(id);
  if (element) element.textContent = text;
}
function setMessage(text) {
  setText("message", text);
}
// ---- simulation time of day: tick t is at epoch_s + t (epoch_s = system time of tick 0, fixed by
// the clock, which keeps the simulation on the system time at real-time speed). Shown as local
// clock time HH:MM:SS; seconds since start only while the epoch is not known yet.
function epochS() {
  const epoch = state.latestSnapshot?.epoch_s;
  return epoch == null || !Number.isFinite(Number(epoch)) ? null : Number(epoch);
}
const pad2 = (n) => String(n).padStart(2, "0");
function clockAt(tick) {
  if (tick == null || !Number.isFinite(Number(tick))) return "--:--:--";
  const epoch = epochS();
  if (epoch == null) return `${Math.round(Number(tick))} s`;
  const d = new Date(Math.round((epoch + Number(tick)) * 1000));
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
}
// "HH:MM" / "HH:MM:SS" (local) -> tick, the occurrence nearest to the current simulation time;
// a plain number is taken as a tick (seconds since tick 0). null when it cannot be read.
function tickFromClock(text) {
  const value = String(text || "").trim();
  if (/^\d+(\.\d+)?$/.test(value)) return Math.round(Number(value));
  const m = value.match(/^(\d{1,2}):(\d{2})(?::(\d{2}))?$/);
  const epoch = epochS();
  const tick = state.latestSnapshot?.tick ?? 0;
  if (!m || epoch == null) return null;
  const now = new Date(Math.round((epoch + tick) * 1000));
  const at = new Date(now);
  at.setHours(Number(m[1]), Number(m[2]), Number(m[3] || 0), 0);
  let diff = (at - now) / 1000;
  if (diff < -43200) diff += 86400;
  if (diff > 43200) diff -= 86400;
  return Math.max(0, tick + Math.round(diff));
}
function latText(lat) {
  return `${Math.abs(lat).toFixed(5)}°${lat >= 0 ? "N" : "S"}`;
}
function lonText(lon) {
  return `${Math.abs(lon).toFixed(5)}°${lon >= 0 ? "E" : "W"}`;
}
function offsetM(a, b) {
  const meanLat = ((a.latitude + b.latitude) / 2) * Math.PI / 180;
  const north = (b.latitude - a.latitude) * Math.PI / 180 * EARTH_R;
  const east = (b.longitude - a.longitude) * Math.PI / 180 * EARTH_R * Math.cos(meanLat);
  return { east, north };
}
function angleDiff(a, b) {
  return ((a - b + 540) % 360) - 180;
}
function destination(position, bearingDeg, distanceM) {
  const lat = position.latitude * Math.PI / 180;
  const b = bearingDeg * Math.PI / 180;
  const dLat = (distanceM * Math.cos(b)) / EARTH_R;
  const dLon = (distanceM * Math.sin(b)) / (EARTH_R * Math.cos(lat));
  return { longitude: position.longitude + dLon * 180 / Math.PI, latitude: position.latitude + dLat * 180 / Math.PI };
}
function pointInPolygon(lon, lat, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if ((yi > lat) !== (yj > lat) && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function tabVisible(id) {
  const panel = $(id);
  return !panel || !panel.classList || !panel.classList.contains || panel.classList.contains("active");
}

function setHtml(element, html) {
  // skip DOM work when nothing changed (most panels change rarely)
  if (!element || element._html === html) return;
  element.innerHTML = html;
  element._html = html;
}

function selectedMode() {
  const input = document.querySelector("input[name='panel-mode']:checked");
  return input ? input.value : "ONLINE";
}
function selectedEstimate(snapshot = state.latestSnapshot) {
  if (!snapshot) return null;
  return snapshot.estimates.find((item) => item.mode === selectedMode()) || snapshot.estimates[0] || null;
}
function escapeHtml(text) {
  return String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// ================================================================== tabs & panel resizer
for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    for (const other of document.querySelectorAll(".tab")) other.classList.toggle("active", other === tab);
    for (const panel of document.querySelectorAll(".tab-panel")) panel.classList.toggle("active", panel.id === tab.dataset.tab);
    if (state.latestSnapshot) render(state.latestSnapshot); // fill the newly visible panel now
  });
}

(function setupResizer() {
  const resizer = $("panel-resizer");
  const app = $("app");
  if (!resizer || !app || !app.style) return;
  try {
    const saved = Number(localStorage.getItem("aqua-panel-width"));
    if (saved >= 340) app.style.setProperty("--panel-w", `${saved}px`);
  } catch (error) {
    /* storage unavailable: default width */
  }
  let dragging = false;
  resizer.addEventListener("pointerdown", (event) => {
    dragging = true;
    resizer.setPointerCapture(event.pointerId);
  });
  resizer.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const width = Math.min(Math.max(window.innerWidth - event.clientX, 340), window.innerWidth * 0.65);
    app.style.setProperty("--panel-w", `${Math.round(width)}px`);
    viewer.resize();
    scene.requestRender();
    drawCharts();
  });
  resizer.addEventListener("pointerup", () => {
    dragging = false;
    try {
      localStorage.setItem("aqua-panel-width", String(parseInt(getComputedStyle(app).getPropertyValue("--panel-w"), 10)));
    } catch (error) {
      /* ignore */
    }
  });
})();

// ================================================================== tracks (chunked GPU polylines)
// A track is split into fixed chunks of CHUNK points. Only chunks touched by an update are
// re-uploaded: appending a point rewrites one small chunk instead of the whole line. The
// newest segment is a separate "tip" that ends at the (interpolated) marker position.
const CHUNK = 128;

class ChunkedTrack {
  constructor(width, material) {
    this.width = width;
    this.material = material;
    this.points = [];
    this.chunks = [];
    this.visible = true;
    this.tip = gpu.lines.add({ positions: [], width, material, show: false });
  }

  apply(reset, keep, xyz) {
    if (reset) this.points = [];
    else this.points.length = Math.min(keep, this.points.length);
    const first = this.points.length;
    for (let i = 0; i < xyz.length; i += 3) this.points.push(new Cesium.Cartesian3(xyz[i], xyz[i + 1], xyz[i + 2]));
    this.rebuild(reset ? 0 : first);
  }

  rebuild(fromIndex) {
    const body = Math.max(this.points.length - 1, 0); // the last point is drawn by the tip
    const needed = body >= 2 ? Math.ceil((body - 1) / CHUNK) : 0;
    const k0 = Math.floor(Math.max(fromIndex - 2, 0) / CHUNK);
    for (let k = k0; k < needed; k += 1) {
      const positions = this.points.slice(k * CHUNK, Math.min((k + 1) * CHUNK + 1, body));
      let line = this.chunks[k];
      if (!line) {
        line = gpu.lines.add({ positions, width: this.width, material: this.material });
        this.chunks[k] = line;
      } else {
        line.positions = positions;
      }
      line.show = this.visible && positions.length >= 2;
    }
    while (this.chunks.length > needed) gpu.lines.remove(this.chunks.pop());
    this.setTip(null);
    telemetry.vertexUploads += this.points.length - Math.min(k0 * CHUNK, this.points.length);
  }

  setTip(end) {
    const n = this.points.length;
    if (n < 1) {
      this.tip.show = false;
      return;
    }
    const start = n >= 2 ? this.points[n - 2] : this.points[0];
    this.tip.positions = [start, end || this.points[n - 1]];
    this.tip.show = this.visible && (n >= 2 || Boolean(end));
  }

  setVisible(visible) {
    if (visible === this.visible) return;
    this.visible = visible;
    for (const line of this.chunks) line.show = visible && line.positions.length >= 2;
    this.tip.show = visible && this.points.length >= 2;
  }

  dispose() {
    for (const line of this.chunks) gpu.lines.remove(line);
    gpu.lines.remove(this.tip);
    this.chunks = [];
  }
}

const tracks = new Map(); // key -> ChunkedTrack

function trackFor(key) {
  let track = tracks.get(key);
  if (!track) {
    if (key === "truth") track = new ChunkedTrack(2, colorMaterial(COLORS.truth.withAlpha(0.6)));
    else if (key === "est:ONLINE") track = new ChunkedTrack(2, dashMaterial(COLORS.online.withAlpha(0.9), 12));
    else if (key === "est:SMOOTHED") track = new ChunkedTrack(3, colorMaterial(COLORS.smoothed.withAlpha(0.85)));
    else track = new ChunkedTrack(1, colorMaterial(COLORS.observer.withAlpha(0.45)));
    tracks.set(key, track);
  }
  return track;
}

function applyTrackUpdates(updates) {
  for (const update of updates) {
    if (update.key === "*") {
      for (const track of tracks.values()) track.dispose();
      tracks.clear();
      continue;
    }
    trackFor(update.key).apply(update.reset, update.keep, update.xyz);
  }
}

// ================================================================== markers (smooth motion)
// Data arrive at 1 Hz. With smooth display on, each marker glides from where it is drawn to
// the new position over one update interval, rendered at up to TARGET_FPS; the track tip
// follows the marker so line and marker stay joined.
const TARGET_FPS = 30;
viewer.targetFrameRate = TARGET_FPS;
const anim = { enabled: true, active: new Set(), interval: 1000, lastArrival: 0, slowSince: null };

class Marker {
  constructor(point, label, trackKey) {
    this.point = point;
    this.label = label;
    this.trackKey = trackKey;
    this.pos = null;
    this.from = new Cesium.Cartesian3();
    this.to = new Cesium.Cartesian3();
    this.t0 = 0;
  }

  set(position) {
    if (!this.pos || !anim.enabled) {
      this.pos = Cesium.Cartesian3.clone(position, this.pos || new Cesium.Cartesian3());
      Cesium.Cartesian3.clone(position, this.to);
      anim.active.delete(this);
      this.draw();
      return;
    }
    Cesium.Cartesian3.clone(this.pos, this.from);
    Cesium.Cartesian3.clone(position, this.to);
    this.t0 = performance.now();
    anim.active.add(this);
  }

  step(now) {
    const a = Math.min(1, (now - this.t0) / anim.interval);
    Cesium.Cartesian3.lerp(this.from, this.to, a, this.pos);
    this.draw();
    return a < 1;
  }

  finish() {
    if (this.pos) Cesium.Cartesian3.clone(this.to, this.pos);
    this.draw();
  }

  draw() {
    if (!this.pos) return;
    this.point.position = this.pos;
    if (this.label) this.label.position = this.pos;
    if (this.trackKey && tracks.has(this.trackKey)) tracks.get(this.trackKey).setTip(anim.enabled ? this.pos : null);
  }

  show(visible) {
    this.point.show = visible;
    if (this.label) this.label.show = visible;
  }
}

function setSmooth(enabled, reason = "") {
  anim.enabled = enabled;
  if ($("smooth-motion")) $("smooth-motion").checked = enabled;
  if (!enabled) {
    for (const marker of anim.active) marker.finish();
    anim.active.clear();
    for (const track of tracks.values()) track.setTip(null);
  }
  if (reason) setMessage(reason);
  scene.requestRender();
}

scene.preUpdate.addEventListener(() => {
  // runs before the camera is updated: markers move first, then the camera follows them
  if (anim.active.size) {
    const now = performance.now();
    for (const marker of anim.active) if (!marker.step(now)) anim.active.delete(marker);
    updateErrorLine();
  }
  followTick();
});

// ================================================================== truth
function updateTruth(target) {
  if (!target) return;
  if (!state.truth) {
    state.truth = new Marker(
      gpu.points.add({ pixelSize: 10, color: COLORS.truth, outlineColor: Cesium.Color.BLACK, outlineWidth: 2, id: "truth" }),
      gpu.labels.add({ text: "TRUTH", font: "12px sans-serif", fillColor: COLORS.truth, pixelOffset: new Cesium.Cartesian2(0, -18) }),
      "truth",
    );
  }
  const show = checked("show-truth");
  state.truth.show(show);
  if (tracks.has("truth")) tracks.get("truth").setVisible(show);
  state.truth.set(cartOf(target.position));
}

// ================================================================== observers & bearings
function updateObservers(records, batch, config) {
  const detecting = new Set((batch?.observations || []).filter((o) => o.detected).map((o) => o.observer_id));
  const active = new Set();
  const r = config.max_slant_range_yd * YD_TO_M;
  for (const record of records) {
    const observer = record.state;
    const id = observer.observer_id;
    active.add(id);
    let item = state.observers.get(id);
    if (!item) {
      item = {
        marker: new Marker(
          gpu.points.add({ pixelSize: 8, outlineColor: Cesium.Color.BLACK, outlineWidth: 1, id: `observer-${id}` }),
          gpu.labels.add({ text: id, font: "11px sans-serif", fillColor: COLORS.observer, pixelOffset: new Cesium.Cartesian2(0, -14), scale: 0.9 }),
          `obs:${id}`,
        ),
        range: null,
        detecting: null,
      };
      state.observers.set(id, item);
    }
    const position = cartOf(observer.position);
    item.marker.set(position);
    const isDetecting = detecting.has(id);
    if (item.detecting !== isDetecting) {
      item.marker.point.color = isDetecting ? COLORS.detecting : COLORS.observer;
      item.detecting = isDetecting;
    }
    if (checked("show-range")) {
      if (!item.range) {
        item.range = viewer.entities.add({
          ellipsoid: {
            radii: new Cesium.Cartesian3(r, r, r * exaggeration()),
            material: COLORS.observer.withAlpha(0.05),
            outline: true,
            outlineColor: COLORS.observer.withAlpha(0.25),
            slicePartitions: 12,
            stackPartitions: 8,
          },
        });
      }
      item.range.position = position;
      item.range.ellipsoid.radii = new Cesium.Cartesian3(r, r, r * exaggeration());
      item.range.show = true;
    } else if (item.range) {
      item.range.show = false;
    }
  }
  for (const [id, item] of state.observers) {
    if (active.has(id) || item.detecting === "inactive") continue;
    item.marker.point.color = COLORS.inactive; // evicted / expired: drift history stays visible
    item.detecting = "inactive";
    if (item.range) item.range.show = false;
  }
  setText("observer-count", String(records.length));
  setText("detecting-count", String(detecting.size));
}

// Solid while the observer holds contact and its bearing is current; once contact is lost the
// last bearing stays as a dashed line until the observer is evicted or the run is reset.
function updateBearings(bearings, batch, observers, config) {
  const show = checked("show-bearing");
  const detecting = new Set((batch?.observations || []).filter((o) => o.detected).map((o) => o.observer_id));
  const active = new Set(observers.map((r) => r.state.observer_id));
  const fresh = new Set();
  const length = config.max_slant_range_yd * YD_TO_M;
  for (const item of bearings) {
    fresh.add(item.observer_id);
    let line = state.bearingLines.get(item.observer_id);
    if (!line) {
      line = { polyline: gpu.lines.add({ positions: [], width: 1.5 }), key: "", solid: null };
      state.bearingLines.set(item.observer_id, line);
    }
    const key = `${item.tick}-${exaggeration()}`;
    if (line.key !== key) { // a bearing changes only every bearing interval
      const end = destination(item.observer_position, item.bearing_deg, length);
      line.polyline.positions = [cartOf(item.observer_position), cart(end.longitude, end.latitude, item.observer_position.depth_ft)];
      line.key = key;
    }
  }
  for (const [id, line] of state.bearingLines) {
    const solid = fresh.has(id) && detecting.has(id);
    if (line.solid !== solid) {
      line.polyline.material = solid ? colorMaterial(COLORS.bearing.withAlpha(0.85)) : dashMaterial(COLORS.bearing.withAlpha(0.85), 8);
      line.solid = solid;
    }
    line.polyline.show = show && active.has(id);
  }
}

function clearBearings() {
  for (const line of state.bearingLines.values()) gpu.lines.remove(line.polyline);
  state.bearingLines.clear();
}

// ================================================================== forward deployment
function updateDeployment(deployment) {
  const status = deployment || { history: [], standby_count: 0, pending_placements: 0 };
  // a planned point is shown until its observer is in the water: with the layer while its drop
  // task is open (proposed / approved; laid, replaced, cancelled... -> hidden), without the layer
  // (no task ids) only the latest plan while its observers are still being started
  const openIds = new Set((status.tasks || []).filter((t) => t.status === "PROPOSED" || t.status === "APPROVED").map((t) => t.task_id));
  const latest = status.history[status.history.length - 1];
  const planned = (record) => (record.task_ids?.length
    ? record.positions.filter((_, k) => openIds.has(record.task_ids[k]))
    : record === latest && status.pending_placements > 0 ? record.positions : []);
  const key = `${status.history.length}-${status.last_deploy_tick}-${[...openIds].join(",")}-${status.pending_placements}-${checked("show-drops")}-${exaggeration()}`;
  if (key !== state.dropKey) {
    state.dropKey = key;
    gpu.drops.removeAll();
    gpu.dropLabels.removeAll();
    if (checked("show-drops")) {
      for (const record of status.history) {
        const positions = planned(record);
        for (const position of positions) {
          gpu.drops.add({ position: cartOf(position), pixelSize: 9, color: COLORS.drop.withAlpha(0.25), outlineColor: COLORS.drop, outlineWidth: 2 });
        }
        if (positions.length) {
          gpu.dropLabels.add({
            position: cartOf(positions[0]),
            text: `前程 ${clockAt(record.tick)}`,
            font: "11px sans-serif",
            fillColor: COLORS.drop,
            pixelOffset: new Cesium.Cartesian2(0, -14),
          });
        }
      }
    }
  }
  if (!tabVisible("tab-deploy")) return;
  const enabled = state.latestConfig?.forward?.enabled;
  const starting = status.pending_placements > 0 ? "（観測者コンテナを起動中）" : "";
  setHtml($("deploy-status"), `自動前程配置 <b>${enabled ? "有効" : "無効"}</b>　投入待ち ${status.pending_placements}${starting}　` +
    `待機中 ${status.standby_count}　最終配置 ${status.last_deploy_tick == null ? "--" : clockAt(status.last_deploy_tick)}`);
  const rows = status.history.slice().reverse().map((r) => {
    const depths = r.positions.map((p) => Math.round(p.depth_ft)).join("/");
    return `<tr><td>${clockAt(r.tick)}</td><td>${r.positions.length}<br /><small>${depths} Ft</small></td><td>${escapeHtml(deployReason(r.reason))}</td></tr>`;
  });
  setHtml($("deploy-table").querySelector("tbody"), rows.join("") || "<tr><td colspan='3'>配置なし</td></tr>");
}

// ================================================================== layer (設標者) and drop tasks
const TASK_STATUS = { PROPOSED: "了承待ち", APPROVED: "設標に向かう", DONE: "投入済み", REJECTED: "却下", EXPIRED: "失効", CANCELLED: "取消", REPLACED: "再計画で置換" };

function circlePositions(center, radiusM, n = 72) {
  const out = [];
  for (let k = 0; k <= n; k += 1) {
    const a = (2 * Math.PI * k) / n;
    const p = destination(center, (a * 180) / Math.PI, radiusM);
    out.push(Cesium.Cartesian3.fromDegrees(p.longitude, p.latitude, 0));
  }
  return out;
}

// order the layer flies the drops in (設標順): planned time (as soon as possible first), then the
// operator's drop order -- the same key as the backend (DropTask.flight_key)
const flightOrder = (a, b) => (a.planned_tick ?? -1) - (b.planned_tick ?? -1)
  || (a.sequence || a.task_id) - (b.sequence || b.task_id) || a.task_id - b.task_id;

function updateLayer(deployment) {
  const status = deployment || {};
  const layer = status.layer;
  const tasks = status.tasks || [];
  const open = tasks.filter((t) => t.status === "PROPOSED" || t.status === "APPROVED").sort(flightOrder);
  state.openDropOrder = open.map((t) => t.task_id);
  // ---- map: layer marker, route to the drop points, orbit circle, task points
  if (layer) {
    if (!state.layerMarker) {
      state.layerMarker = new Marker(
        gpu.points.add({ pixelSize: 11, color: COLORS.layer, outlineColor: Cesium.Color.BLACK, outlineWidth: 2, id: "layer" }),
        gpu.labels.add({ text: "設標者", font: "12px sans-serif", fillColor: COLORS.layer, pixelOffset: new Cesium.Cartesian2(0, -18) }),
        null,
      );
      state.layerRoute = gpu.layerLines.add({ width: 2, material: dashMaterial(COLORS.layer, 10), positions: [] });
      state.layerOrbit = gpu.layerLines.add({ width: 1, material: dashMaterial(COLORS.layer.withAlpha(0.45), 16), positions: [] });
    }
    state.layerMarker.show(true);
    state.layerMarker.set(Cesium.Cartesian3.fromDegrees(layer.position.longitude, layer.position.latitude, 0));
    const doing = layer.mode === "TRANSIT" ? `設標へ #${layer.task_id} 到着 ${layer.eta_s != null ? clockAt((layer.tick || 0) + layer.eta_s) : "--"}`
      : layer.mode === "HOLD" ? `#${layer.task_id} 設標点で計画時刻待ち` : "旋回待機";
    state.layerMarker.label.text = `設標者 ${fmt(layer.altitude_ft, 0)} ft ${fmt(layer.speed_kt, 0)} kt ${doing}`;
    // planned flight path (飛行予定経路, dashed): the turn-limited path the layer will fly through
    // the drop points, from the backend (straight legs from an older backend without it)
    const approved = open.filter((t) => t.status === "APPROVED");
    const planned = layer.planned_path || [];
    const route = planned.length > 1
      ? planned.map(([lat, lon]) => Cesium.Cartesian3.fromDegrees(lon, lat, 0))
      : approved.length && !("planned_path" in layer)
        ? [Cesium.Cartesian3.fromDegrees(layer.position.longitude, layer.position.latitude, 0),
          ...approved.map((t) => Cesium.Cartesian3.fromDegrees(t.position.longitude, t.position.latitude, 0))]
        : [];
    state.layerRoute.positions = route;
    state.layerRoute.show = route.length > 1;
    const orbitKey = layer.mode === "ORBIT" && layer.orbit_center
      ? `${layer.orbit_center.latitude.toFixed(4)},${layer.orbit_center.longitude.toFixed(4)},${Math.round(layer.orbit_radius_yd / 100)}` : "";
    if (orbitKey !== state.layerOrbitKey) {
      state.layerOrbitKey = orbitKey;
      state.layerOrbit.positions = orbitKey ? circlePositions(layer.orbit_center, layer.orbit_radius_yd * YD_TO_M) : [];
      state.layerOrbit.show = Boolean(orbitKey);
    }
  } else if (state.layerMarker) {
    state.layerMarker.show(false);
    state.layerRoute.show = false;
    state.layerOrbit.show = false;
  }
  const releaseKey = (p) => (p ? `${p.latitude.toFixed(4)}:${p.longitude.toFixed(4)}` : "");
  // the latest released drops: release point -> entry point (where the observer fell in the wind)
  const released = tasks.filter((t) => t.status === "DONE" && t.release_position).slice(-6);
  const taskKey = open.map((t) => `${t.task_id}:${t.status}:${t.planned_tick}:${t.sequence}:${t.cancel_suggestion ? 1 : 0}:${t.position.latitude.toFixed(4)}:${t.position.longitude.toFixed(4)}:${releaseKey(t.release_position)}`).join("|")
    + `|${released.map((t) => `${t.task_id}:${t.splash_tick}:${t.miss_yd}`).join(",")}|${exaggeration()}`;
  if (taskKey !== state.taskKey) {
    state.taskKey = taskKey;
    gpu.taskPoints.removeAll();
    gpu.taskLabels.removeAll();
    gpu.releaseLines.removeAll();
    const surface = (p) => Cesium.Cartesian3.fromDegrees(p.longitude, p.latitude, 0);
    for (const t of open.filter((x) => x.release_position)) {
      // corrected release point (投下点) and its predicted fall to the drop point
      gpu.taskPoints.add({ position: surface(t.release_position), pixelSize: 6, color: COLORS.layer, outlineColor: COLORS.drop, outlineWidth: 1 });
      gpu.releaseLines.add({ width: 1, material: dashMaterial(COLORS.drop.withAlpha(0.8), 6), positions: [surface(t.release_position), surface(t.position)] });
    }
    for (const t of released) {
      const target = t.planned_position || t.position;
      gpu.releaseLines.add({ width: 1, material: colorMaterial(COLORS.drop.withAlpha(0.6)), positions: [surface(t.release_position), surface(t.position)] });
      gpu.taskPoints.add({ position: surface(target), pixelSize: 5, color: Cesium.Color.TRANSPARENT, outlineColor: COLORS.proposed, outlineWidth: 1 });
      if (t.miss_yd != null) {
        gpu.taskLabels.add({ position: surface(t.position), text: `#${t.task_id} 着水誤差 ${Math.round(t.miss_yd)} YD`, font: "10px sans-serif",
          fillColor: COLORS.drop, pixelOffset: new Cesium.Cartesian2(0, 14) });
      }
    }
    for (const [index, t] of open.entries()) {
      const proposed = t.status === "PROPOSED";
      const color = proposed ? COLORS.proposed : t.cancel_suggestion ? COLORS.cancel : COLORS.drop;
      gpu.taskPoints.add({ position: cartOf(t.position), pixelSize: 10, color: color.withAlpha(proposed ? 0.15 : 0.6), outlineColor: color, outlineWidth: 2 });
      gpu.taskLabels.add({
        position: cartOf(t.position),
        text: `${open.length > 1 ? `${index + 1}番目 ` : ""}${proposed ? "提案 " : ""}#${t.task_id}（${Math.round(t.position.depth_ft)} Ft）`
          + (t.planned_tick != null ? `計画 ${clockAt(t.planned_tick)}` : "すぐ") + (proposed ? " 了承待ち" : "")
          + (t.cancel_suggestion ? " 中止を提案" : ""),
        font: "11px sans-serif", fillColor: color, pixelOffset: new Cesium.Cartesian2(0, -14),
      });
    }
  }
  // ---- approval alert (always visible) and the panel
  const proposed = open.filter((t) => t.status === "PROPOSED");
  $("drop-alert").hidden = proposed.length === 0;
  setText("drop-alert-count", proposed.length);
  // drops the replanner proposes to cancel (no longer help detection); the layer keeps flying them
  // shown on the map until cancelled or the operator chooses to keep it (継続)
  const suggested = open.filter((t) => t.status === "APPROVED" && t.cancel_suggestion
    && !state.cancelKept.has(cancelKey(t)));
  state.cancelSuggested = suggested.map((t) => t.task_id);
  $("drop-cancel-alert").hidden = suggested.length === 0;
  setText("drop-cancel-count", suggested.length);
  const alertKey = suggested.map((t) => `${cancelKey(t)}:${t.planned_tick}:${t.cancel_suggestion}`).join("|");
  if (alertKey !== state.cancelAlertKey) {
    state.cancelAlertKey = alertKey;
    state.cancelTasks = new Map(suggested.map((t) => [t.task_id, t]));
    setHtml($("drop-cancel-list"), suggested.map((t) => `<li><span class="what">#${t.task_id}・`
      + `${t.planned_tick != null ? `計画 ${clockAt(t.planned_tick)}` : "すぐ"}・${Math.round(t.position.depth_ft)} Ft</span>`
      + `<span class="why">${escapeHtml(t.cancel_suggestion)}</span>`
      + `<button type="button" class="mini" data-cancel-one="${t.task_id}">中止</button>`
      + `<button type="button" class="mini ghost" data-cancel-keep="${t.task_id}" title="中止せずに設標を続ける（この提案を閉じる）">継続</button>`
      + `<button type="button" class="mini ghost" data-cancel-focus="${t.task_id}">地図で表示</button></li>`).join(""));
  }
  if (!state.dropApprovalPending && $("drop-approval").value !== (status.approval || "auto")) $("drop-approval").value = status.approval || "auto";
  if (!tabVisible("tab-deploy")) return;
  const lay = state.latestConfig?.layer || {};
  const enabled = lay.enabled !== false;
  const paused = Boolean(lay.paused);
  $("layer-status").textContent = !enabled ? "設標者なし（追加の観測者は即時に投入）"
    : layer ? `設標者 ${fmt(layer.speed_kt, 0)} kt・バンク ${fmt(Math.abs(layer.bank_deg), 1)}°（${turnName(layer.bank_deg)}）・${layer.mode === "TRANSIT" ? `設標 #${layer.task_id} へ移動中（到着 ${layer.eta_s != null ? clockAt((layer.tick || 0) + layer.eta_s) : "--"}）` : layer.mode === "HOLD" ? `設標 #${layer.task_id} の地点で計画時刻まで旋回` : "目標推定位置の周囲を旋回待機"}${paused ? "【設標一時停止中】" : ""}　了承待ち ${proposed.length}・設標待ち ${open.length - proposed.length}`
      : "設標者の準備中";
  const next = open.find((t) => t.status === "APPROVED");
  const detail = layer && enabled ? [
    ["状態", { ORBIT: "旋回待機", TRANSIT: "設標へ移動", HOLD: "設標点で時刻待ち" }[layer.mode] || layer.mode],
    ["位置", `${fmt(layer.position.latitude, 4)}, ${fmt(layer.position.longitude, 4)}`],
    ["高度", `${fmt(layer.altitude_ft, 0)} ft（巡航 ${fmt(lay.cruise_altitude_ft, 0)} ft・投下 ${fmt(lay.drop_altitude_ft, 0)} ft）`],
    ["速力・針路", `${fmt(layer.speed_kt, 0)} kt・${fmt(layer.heading_deg, 0)}°`],
    ["対地速力・航跡", layer.ground_speed_kt != null ? `${fmt(layer.ground_speed_kt, 0)} kt・${fmt(layer.track_deg, 0)}°` : "--"],
    ["飛行高度の風", layer.wind_speed_kt != null ? `${fmt(layer.wind_direction_deg, 0)}°・${fmt(layer.wind_speed_kt, 1)} kt` : "--"],
    ["投下修正の風", correctionWind(layer, (status.wind_estimates || []).slice(-1)[0], lay.wind_correction !== false).text],
    ["バンク", `${fmt(Math.abs(layer.bank_deg), 1)}°（${turnName(layer.bank_deg)}）`],
    ["基準旋回", `${lay.preferred_turn === "right" ? "右" : "左"}旋回（反対旋回は ${fmt(lay.turn_margin_s ?? 10, 0)} 秒以上早い場合）`],
    ["実施中", layer.task_id != null ? `#${layer.task_id}・到着 ${layer.eta_s != null ? `${clockAt((layer.tick || 0) + layer.eta_s)}（あと ${fmt(layer.eta_s, 0)} s）` : "--"}` : "なし"],
    ["次の設標", next ? `#${next.task_id}・${next.planned_tick != null ? `計画 ${clockAt(next.planned_tick)}（あと ${Math.max(0, next.planned_tick - (layer.tick || 0))} s）` : "すぐ"}` : "なし"],
    ["設標", paused ? "一時停止中" : "実施"],
  ] : [];
  setHtml($("layer-detail").querySelector("tbody") || $("layer-detail"),
    detail.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("") || "<tr><td>--</td></tr>");
  // open drops first in the drop order (設標順, ▲▼ to change it), then the latest closed ones
  const closed = tasks.filter((t) => !open.includes(t)).reverse();
  const rows = [...open, ...closed].slice(0, 15).map((t) => {
    const index = open.indexOf(t);
    const move = index >= 0 && open.length > 1
      ? `<button type="button" class="mini ghost" data-up="${t.task_id}" title="設標順を前へ"${index === 0 ? " disabled" : ""}>▲</button>`
        + `<button type="button" class="mini ghost" data-down="${t.task_id}" title="設標順を後へ"${index === open.length - 1 ? " disabled" : ""}>▼</button>` : "";
    const actions = move + (t.status === "PROPOSED"
      ? `<button type="button" data-approve="${t.task_id}">了承</button><button type="button" class="ghost" data-reject="${t.task_id}">却下</button>`
      : t.status === "APPROVED"
        ? `<button type="button" class="mini" data-now="${t.task_id}">今すぐ</button>`
          + `<input class="time" type="text" inputmode="numeric" placeholder="HH:MM:SS" title="投入時刻（時:分:秒）" data-time-input="${t.task_id}" />`
          + `<button type="button" class="mini" data-time="${t.task_id}">時刻</button>`
          + `<button type="button" class="mini ghost" data-cancel="${t.task_id}">中止</button>` : "");
    const plan = t.planned_tick != null ? `計画 ${clockAt(t.planned_tick)}` : "すぐ";
    const late = t.status === "DONE" && t.planned_tick != null ? `（${t.done_tick - t.planned_tick >= 0 ? "+" : ""}${t.done_tick - t.planned_tick} s）` : "";
    const fall = t.status === "DONE" && t.splash_tick != null
      ? (t.miss_yd != null ? `・着水 ${clockAt(t.splash_tick)}（誤差 ${Math.round(t.miss_yd)} YD）` : "・落下中") : "";
    const when = t.status === "APPROVED" ? `${plan}・あと ${fmt(t.eta_s, 0)} s` : t.status === "DONE" ? `${clockAt(t.done_tick)} 投下${late}${fall}` : plan;
    const src = { forward: "自動", operator: "即時配置", manual: "手動配置" }[t.source] || t.source;
    const rank = index >= 0 && open.length > 1 ? `・${index + 1}番目` : "";
    const suggest = t.status === "APPROVED" && t.cancel_suggestion
      ? `<br /><small class="cancel-note">中止を提案：${escapeHtml(t.cancel_suggestion)}</small>` : "";
    return `<tr${suggest ? ' class="cancel-suggested"' : ""}><td>${t.task_id}<br /><small>${src}${rank}</small></td><td>${TASK_STATUS[t.status] || t.status}${suggest}</td>`
      + `<td>${Math.round(t.position.depth_ft)} Ft<br /><small>${when}</small></td><td>${actions}</td></tr>`;
  });
  // keep a drop time the operator is typing: do not rebuild the table under the cursor
  const editing = document.activeElement?.closest?.("#drop-table") && document.activeElement.matches("input");
  if (!editing) setHtml($("drop-table").querySelector("tbody"), rows.join("") || "<tr><td colspan='4'>設標計画なし</td></tr>");
}

async function decideDrops(taskIds, approve) {
  const response = await fetch(`/api/drops/${approve ? "approve" : "reject"}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ task_ids: taskIds }),
  });
  if (!response.ok) {
    setMessage(`設標計画の${approve ? "了承" : "却下"}に失敗: ${await response.text()}`);
    return;
  }
  const changed = await response.json();
  setMessage(`設標計画 ${changed.length} 件を${approve ? "了承しました（設標者が向かいます）" : "却下しました"}。`);
}

async function postDrops(path, body, done) {
  const response = await fetch(`/api/drops/${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!response.ok) {
    setMessage(`設標計画の変更に失敗: ${await response.text()}`);
    return;
  }
  setMessage(done(await response.json()));
}

const cancelDrops = (taskIds) => postDrops("cancel", { task_ids: taskIds }, (changed) => `設標計画 ${changed.length} 件を中止しました。`);
const rescheduleDrop = (taskId, plannedTick) => postDrops("reschedule", { task_id: taskId, planned_tick: plannedTick },
  (task) => (task.planned_tick == null ? `設標 #${task.task_id} はすぐに向かいます。` : `設標 #${task.task_id} の投入時刻を ${clockAt(task.planned_tick)} にしました。`));

// move an open drop one place earlier (-1) or later (+1) in the drop order; the backend plans
// the drop times again along the new order
function moveDrop(taskId, offset) {
  const order = (state.openDropOrder || []).slice();
  const from = order.indexOf(taskId);
  const to = from + offset;
  if (from < 0 || to < 0 || to >= order.length) return;
  [order[from], order[to]] = [order[to], order[from]];
  postDrops("reorder", { task_ids: order }, (tasks) => `設標順を変更しました：${tasks.map((t) => `#${t.task_id}`).join(" → ")}（投入時刻を再計算）`);
}

$("drop-table").addEventListener("click", (event) => {
  const target = (key) => event.target.closest(`[data-${key}]`);
  const id = (el, key) => Number(el.dataset[key]);
  let el;
  if ((el = target("approve"))) decideDrops([id(el, "approve")], true);
  if ((el = target("reject"))) decideDrops([id(el, "reject")], false);
  if ((el = target("now"))) rescheduleDrop(id(el, "now"), null);
  if ((el = target("cancel"))) cancelDrops([id(el, "cancel")]);
  if ((el = target("up"))) moveDrop(id(el, "up"), -1);
  if ((el = target("down"))) moveDrop(id(el, "down"), 1);
  if ((el = target("time"))) {
    const input = $("drop-table").querySelector(`[data-time-input="${el.dataset.time}"]`);
    const tick = input ? tickFromClock(input.value) : null;
    if (tick == null) { setMessage("投入時刻を 時:分:秒（例 14:05:30）で入力してください。"); return; }
    rescheduleDrop(id(el, "time"), tick);
    input.blur();
  }
});
$("drop-approve-all").addEventListener("click", () => decideDrops(null, true));
$("drop-cancel-suggested").addEventListener("click", () => {
  if (state.cancelSuggested?.length) cancelDrops(state.cancelSuggested);
});
$("drop-cancel-list").addEventListener("click", (event) => {
  const el = event.target.closest("[data-cancel-one], [data-cancel-keep], [data-cancel-focus]");
  if (!el) return;
  const id = Number(el.dataset.cancelOne ?? el.dataset.cancelKeep ?? el.dataset.cancelFocus);
  const task = state.cancelTasks?.get(id);
  if (el.dataset.cancelOne != null) cancelDrops([id]);
  else if (el.dataset.cancelKeep != null && task) {
    state.cancelKept.add(cancelKey(task));
    state.cancelSuggested = state.cancelSuggested.filter((x) => x !== id);
    el.closest("li")?.remove();
    $("drop-cancel-alert").hidden = state.cancelSuggested.length === 0;
    setText("drop-cancel-count", state.cancelSuggested.length);
    setMessage(`設標 #${id} は中止せずに続けます。`);
  } else if (task) {
    viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
    viewer.camera.flyToBoundingSphere(new Cesium.BoundingSphere(cartOf(task.position), 2000), flightOptions({
      offset: new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-60), 9000), duration: 1.0,
    }));
  }
});
$("drop-approve-all-tab").addEventListener("click", () => decideDrops(null, true));
$("drop-cancel-all").addEventListener("click", () => cancelDrops(null));
$("layer-enabled").addEventListener("change", () => {
  const on = checked("layer-enabled");
  putConfig((next) => { next.layer.enabled = on; })
    .then(() => setMessage(on ? "設標者が移動して観測者を投入します。" : "設標者なし：追加の観測者は即時に投入します。"))
    .catch((error) => setMessage(`設定エラー: ${error.message}`));
});
$("layer-paused").addEventListener("change", () => {
  const paused = checked("layer-paused");
  putConfig((next) => { next.layer.paused = paused; })
    .then(() => setMessage(paused ? "設標を一時停止しました（目標推定位置の周囲で旋回待機）。" : "設標を再開しました。"))
    .catch((error) => setMessage(`設定エラー: ${error.message}`));
});
$("layer-turn").addEventListener("change", () => {
  const turn = $("layer-turn").value;
  putConfig((next) => { next.layer.preferred_turn = turn; })
    .then(() => setMessage(`設標者の基準旋回を${turn === "right" ? "右" : "左"}旋回にしました。`))
    .catch((error) => setMessage(`設定エラー: ${error.message}`));
});
$("layer-form").addEventListener("submit", (event) => {
  event.preventDefault();
  putConfig((next) => {
    next.layer.speed_kt = num("layer-speed");
    next.layer.speed_spread_kt = num("layer-spread");
    next.layer.max_bank_deg = num("layer-bank");
    next.layer.orbit_radius_yd = num("layer-orbit");
    next.layer.turn_margin_s = num("layer-turn-margin");
    next.layer.proposal_timeout_s = num("layer-timeout");
    next.layer.cruise_altitude_ft = num("layer-cruise-alt");
    next.layer.drop_altitude_ft = num("layer-drop-alt");
    next.layer.climb_rate_fpm = num("layer-climb");
  })
    .then(() => setMessage("設標者の条件を反映しました。"))
    .catch((error) => setMessage(`設定エラー: ${error.message}`));
});
$("layer-wind-correction").addEventListener("change", () => {
  const on = checked("layer-wind-correction");
  putConfig((next) => { next.layer.wind_correction = on; })
    .then(() => setMessage(on ? "推定した平均風で投下点を修正します。" : "投下点を風で修正しません（無風の自由落下で計算）。"))
    .catch((error) => setMessage(`設定エラー: ${error.message}`));
});
$("drop-reject-all").addEventListener("click", () => decideDrops(null, false));

// ================================================================== map: layer state, winds and the external force
const KT_TO_MPS = 1852 / 3600;
const WIND_ARROW_S = 60; // wind arrows: distance the air moves in 1 min
const CURRENT_ARROW_S = 1200; // current arrows: distance the water moves in 20 min
const turnName = (bank) => (Math.abs(bank) < 0.5 ? "直進" : bank < 0 ? "左旋回" : "右旋回");
const dirSpeedText = (dirFrom, speed, digits = 1) => `${pad3(dirFrom)}°・${fmt(speed, digits)} kt`;
// the wind the release points are corrected with (投下修正の風) and where it comes from
// (LayerState.correction_source): the mean wind estimated from the last drop, else the wind at
// the layer's current altitude, else none (correction off or no wind)
const CORRECTION_SOURCE_NAMES = { estimate: "推定風", flight_altitude: "現在高度の風", none: "なし" };
function correctionWind(layer, windEstimate, correction) {
  const source = layer?.correction_source || "none";
  const name = CORRECTION_SOURCE_NAMES[source] || source;
  if (source === "none" || layer?.correction_wind_speed_kt == null) {
    return { source: "none", name: CORRECTION_SOURCE_NAMES.none, arrow: null,
      text: `【なし】${correction ? "" : "修正オフ・"}無風で投下点を計算` };
  }
  const dir = layer.correction_wind_direction_deg, speed = layer.correction_wind_speed_kt;
  const rad = (dir + 180) * Math.PI / 180;
  const detail = source === "estimate" && windEstimate
    ? `#${windEstimate.task_id}・${fmt(windEstimate.altitude_ft, 0)} ft〜海面・${clockAt(windEstimate.tick)}`
    : `推定なし・${fmt(layer.altitude_ft, 0)} ft`;
  return { source, name, arrow: { east: speed * Math.sin(rad), north: speed * Math.cos(rad) },
    short: `【${name}】${dirSpeedText(dir, speed)}`, text: `【${name}】${dirSpeedText(dir, speed)}（${detail}）` };
}
function pad3(deg) {
  return deg == null || !Number.isFinite(Number(deg)) ? "---" : String(Math.round(((Number(deg) % 360) + 360) % 360) % 360).padStart(3, "0");
}
// direction the vector points to (towards), degrees true
const towards = (east, north) => (Math.atan2(east, north) * 180 / Math.PI + 360) % 360;

// one arrow (PolylineArrow) and its label, created once and moved every update
function vectorArrow(key, color) {
  state.vectors = state.vectors || {};
  if (!state.vectors[key]) {
    state.vectors[key] = {
      line: gpu.vectors.add({ width: 9, material: Cesium.Material.fromType("PolylineArrow", { color }), positions: [] }),
      label: gpu.vectorLabels.add({ text: "", font: "11px sans-serif", fillColor: color, outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2, style: Cesium.LabelStyle.FILL_AND_OUTLINE, pixelOffset: new Cesium.Cartesian2(8, 0),
        horizontalOrigin: Cesium.HorizontalOrigin.LEFT }),
    };
  }
  return state.vectors[key];
}
function setArrow(key, color, from, east, north, seconds, text) {
  const arrow = vectorArrow(key, color);
  const speed = Math.hypot(east, north);
  if (!from || !(speed > 0.01)) {
    arrow.line.show = false;
    arrow.label.show = false;
    return;
  }
  const length = speed * KT_TO_MPS * seconds;
  const tip = destination(from, towards(east, north), length);
  const start = Cesium.Cartesian3.fromDegrees(from.longitude, from.latitude, 0);
  const end = Cesium.Cartesian3.fromDegrees(tip.longitude, tip.latitude, 0);
  arrow.line.positions = [start, end];
  arrow.line.show = true;
  arrow.label.position = end;
  arrow.label.text = text;
  arrow.label.show = true;
}
function hideArrows() {
  for (const arrow of Object.values(state.vectors || {})) {
    arrow.line.show = false;
    arrow.label.show = false;
  }
}

// current estimated from the observers' drift: v(p) = a + G (p - p_ref), G in kt/NM
function currentAt(current, position) {
  const b = current.base_velocity;
  const ref = current.reference_position;
  if (!ref || !position) return { east: b.east_kt, north: b.north_kt };
  const d = offsetM(ref, position);
  const x = d.east / 1852;
  const y = d.north / 1852;
  const g = current.gradient_per_nm || [[0, 0], [0, 0]];
  return { east: b.east_kt + g[0][0] * x + g[0][1] * y, north: b.north_kt + g[1][0] * x + g[1][1] * y };
}

// winds and the external force: arrows on the map; the 設標者 panel (bottom left) and the
// 風・外力 panel (below the camera readout)
function updateForces(snapshot, estimate) {
  const deployment = snapshot.deployment || {};
  const layer = deployment.layer;
  const config = state.latestConfig || snapshot.config || {};
  const lay = config.layer || {};
  const layerOn = lay.enabled !== false && layer;
  const windEstimate = (deployment.wind_estimates || []).slice(-1)[0];
  const current = snapshot.current_estimate;
  const correction = lay.wind_correction !== false;
  const where = layerOn ? layer.position : null;
  if (checked("show-forces")) {
    // wind at the flight altitude (from -> towards) and the mean wind of the drop correction,
    // both drawn from the layer; the external force at the fit's reference point and the estimate
    const flightRad = ((layer?.wind_direction_deg ?? 0) + 180) * Math.PI / 180;
    const fw = layerOn && layer.wind_speed_kt != null
      ? { east: layer.wind_speed_kt * Math.sin(flightRad), north: layer.wind_speed_kt * Math.cos(flightRad) } : null;
    setArrow("flightWind", COLORS.flightWind, fw ? where : null, fw?.east || 0, fw?.north || 0, WIND_ARROW_S,
      fw ? `飛行高度の風 ${dirSpeedText(layer.wind_direction_deg, layer.wind_speed_kt)}` : "");
    const dw = layerOn ? correctionWind(layer, windEstimate, correction) : null;
    setArrow("dropWind", COLORS.dropWind, dw?.arrow ? where : null, dw?.arrow?.east || 0, dw?.arrow?.north || 0, WIND_ARROW_S,
      dw?.arrow ? `投下修正の風 ${dw.short}` : "");
    const refPos = current?.reference_position;
    const atRef = current ? currentAt(current, refPos) : null;
    setArrow("currentRef", COLORS.current, current ? refPos : null, atRef?.east || 0, atRef?.north || 0, CURRENT_ARROW_S,
      atRef ? `外力 流向 ${dirSpeedText(towards(atRef.east, atRef.north), Math.hypot(atRef.east, atRef.north), 2)}（観測者の移動量）` : "");
    const estPos = estimate?.current_position;
    const atEst = current && estPos ? currentAt(current, estPos) : null;
    setArrow("currentEst", COLORS.current, atEst ? estPos : null, atEst?.east || 0, atEst?.north || 0, CURRENT_ARROW_S,
      atEst ? `外力 ${fmt(Math.hypot(atEst.east, atEst.north), 2)} kt` : "");
  } else {
    hideArrows();
  }

  let rows = [];
  const row = (k, v, cls = "") => rows.push(`<tr${cls ? ` class="${cls}"` : ""}><th>${k}</th><td>${v}</td></tr>`);
  const fill = (id, show) => {
    const panel = $(id);
    if (!panel) return;
    panel.hidden = !show;
    if (show) setHtml(panel.querySelector("tbody"), rows.join(""));
    rows = [];
  };
  const showLayer = checked("show-layer-hud");
  const showForces = checked("show-forces");
  if (showLayer) {
    rows.push(`<tr class="head"><th colspan="2">設標者</th></tr>`);
    if (lay.enabled === false) row("状態", "設標者なし");
    else if (!layer) row("状態", "準備中");
    else {
      const tasks = deployment.tasks || [];
      const open = tasks.filter((t) => t.status === "PROPOSED" || t.status === "APPROVED").sort(flightOrder);
      const next = open.find((t) => t.status === "APPROVED" && t.task_id !== layer.task_id);
      const proposed = open.filter((t) => t.status === "PROPOSED").length;
      const mode = { ORBIT: "旋回待機", TRANSIT: `設標 #${layer.task_id} へ移動`, HOLD: `設標 #${layer.task_id} の地点で時刻待ち` }[layer.mode] || layer.mode;
      row("状態", `${mode}${lay.paused ? "（一時停止中）" : ""}`);
      row("位置", `${latText(layer.position.latitude)} ${lonText(layer.position.longitude)}`);
      row("高度", `${fmt(layer.altitude_ft, 0)} ft`);
      row("速力・針路", `${fmt(layer.speed_kt, 0)} kt・${pad3(layer.heading_deg)}°`);
      row("対地・航跡", layer.ground_speed_kt != null ? `${fmt(layer.ground_speed_kt, 0)} kt・${pad3(layer.track_deg)}°` : "--");
      row("バンク", `${fmt(Math.abs(layer.bank_deg), 1)}°（${turnName(layer.bank_deg)}）`);
      if (layer.task_id != null && layer.eta_s != null) row("到着", `${clockAt((layer.tick || 0) + layer.eta_s)}（あと ${fmt(layer.eta_s, 0)} s）`);
      if (next) row("次の設標", `#${next.task_id} ${next.planned_tick != null ? `計画 ${clockAt(next.planned_tick)}` : "すぐ"}`);
      row("設標待ち", `${open.length - proposed} 件${proposed ? `・了承待ち ${proposed} 件` : ""}`);
    }
  }
  fill("layer-info", showLayer);
  if (showForces) {
    rows.push(`<tr class="head"><th colspan="2">風・外力</th></tr>`);
    row(`<i class="sw flight-wind"></i>飛行高度の風`, layerOn && layer.wind_speed_kt != null
      ? `${dirSpeedText(layer.wind_direction_deg, layer.wind_speed_kt)}（${fmt(layer.altitude_ft, 0)} ft）` : "--");
    row(`<i class="sw drop-wind"></i>投下修正の風`, layerOn ? correctionWind(layer, windEstimate, correction).text : "--");
    const base = current ? currentAt(current, current.reference_position) : null;
    row(`<i class="sw current-force"></i>外力（潮流）`, base
      ? `流向 ${dirSpeedText(towards(base.east, base.north), Math.hypot(base.east, base.north), 2)}（観測者 ${current.observer_count}・${fmt(current.window_seconds / 60, 0)} 分、残差 ${fmt(current.residual_kt, 2)} kt）`
      : "推定なし");
    rows.push(`<tr class="note"><td colspan="2">風は吹いてくる方向、外力は流れる方向。矢印の長さ：風 1 分・外力 20 分の移動量</td></tr>`);
  }
  fill("force-info", showForces);
}

// ================================================================== wind (風向風速) and the mean wind estimates
const WIND_TOP_FT = 30000;
const WIND_STEP_FT = 1000;

function windRows(config) {
  // one row every 1,000 ft from the sea surface to 30,000 ft (missing levels: interpolated value)
  const levels = (config.wind?.levels || []).slice().sort((a, b) => a.altitude_ft - b.altitude_ft);
  const rows = [];
  for (let alt = 0; alt <= WIND_TOP_FT; alt += WIND_STEP_FT) {
    const exact = levels.find((l) => l.altitude_ft === alt);
    if (exact) { rows.push(exact); continue; }
    const below = levels.filter((l) => l.altitude_ft < alt).pop();
    const above = levels.find((l) => l.altitude_ft > alt);
    const pick = below && above ? (alt - below.altitude_ft <= above.altitude_ft - alt ? below : above) : below || above;
    rows.push({ altitude_ft: alt, direction_deg: pick ? pick.direction_deg : 0, speed_kt: pick ? pick.speed_kt : 0 });
  }
  return rows;
}

function fillWindTable(config) {
  const rows = windRows(config).reverse().map((l) => `<tr><td>${l.altitude_ft.toLocaleString()}</td>`
    + `<td><input type="number" min="0" max="360" step="1" data-wind-dir="${l.altitude_ft}" value="${l.direction_deg}" /></td>`
    + `<td><input type="number" min="0" max="300" step="1" data-wind-speed="${l.altitude_ft}" value="${l.speed_kt}" /></td></tr>`);
  setHtml($("wind-table").querySelector("tbody"), rows.join(""));
  $("wind-enabled").checked = config.wind?.enabled !== false;
  $("wind-terminal").value = config.wind?.terminal_velocity_fps ?? 100;
}

function setAllWind(direction, speed) {
  for (const input of $("wind-table").querySelectorAll("[data-wind-dir]")) input.value = direction;
  for (const input of $("wind-table").querySelectorAll("[data-wind-speed]")) input.value = speed;
}

$("wind-fill").addEventListener("click", () => setAllWind(num("wind-fill-dir") % 360, Math.max(0, num("wind-fill-speed"))));
$("wind-calm").addEventListener("click", () => setAllWind(0, 0));
$("wind-apply").addEventListener("click", () => {
  const levels = [];
  for (let alt = 0; alt <= WIND_TOP_FT; alt += WIND_STEP_FT) {
    const dir = Number($("wind-table").querySelector(`[data-wind-dir="${alt}"]`)?.value || 0);
    const speed = Number($("wind-table").querySelector(`[data-wind-speed="${alt}"]`)?.value || 0);
    levels.push({ altitude_ft: alt, direction_deg: ((dir % 360) + 360) % 360, speed_kt: Math.max(0, speed) });
  }
  putConfig((next) => {
    next.wind = { ...(next.wind || {}), enabled: checked("wind-enabled"), levels, terminal_velocity_fps: num("wind-terminal") };
  })
    .then(() => setMessage("風向風速を反映しました（海面〜30,000 ft）。"))
    .catch((error) => setMessage(`設定エラー: ${error.message}`));
});

function updateWind(deployment) {
  if (!tabVisible("tab-environment")) return;
  const estimates = (deployment?.wind_estimates || []).slice().reverse();
  const latest = estimates[0];
  const falling = deployment?.falling || 0;
  const correction = state.latestConfig?.layer?.wind_correction !== false;
  const applied = deployment?.layer ? correctionWind(deployment.layer, latest, correction) : null;
  $("wind-status").textContent = (latest
    ? `最新の推定（#${latest.task_id}、${fmt(latest.altitude_ft, 0)} ft〜海面）：${fmt(latest.direction_deg, 0)}°・${fmt(latest.speed_kt, 1)} kt`
      + `（真値 ${fmt(latest.true_direction_deg, 0)}°・${fmt(latest.true_speed_kt, 1)} kt）`
    : "まだ着水した観測者がありません（推定風がない間は現在高度の風で投下点を修正）")
    + `　投下修正の風 ${applied ? applied.text : "--"}`
    + (falling ? `　落下中 ${falling}` : "");
  const rows = estimates.map((w) => `<tr><td>${w.task_id}<br /><small>${clockAt(w.tick)}</small></td>`
    + `<td>${fmt(w.altitude_ft, 0)} ft<br /><small>落下 ${fmt(w.fall_time_s, 1)} s</small></td>`
    + `<td>${fmt(w.offset_yd, 0)} YD</td>`
    + `<td>${fmt(w.direction_deg, 0)}°・${fmt(w.speed_kt, 1)} kt</td>`
    + `<td>${fmt(w.true_direction_deg, 0)}°・${fmt(w.true_speed_kt, 1)} kt<br /><small>着水誤差 ${fmt(w.miss_yd, 0)} YD</small></td></tr>`);
  setHtml($("wind-estimate-table").querySelector("tbody"), rows.join("") || "<tr><td colspan='5'>推定なし</td></tr>");
}
$("drop-approval").addEventListener("change", () => {
  const mode = $("drop-approval").value;
  state.dropApprovalPending = true;
  putConfig((next) => { next.layer.approval = mode; })
    .then(() => setMessage(mode === "auto" ? "設標計画の了承を自動にしました。" : "設標計画は手動で了承します（了承待ちは画面下部に表示）。"))
    .catch((error) => setMessage(`設定エラー: ${error.message}`))
    .finally(() => { state.dropApprovalPending = false; });
});

function deployReason(reason) {
  // replanner: "replan: <why>; drops [..] replaced, predicted error e -> f YD; <optimal planner reason>"
  const r = /^replan: (.*?); drops \[([^\]]*)\] replaced, predicted error (\d+) -> (\d+) YD; (.*)$/.exec(reason || "");
  if (r) {
    const ids = r[2].split(/,\s*/).map((id) => `#${id}`).join("・");
    return `再計画（${replanWhy(r[1])}）：設標 ${ids} を差し替え、予測誤差 ${r[3]}→${r[4]} YD／${deployReason(r[5])}`;
  }
  // optimal planner: "optimal (coverage): 2 observers, depths [..] Ft; predicted error e -> f YD
  // (horizontal a -> b YD, depth c -> d Ft, coverage gaps g -> h %); drop in [..] s; replanned after a maneuver .."
  const m = /^optimal \(([^)]+)\): (\d+) observers, depths \[([^\]]*)\] Ft; predicted error (\d+) -> (\d+) YD \(horizontal (\d+) -> (\d+) YD, depth (\d+) -> (\d+) Ft, coverage gaps (\d+) -> (\d+) %\)(?:; drop in \[([^\]]*)\] s)?/.exec(reason || "");
  if (!m) return reason;
  const why = { coverage: "探知範囲の不足", "information gain": "追尾精度の改善", "operator request": "操作員の指示" }[m[1]] || m[1];
  const times = m[12] ? `、投入 ${m[12].split(/,\s*/).map((t) => `${t} s 後`).join("/")}` : "";
  const replan = /replanned after a maneuver/.test(reason) ? "（機動を検出して再計画）" : "";
  return `最適配置（${why}）${replan}：${m[2]} 本・深度 ${m[3].split(/,\s*/).join("/")} Ft、予測誤差 ${m[4]}→${m[5]} YD`
    + `（水平 ${m[6]}→${m[7]} YD・深度 ${m[8]}→${m[9]} Ft・探知不足 ${m[10]}→${m[11]} %）${times}`;
}

// why the replanner replaced drops (aqua_drift.replanning.estimate_change)
function replanWhy(why) {
  let m = /^maneuver detected at tick (\d+) after drop (\d+) was planned$/.exec(why);
  if (m) return `設標 #${m[2]} の計画後 ${clockAt(Number(m[1]))} に機動を検出`;
  m = /^heading (\d+) -> (\d+) deg since drop (\d+) was planned$/.exec(why);
  if (m) return `設標 #${m[3]} の計画時から推定針路 ${m[1]}°→${m[2]}°`;
  m = /^speed ([\d.]+) -> ([\d.]+) kt since drop (\d+) was planned$/.exec(why);
  if (m) return `設標 #${m[3]} の計画時から推定速力 ${m[1]}→${m[2]} kt`;
  m = /^estimate moved (\d+) YD \(> (\d+)\) since drop (\d+) was planned$/.exec(why);
  if (m) return `設標 #${m[3]} の計画時から推定位置が ${m[1]} YD ずれた・閾値 ${m[2]} YD`;
  return why;
}

$("deploy-now").addEventListener("click", async () => {
  const response = await fetch("/api/deployment/now", { method: "POST" });
  if (response.ok) {
    const result = await response.json();
    const manual = state.latestConfig?.layer?.enabled !== false && state.latestConfig?.layer?.approval === "manual";
    setMessage(`推定位置の前程に ${result.deployed} 点の設標計画を作成しました（${manual ? "了承待ち：画面下部で了承してください" : "設標者が向かいます"}）。`);
  } else {
    setMessage(`前程配置できません: ${await response.text()}`);
  }
});

// ================================================================== estimates
function estimateGraphics(mode) {
  state.estimates = state.estimates || {};
  if (state.estimates[mode]) return state.estimates[mode];
  const color = mode === "ONLINE" ? COLORS.online : COLORS.smoothed;
  const graphics = new Marker(
    gpu.points.add({ pixelSize: 11, color, outlineColor: Cesium.Color.BLACK, outlineWidth: 2, id: `estimate-${mode}` }),
    gpu.labels.add({ text: mode === "ONLINE" ? "EST" : "", font: "12px sans-serif", fillColor: color, pixelOffset: new Cesium.Cartesian2(0, 20) }),
    `est:${mode}`,
  );
  state.estimates[mode] = graphics;
  return graphics;
}

function updateEstimateLayers(estimates) {
  for (const mode of ["ONLINE", "SMOOTHED"]) {
    const marker = estimateGraphics(mode);
    const estimate = estimates.find((e) => e.mode === mode);
    const visible = checked(mode === "ONLINE" ? "show-online" : "show-smoothed");
    const has = Boolean(estimate?.current_position);
    marker.show(visible && has);
    if (tracks.has(`est:${mode}`)) tracks.get(`est:${mode}`).setVisible(visible);
    if (has) marker.set(cartOf(estimate.current_position));
  }
  if (!state.errorLine) {
    state.errorLine = gpu.lines.add({ positions: [], width: 1.5, material: colorMaterial(COLORS.error.withAlpha(0.75)) });
  }
  updateErrorLine();
}

function updateErrorLine() {
  if (!state.errorLine) return;
  const marker = state.estimates?.[selectedMode()];
  const show = Boolean(checked("show-error-line") && marker?.pos && marker.point.show && state.truth?.pos);
  state.errorLine.show = show;
  if (show) state.errorLine.positions = [marker.pos, state.truth.pos];
}

// ---------------------------------------------------------------- presence region (batched GPU geometry)
const MAX_PENDING_MS = 15000;
const telemetry = {
  frames: 0, pendingDropped: 0, swaps: 0, renderMs: 0, maxRenderMs: 0,
  frameIntervalMs: 0, updateMs: 0, maxUpdateMs: 0, bytes: 0, parseMs: 0, updates: 0,
  vertexUploads: 0, quality: "", smooth: true, renderer: "", worker: false,
};
let frameStart = 0;
scene.preRender.addEventListener(() => {
  frameStart = performance.now();
});

function swapWhenReady(slot, now) {
  if (!slot.pending) return false;
  if (!slot.pending.ready) {
    // never spin the render loop forever: a build that is not ready in time is dropped and the
    // previous geometry stays (the next estimate update builds a fresh one)
    if (now - slot.pendingSince > MAX_PENDING_MS) {
      scene.primitives.remove(slot.pending);
      slot.pending = null;
      telemetry.pendingDropped += 1;
      return false;
    }
    return true;
  }
  if (slot.current) scene.primitives.remove(slot.current);
  slot.current = slot.pending;
  slot.pending = null;
  telemetry.swaps += 1;
  return false;
}
let pendingRenderScheduled = false;
scene.postRender.addEventListener(() => {
  telemetry.frames += 1;
  const took = performance.now() - frameStart;
  telemetry.renderMs = telemetry.renderMs ? 0.9 * telemetry.renderMs + 0.1 * took : took;
  telemetry.maxRenderMs = Math.max(telemetry.maxRenderMs, took);
  // async geometry is built in web workers; poll at a modest rate until it is uploaded, then swap
  const now = performance.now();
  const waiting = [gpu.region, gpu.regionOutline, gpu.voxels].map((slot) => swapWhenReady(slot, now)).some(Boolean);
  if (waiting && !pendingRenderScheduled) {
    pendingRenderScheduled = true;
    setTimeout(() => {
      pendingRenderScheduled = false;
      scene.requestRender();
    }, 250); // ~4 Hz polling while web workers build geometry
  }
});

function replacePrimitive(slot, primitive) {
  if (slot.pending) scene.primitives.remove(slot.pending);
  slot.pending = primitive ? scene.primitives.add(primitive) : null;
  slot.pendingSince = performance.now();
  if (!primitive && slot.current) {
    scene.primitives.remove(slot.current);
    slot.current = null;
  }
}

function regionPrimitives(region, withBoxes = true) {
  const fill = [];
  const outline = [];
  const voxels = [];
  const exag = exaggeration();
  const totalVoxels = region.components.reduce((n, c) => n + c.voxels.length, 0);
  const voxelShare = Math.min(1, voxelBoxBudget() / Math.max(totalVoxels, 1));
  region.components.forEach((component) => {
    if (component.polygon.length >= 3) {
      const hierarchy = new Cesium.PolygonHierarchy(Cesium.Cartesian3.fromDegreesArray(component.polygon.flat()));
      const height = heightOf(component.min_depth_ft);
      const extrudedHeight = heightOf(component.max_depth_ft);
      fill.push(new Cesium.GeometryInstance({
        geometry: new Cesium.PolygonGeometry({
          polygonHierarchy: hierarchy,
          height,
          extrudedHeight,
          vertexFormat: Cesium.PerInstanceColorAppearance.VERTEX_FORMAT,
        }),
        attributes: { color: Cesium.ColorGeometryInstanceAttribute.fromColor(COLORS.region.withAlpha(0.12)) },
      }));
      outline.push(new Cesium.GeometryInstance({
        geometry: new Cesium.PolygonOutlineGeometry({ polygonHierarchy: hierarchy, height, extrudedHeight }),
        attributes: { color: Cesium.ColorGeometryInstanceAttribute.fromColor(COLORS.region.withAlpha(0.85)) },
      }));
    }
    if (!withBoxes) return;
    const n = component.voxels.length;
    const size = component.voxel_size_yd * YD_TO_M * 0.9;
    const tall = component.voxel_height_ft * FT_TO_M * exag * 0.9;
    if (!(size > 0 && tall > 0)) return;
    const drawn = component.voxels.slice(0, Math.max(1, Math.round(n * voxelShare))); // densest first
    drawn.forEach((voxel, rank) => {
      const alpha = 0.34 - 0.26 * (rank / Math.max(n - 1, 1)); // densest voxels most opaque
      voxels.push(new Cesium.GeometryInstance({
        geometry: Cesium.BoxGeometry.fromDimensions({
          dimensions: new Cesium.Cartesian3(size, size, tall),
          vertexFormat: Cesium.PerInstanceColorAppearance.VERTEX_FORMAT,
        }),
        modelMatrix: Cesium.Transforms.eastNorthUpToFixedFrame(cart(voxel[0], voxel[1], voxel[2])),
        attributes: { color: Cesium.ColorGeometryInstanceAttribute.fromColor(COLORS.region.withAlpha(alpha)) },
      }));
    });
  });
  const translucent = () => new Cesium.PerInstanceColorAppearance({ translucent: true, closed: true });
  return {
    fill: fill.length ? new Cesium.Primitive({ geometryInstances: fill, appearance: translucent(), asynchronous: true, allowPicking: false }) : null,
    outline: outline.length
      ? new Cesium.Primitive({
        geometryInstances: outline,
        appearance: new Cesium.PerInstanceColorAppearance({ flat: true, translucent: false }),
        asynchronous: true,
        allowPicking: false,
      })
      : null,
    voxels: voxels.length ? new Cesium.Primitive({ geometryInstances: voxels, appearance: translucent(), asynchronous: true, allowPicking: false }) : null,
  };
}

function updateRegion(estimate) {
  const region = estimate?.presence_region;
  const voxelStyle = $("voxel-style") ? $("voxel-style").value : "box";
  const settings = `${estimate?.mode}-${checked("show-region")}-${checked("show-voxels")}-${voxelStyle}-${exaggeration()}-${state.runKey}-q${quality.level}`;
  const key = `${estimate?.tick}-${settings}`;
  if (key === state.lastRegionKey) return;
  // display settings changed -> rebuild now; new estimate only -> at most every REGION_REBUILD_MS
  const now = performance.now();
  const settingsChanged = settings !== state.lastRegionSettings;
  if (!settingsChanged && now - (state.lastRegionBuild || 0) < REGION_REBUILD_MS) return;
  state.lastRegionKey = key;
  state.lastRegionSettings = settings;
  state.lastRegionBuild = now;
  gpu.regionLabels.removeAll();
  gpu.voxelPoints.removeAll();
  if (!region || !region.components.length) {
    replacePrimitive(gpu.region, null);
    replacePrimitive(gpu.regionOutline, null);
    replacePrimitive(gpu.voxels, null);
    return;
  }
  const boxes = checked("show-voxels") && voxelStyle === "box" && voxelBoxBudget() > 0;
  const built = regionPrimitives(region, boxes);
  replacePrimitive(gpu.region, checked("show-region") ? built.fill : null);
  replacePrimitive(gpu.regionOutline, checked("show-region") ? built.outline : null);
  replacePrimitive(gpu.voxels, boxes ? built.voxels : null);
  if (checked("show-voxels") && (voxelStyle === "point" || voxelBoxBudget() === 0)) {
    for (const component of region.components) {
      const n = component.voxels.length;
      component.voxels.forEach((voxel, rank) => {
        gpu.voxelPoints.add({ position: cart(voxel[0], voxel[1], voxel[2]), pixelSize: 4, color: COLORS.region.withAlpha(0.7 - 0.5 * (rank / Math.max(n - 1, 1))) });
      });
    }
  }
  if (checked("show-region")) {
    for (const component of region.components) {
      gpu.regionLabels.add({
        position: cartOf(component.centroid),
        text: `${fmt(component.probability_mass_pct, 0)}%`,
        font: "12px sans-serif",
        fillColor: COLORS.region,
      });
    }
  }
}

// ================================================================== camera: views, centre, follow
function viewRadiusM() {
  const config = state.latestConfig;
  return (config ? config.max_slant_range_yd : 6000) * YD_TO_M;
}

function centreTarget() {
  // what the views, the initial view and follow mode centre on: truth by default
  return $("center-target")?.value === "estimate" ? "estimate" : "truth";
}

function focusPosition(prefer = centreTarget()) {
  const snapshot = state.latestSnapshot;
  if (!snapshot) return null;
  const estimate = selectedEstimate(snapshot);
  if (prefer === "estimate" && estimate?.current_position) return { position: estimate.current_position, source: "推定位置" };
  if (snapshot.target) return { position: snapshot.target.position, source: "真値" };
  if (estimate?.current_position) return { position: estimate.current_position, source: "推定位置" };
  return null;
}

function setProjection(view) {
  const camera = viewer.camera;
  const wantOrtho = checked("orthographic") && view === "top";
  const isOrtho = camera.frustum instanceof Cesium.OrthographicFrustum;
  if (wantOrtho && !isOrtho) camera.switchToOrthographicFrustum();
  if (!wantOrtho && isOrtho) camera.switchToPerspectiveFrustum();
}

function sideHeading(focus) {
  const choice = $("side-direction").value;
  if (choice !== "track") return Number(choice);
  const estimate = selectedEstimate();
  const course = estimate?.cog_deg ?? state.latestSnapshot?.target?.cog_deg ?? 0;
  return (course + 90) % 360; // look across the track: the course runs left -> right
}

function flightOptions(options) {
  // follow mode pauses while the camera flies, then re-anchors on the target
  state.flying = (state.flying || 0) + 1;
  const done = () => {
    state.flying = Math.max(0, (state.flying || 1) - 1);
    scene.requestRender();
  };
  return { ...options, complete: done, cancel: done };
}

function applyView(view) {
  const focus = focusPosition(centreTarget());
  if (!focus) {
    setMessage("表示対象（推定位置または真値）がまだありません。");
    return;
  }
  const camera = viewer.camera;
  camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
  setProjection(view);
  const p = focus.position;
  const radius = viewRadiusM();
  const duration = 1.0;
  if (view === "top") {
    camera.flyTo(flightOptions({
      destination: Cesium.Cartesian3.fromDegrees(p.longitude, p.latitude, heightOf(p.depth_ft) + radius * 2.6),
      orientation: { heading: 0, pitch: -Cesium.Math.PI_OVER_TWO, roll: 0 },
      duration,
    }));
  } else if (view === "side") {
    const heading = sideHeading(p);
    const distance = radius * 2.4;
    const from = destination(p, (heading + 180) % 360, distance);
    camera.flyTo(flightOptions({
      destination: Cesium.Cartesian3.fromDegrees(from.longitude, from.latitude, heightOf(p.depth_ft)),
      orientation: { heading: Cesium.Math.toRadians(heading), pitch: 0, roll: 0 },
      duration,
    }));
  } else {
    camera.flyToBoundingSphere(new Cesium.BoundingSphere(cartOf(p), radius), flightOptions({
      offset: new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-40), radius * 3.2),
      duration,
    }));
  }
  for (const button of document.querySelectorAll(".vt[data-view]")) button.classList.toggle("active", button.dataset.view === view);
  const names = { oblique: "斜視", top: "真上（垂直）", side: "水平（側面）" };
  setMessage(`視点：${names[view]}（中心：${focus.source}）`);
  scene.requestRender();
}

const INITIAL_CAMERA_ALT_FT = 10000; // initial camera altitude above the horizontal plane (sea surface)
const INITIAL_CAMERA_PITCH_DEG = -50;

function initialView() {
  // start-up view: placed at once (no flight, no zoom-in animation) at 10000 ft above the sea
  // surface, looking down obliquely at the centre target (truth by default). Follow is on by
  // default, so the view centre then stays on the truth target; no automatic zoom.
  const focus = focusPosition(centreTarget());
  if (!focus) return;
  const camera = viewer.camera;
  const target = cartOf(focus.position);
  const altitudeM = INITIAL_CAMERA_ALT_FT * FT_TO_M;
  const pitch = Cesium.Math.toRadians(INITIAL_CAMERA_PITCH_DEG);
  const below = altitudeM - heightOf(focus.position.depth_ft);
  let range = below / Math.sin(-pitch);
  // the target stays exactly on the line of sight; the range is refined so that the altitude
  // is the set value despite the earth's curvature (converges in 2-3 steps)
  for (let i = 0; i < 4; i += 1) {
    camera.lookAt(target, new Cesium.HeadingPitchRange(0, pitch, range));
    camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
    const error = camera.positionCartographic.height - altitudeM;
    if (!Number.isFinite(error) || Math.abs(error) < 0.01) break;
    range -= error / Math.sin(-pitch);
  }
  state.initialCamera = { height: altitudeM };
  for (const button of document.querySelectorAll(".vt[data-view]")) button.classList.toggle("active", button.dataset.view === "oblique");
  setMessage(`初期表示：高度 ${INITIAL_CAMERA_ALT_FT} ft、俯角 ${-INITIAL_CAMERA_PITCH_DEG}°。追従オン：${focus.source}を視点中心に保ちます（ツールバーで切替）`);
  scene.requestRender();
}

function centerOn(prefer, quiet = false) {
  const focus = focusPosition(prefer);
  if (!focus) {
    setMessage("中心に置く対象がまだありません。");
    return;
  }
  const camera = viewer.camera;
  camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
  const target = cartOf(focus.position);
  const range = Math.max(Cesium.Cartesian3.distance(camera.positionWC, target), 50);
  // keep the current viewing direction and distance (no zoom), move the target to the centre
  camera.flyToBoundingSphere(new Cesium.BoundingSphere(target, 1), flightOptions({
    offset: new Cesium.HeadingPitchRange(camera.heading, camera.pitch, range),
    duration: 0.8,
  }));
  if (!quiet) setMessage(`${focus.source}を画面中心にしました。`);
  scene.requestRender();
}

function followOn() {
  return checked("follow-on");
}

function focusMarker() {
  // the drawn (interpolated) marker of the follow target, so the camera moves as smoothly
  if (centreTarget() === "estimate") {
    const marker = state.estimates?.[selectedMode()];
    return marker?.pos && marker.point.show ? { marker, source: "推定位置" } : null;
  }
  return state.truth?.pos ? { marker: state.truth, source: "真値" } : null;
}

const scratchFollow = new Cesium.Cartesian3();

function enuRotation(position) {
  return Cesium.Matrix4.getMatrix3(Cesium.Transforms.eastNorthUpToFixedFrame(position), new Cesium.Matrix3());
}

function followTick() {
  // follow on: every frame the camera moves with the follow target (truth or estimate) so that
  // the view centre (the point on the camera's line of sight) stays on the target.
  //  1. the camera is carried rigidly from the target's previous local frame (east-north-up) to
  //     its current one, so altitude above the sea surface, depression angle and heading stay
  //     exactly as they were (no drift from the earth's curvature while the target travels)
  //  2. it is then slid along its viewing plane so the target lies on the line of sight
  //     (snaps to the target when follow is switched on or the target is changed)
  // Viewing direction and distance remain under the user's control (rotate / zoom).
  if (!followOn() || state.flying) {
    state.followPrev = null;
    return;
  }
  const focus = focusMarker();
  if (!focus) return;
  const camera = viewer.camera;
  camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
  const now = focus.marker.pos;
  const prev = state.followPrev;
  if (prev && state.followSource === focus.source && Cesium.Cartesian3.distance(prev, now) > 1e-3) {
    const rotation = Cesium.Matrix3.multiply(enuRotation(now), Cesium.Matrix3.transpose(enuRotation(prev), new Cesium.Matrix3()), new Cesium.Matrix3());
    const offset = Cesium.Cartesian3.subtract(camera.position, prev, new Cesium.Cartesian3());
    Cesium.Matrix3.multiplyByVector(rotation, offset, offset);
    Cesium.Cartesian3.add(now, offset, camera.position);
    for (const axis of ["direction", "up", "right"]) Cesium.Matrix3.multiplyByVector(rotation, camera[axis], camera[axis]);
  }
  const toTarget = Cesium.Cartesian3.subtract(now, camera.position, scratchFollow);
  let along = Cesium.Cartesian3.dot(toTarget, camera.direction);
  if (!(along > 1)) along = Math.max(Cesium.Cartesian3.magnitude(toTarget), 50);
  const position = Cesium.Cartesian3.subtract(now, Cesium.Cartesian3.multiplyByScalar(camera.direction, along, new Cesium.Cartesian3()), new Cesium.Cartesian3());
  if (Cesium.Cartesian3.distance(position, camera.position) > 0.01) Cesium.Cartesian3.clone(position, camera.position);
  state.followSource = focus.source;
  state.followPrev = Cesium.Cartesian3.clone(now, state.followPrev || new Cesium.Cartesian3());
}

// ---------------------------------------------------------------- camera / view-centre readout
const hudEllipsoids = new Map();
function planeEllipsoid(heightM) {
  // ellipsoid raised/lowered by heightM: the horizontal plane at that height (drawn scale)
  const key = Math.round(heightM * 10) / 10;
  if (!hudEllipsoids.has(key)) {
    const r = Cesium.Ellipsoid.WGS84.radii;
    if (hudEllipsoids.size > 32) hudEllipsoids.clear();
    hudEllipsoids.set(key, new Cesium.Ellipsoid(r.x + key, r.y + key, r.z + key));
  }
  return hudEllipsoids.get(key);
}

function cameraInfo() {
  // camera position, altitude above the horizontal plane (sea surface, ft), depression angle,
  // heading and the view centre: where the line of sight meets the horizontal plane at the
  // follow target's depth (equals the target while following)
  const camera = viewer.camera;
  const carto = Cesium.Cartographic.fromCartesian(camera.positionWC);
  if (!carto) return null;
  const focus = focusPosition(centreTarget());
  const depthFt = focus ? focus.position.depth_ft : 0;
  const info = {
    latitude: Cesium.Math.toDegrees(carto.latitude),
    longitude: Cesium.Math.toDegrees(carto.longitude),
    altitudeFt: carto.height / FT_TO_M,
    depressionDeg: -Cesium.Math.toDegrees(camera.pitch),
    headingDeg: Cesium.Math.toDegrees(camera.heading),
    planeDepthFt: depthFt,
    centre: null,
  };
  const ray = new Cesium.Ray(camera.positionWC, camera.directionWC);
  const hit = Cesium.IntersectionTests.rayEllipsoid(ray, planeEllipsoid(heightOf(depthFt)));
  if (hit) {
    const point = Cesium.Ray.getPoint(ray, Math.max(hit.start, 0));
    const c = Cesium.Cartographic.fromCartesian(point);
    if (c) {
      info.centre = {
        latitude: Cesium.Math.toDegrees(c.latitude), longitude: Cesium.Math.toDegrees(c.longitude),
        slantYd: Cesium.Cartesian3.distance(camera.positionWC, point) / YD_TO_M,
      };
    }
  }
  return info;
}

function latText(value) {
  return `${Math.abs(value).toFixed(5)}°${value >= 0 ? "N" : "S"}`;
}
function lonText(value) {
  return `${Math.abs(value).toFixed(5)}°${value >= 0 ? "E" : "W"}`;
}

let lastCameraHud = "";
function updateCameraHud() {
  const box = $("camera-hud");
  if (!box) return;
  const info = cameraInfo();
  if (!info) return;
  const centre = info.centre
    ? `${latText(info.centre.latitude)} ${lonText(info.centre.longitude)}`
    : "交点なし（水平より上を向いている）";
  const follow = followOn() ? `追従中（${centreTarget() === "estimate" ? "推定" : "真値"}）` : "追従オフ";
  const text = [
    `カメラ  ${latText(info.latitude)} ${lonText(info.longitude)}`,
    `高度    ${Math.round(info.altitudeFt).toLocaleString("ja-JP")} ft（海面から）`,
    `俯角    ${info.depressionDeg.toFixed(1)}°   方位 ${(((Math.round(info.headingDeg * 10) / 10) % 360 + 360) % 360).toFixed(1)}°`,
    `視点中心 ${centre}`,
    `        深度 ${fmt(info.planeDepthFt, 0)} ft 面${info.centre ? `・斜距離 ${Math.round(info.centre.slantYd).toLocaleString("ja-JP")} yd` : ""}  ${follow}`,
  ].join("\n");
  if (text !== lastCameraHud) {
    lastCameraHud = text;
    box.textContent = text;
  }
}
scene.postRender.addEventListener(updateCameraHud);

for (const button of document.querySelectorAll(".vt[data-view]")) {
  button.addEventListener("click", () => applyView(button.dataset.view));
}
function centreButton(which) {
  // while following, a centre button also switches what is followed
  if (followOn()) $("center-target").value = which;
  centerOn(which);
}
$("center-estimate").addEventListener("click", () => centreButton("estimate"));
$("center-truth").addEventListener("click", () => centreButton("truth"));
$("follow-on").addEventListener("change", () => {
  setMessage(followOn() ? `追従オン：${centreTarget() === "estimate" ? "推定位置" : "真値"}を視点中心に保ちます。` : "追従オフ");
  scene.requestRender();
});
$("center-target").addEventListener("change", () => {
  if (followOn()) scene.requestRender(); // the next frame snaps to the new target
  else centerOn(centreTarget());
});
$("orthographic").addEventListener("change", () => {
  const active = document.querySelector(".vt[data-view].active");
  setProjection(active ? active.dataset.view : "oblique");
  scene.requestRender();
});
$("show-fps").addEventListener("change", () => {
  scene.debugShowFramesPerSecond = checked("show-fps");
  // FPS needs continuous rendering to be meaningful
  scene.requestRenderMode = !checked("show-fps");
  scene.requestRender();
});

// ================================================================== adaptive quality & performance display
const QUALITY_LEVELS = [
  // boxes: translucent presence-region voxel boxes drawn (overdraw budget); 0 = drawn as points
  { name: "高", res: Math.min(window.devicePixelRatio || 1, 2), msaa: 4, fxaa: true, boxes: 400 },
  { name: "中", res: 1.0, msaa: 2, fxaa: true, boxes: 200 },
  { name: "低", res: 0.75, msaa: 1, fxaa: true, boxes: 80 },
  { name: "最低", res: 0.5, msaa: 1, fxaa: false, boxes: 0 },
];
// start one level below the top: a heavy close-up at full device resolution with 4x MSAA can
// saturate an integrated GPU before the first frame times are known; "auto" steps up to 高
// after a few seconds of fast frames
const START_QUALITY = 1;
const quality = { level: -1, lastFrame: 0, ema: 0, slowMs: 0, fastMs: 0, continuing: false, lastDrop: 0, requestedAt: 0 };
const TARGET_INTERVAL = 1000 / TARGET_FPS;

function applyQuality(level) {
  if (level === quality.level) return;
  quality.level = level;
  const q = QUALITY_LEVELS[level];
  viewer.resolutionScale = q.res;
  scene.msaaSamples = q.msaa;
  if (scene.postProcessStages && scene.postProcessStages.fxaa) scene.postProcessStages.fxaa.enabled = q.fxaa;
  telemetry.quality = q.name;
  state.lastRegionKey = ""; // the voxel-box budget depends on the level
  scene.requestRender();
}
applyQuality(START_QUALITY);

function voxelBoxBudget() {
  return QUALITY_LEVELS[Math.max(quality.level, 0)].boxes;
}

scene.postRender.addEventListener(() => {
  const now = performance.now();
  const dt = now - quality.lastFrame;
  quality.lastFrame = now;
  // a frame interval is meaningful only when the previous frame asked for the next one
  // (smooth markers gliding, the follow camera with them); otherwise it is on-demand idle time.
  // Slow continuous frames (also > 250 ms) are what can saturate the GPU and freeze the PC.
  const measured = quality.continuing && dt < 10000;
  quality.continuing = anim.active.size > 0; // gliding markers request the next frame (the follow camera rides on them)
  // latency from a data update to its frame on screen: the browser delays the next frame while
  // the GPU is still busy, so this also catches a GPU overloaded at the 1 Hz update rate
  const latency = quality.requestedAt ? now - quality.requestedAt : 0;
  quality.requestedAt = 0;
  telemetry.updateLatencyMs = latency || telemetry.updateLatencyMs || 0;
  if (latency > 400 && $("quality").value === "auto" && now - quality.lastDrop > 1000) {
    if (quality.level < QUALITY_LEVELS.length - 1) applyQuality(quality.level + 1);
    quality.lastDrop = now;
    if (latency > 1000 && anim.enabled) setSmooth(false, "描画が追いつかないため、なめらか表示を自動で停止しました（品質を下げるか、GPUの有効化を確認してください）。");
  }
  if (measured) {
    quality.ema = quality.ema ? 0.8 * quality.ema + 0.2 * dt : dt;
    telemetry.frameIntervalMs = quality.ema;
    quality.slowMs = quality.ema > 1.6 * TARGET_INTERVAL ? quality.slowMs + dt : 0;
    quality.fastMs = quality.ema < 1.15 * TARGET_INTERVAL ? quality.fastMs + dt : 0;
    const auto = $("quality").value === "auto";
    if (auto && dt > 250 && now - quality.lastDrop > 1000 && quality.level < QUALITY_LEVELS.length - 1) {
      applyQuality(quality.level + 1); // a very slow frame: lower at once, do not wait
      quality.lastDrop = now;
      quality.slowMs = 0;
    } else if (auto && quality.slowMs > 1500 && quality.level < QUALITY_LEVELS.length - 1) {
      applyQuality(quality.level + 1);
      quality.lastDrop = now;
      quality.slowMs = 0;
    } else if (auto && quality.fastMs > 6000 && quality.level > 0) {
      applyQuality(quality.level - 1);
      quality.fastMs = 0;
    }
    // continuous rendering needs ~10 fps at least; otherwise fall back to 1 Hz updates
    // (smooth off; following then moves the camera once per update)
    if (anim.enabled && (quality.ema > 100 || dt > 1000)) {
      anim.slowSince = anim.slowSince || now;
      if (dt > 1000 || now - anim.slowSince > 2000) setSmooth(false, "描画が追いつかないため、なめらか表示を自動で停止しました（品質を下げるか、GPUの有効化を確認してください）。");
    } else {
      anim.slowSince = null;
    }
  }
  if (anim.active.size) scene.requestRender(); // keep animating until markers arrive
  telemetry.smooth = anim.enabled;
});

$("quality").addEventListener("change", () => {
  const value = $("quality").value;
  quality.slowMs = quality.fastMs = 0;
  applyQuality(value === "auto" ? START_QUALITY : Number(value));
});
$("smooth-motion").addEventListener("change", () => setSmooth(checked("smooth-motion")));
$("show-sea").addEventListener("change", applyGlobe);
$("voxel-style").addEventListener("change", () => {
  state.lastRegionKey = "";
  if (state.latestSnapshot) render(state.latestSnapshot);
});

(function detectRenderer() {
  try {
    const gl = scene.context._gl;
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    telemetry.renderer = String(info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
  } catch (error) {
    telemetry.renderer = "unknown";
  }
  telemetry.software = /swiftshader|llvmpipe|software|basic render/i.test(telemetry.renderer);
  if (telemetry.software) {
    setMessage("ソフトウェア描画です（GPUが使われていません）。ブラウザのハードウェアアクセラレーション設定を確認してください。");
  }
})();

let perfFrames = 0;
setInterval(() => {
  const frames = telemetry.frames - perfFrames;
  perfFrames = telemetry.frames;
  const hud = $("perf-hud");
  if (!hud || !checked("show-perf")) {
    if (hud) hud.hidden = true;
    return;
  }
  hud.hidden = false;
  hud.textContent = [
    `描画     ${frames} fps（フレーム間隔 ${fmt(telemetry.frameIntervalMs, 0)} ms）`,
    `更新処理 ${fmt(telemetry.updateMs, 1)} ms（最大 ${fmt(telemetry.maxUpdateMs, 0)} ms）`,
    `解読     ${fmt(telemetry.parseMs, 1)} ms（${telemetry.worker ? "Worker" : "メインスレッド"}）`,
    `受信     ${fmt(telemetry.bytes / 1024, 1)} KB/更新`,
    `品質     ${telemetry.quality}${$("quality").value === "auto" ? "（自動）" : ""}　なめらか ${anim.enabled ? "オン" : "オフ"}`,
    `GPU      ${telemetry.renderer}${telemetry.software ? "  ⚠ ソフトウェア描画" : ""}`,
  ].join("\n");
}, 1000);

// ================================================================== estimation control
function updateRunState(snapshot) {
  const control = snapshot.estimation || {};
  const badge = $("run-badge");
  badge.textContent = control.running ? "推定中" : "停止中";
  badge.className = `badge ${control.running ? "running" : "stopped"}`;
  const end = control.running ? snapshot.tick : (control.stopped_tick ?? snapshot.tick);
  setText("run-info", `Run #${control.run_id ?? 0}　開始 ${clockAt(control.started_tick ?? 0)}　経過 ${Math.max(0, end - (control.started_tick ?? 0))} s` +
    (control.running ? "" : `　停止 ${control.stopped_tick == null ? "--" : clockAt(control.stopped_tick)}`));
  $("est-stop").disabled = !control.running;
  const key = `${snapshot.generation}-${control.run_id}`;
  if (key !== state.runKey) {
    state.runKey = key;
    state.history = [];
    state.trueCpa.clear();
    if (snapshot.generation !== state.generation) {
      state.generation = snapshot.generation;
      clearBearings();
    }
  }
}

async function postControl(path, message) {
  const response = await fetch(path, { method: "POST" });
  if (!response.ok) throw new Error(await response.text());
  setMessage(message);
  return response.json();
}
$("est-start").addEventListener("click", () => {
  postControl("/api/estimation/start", "推定を開始しました（現在時刻以降の観測を使用）。").catch((error) => setMessage(`推定開始エラー: ${error.message}`));
});
$("est-stop").addEventListener("click", () => {
  postControl("/api/estimation/stop", "推定を停止しました（最後の推定結果を保持表示）。").catch((error) => setMessage(`推定停止エラー: ${error.message}`));
});

// ================================================================== simulation speed (time scale)
const clock = { scale: 1, samples: [] };

function formatScale(scale) {
  return `${Number(scale.toFixed(2))}倍`;
}

function showClock(scale) {
  clock.scale = scale;
  const badge = $("clock-badge");
  badge.textContent = scale > 0 ? `${formatScale(scale)}速` : "一時停止";
  badge.className = `badge ${scale > 0 ? "running" : "paused"}`;
  for (const button of document.querySelectorAll("[data-time-scale]")) {
    button.classList.toggle("active", Number(button.dataset.timeScale) === scale);
  }
  if (document.activeElement !== $("time-scale")) $("time-scale").value = scale;
  if (scale === 0) setText("clock-info", "実効 0倍（停止中）");
}

function noteClockTick(tick) {
  // achieved speed = simulation seconds per wall second over the last ~5 s
  const now = performance.now();
  const samples = clock.samples;
  if (samples.length && tick < samples[samples.length - 1].tick) samples.length = 0;
  samples.push({ tick, now });
  while (samples.length > 2 && now - samples[0].now > 5000) samples.shift();
  const first = samples[0];
  const span = (now - first.now) / 1000;
  if (clock.scale === 0) setText("clock-info", "実効 0倍（停止中）");
  else if (span >= 1.5) setText("clock-info", `実効 ${formatScale((tick - first.tick) / span)}`);
}

async function setTimeScale(scale) {
  const response = await fetch("/api/clock", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ time_scale: scale }),
  });
  if (!response.ok) throw new Error(await response.text());
  const settings = await response.json();
  clock.samples.length = 0;
  showClock(settings.time_scale);
  setMessage(settings.time_scale > 0 ? `時間倍率を ${formatScale(settings.time_scale)} にしました。` : "シミュレーション時間を一時停止しました。");
}

function applyTimeScale(scale) {
  if (!Number.isFinite(scale) || scale < 0 || scale > 100) {
    setMessage("時間倍率は 0〜100 の数値で指定してください（0 で一時停止）。");
    return;
  }
  setTimeScale(scale).catch((error) => setMessage(`時間倍率の変更エラー: ${error.message}`));
}

for (const button of document.querySelectorAll("[data-time-scale]")) {
  button.addEventListener("click", () => applyTimeScale(Number(button.dataset.timeScale)));
}
$("clock-form").addEventListener("submit", (event) => {
  event.preventDefault();
  applyTimeScale(Number($("time-scale").value));
});


// ================================================================== comparison (cards, no horizontal scroll)
function judge(error, sigma) {
  if (error == null || !Number.isFinite(error) || !(sigma > 0)) return { cls: "s-none", text: "" };
  const ratio = Math.abs(error) / sigma;
  if (ratio <= 1) return { cls: "s-good", text: "● 1σ内" };
  if (ratio <= 3) return { cls: "s-warn", text: "▲ 3σ内" };
  return { cls: "s-bad", text: "■ 3σ超" };
}

function card(title, errorText, status, rows, wide = false) {
  const body = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  return `<div class="card${wide ? " wide" : ""}"><div class="card-head"><span class="card-title">${title}</span>` +
    `<span class="status ${status.cls}">${status.text}</span></div><div class="card-err">${errorText}</div><dl>${body}</dl></div>`;
}

function updateComparison(snapshot, estimate) {
  const target = snapshot.target;
  const container = $("compare-cards");
  if (!estimate || !estimate.current_position || !target) {
    setHtml(container, `<p class="empty">${snapshot.estimation?.running ? "探知待ち（推定前）" : "推定停止中"}</p>`);
    setText("region-check", "--");
    return;
  }
  const u = estimate.uncertainty;
  const tp = target.position;
  const ep = estimate.current_position;
  const d = offsetM(tp, ep);
  const hErr = Math.hypot(d.east, d.north) / YD_TO_M;
  const trueVertical = target.through_water_velocity.vertical_fps;
  const trueBias = snapshot.config.source.shared_recognition_bias_hz;
  const depthErr = estimate.depth_ft - tp.depth_ft;
  const hdgErr = angleDiff(estimate.hdg_deg, target.hdg_deg);
  const cogErr = angleDiff(estimate.cog_deg, target.cog_deg);
  const stwErr = estimate.through_water_speed_kt - target.through_water_speed_kt;
  const sogErr = estimate.ground_speed_kt - target.ground_speed_kt;
  const biasErr = estimate.source_bias_hz - trueBias;
  const last = state.history[state.history.length - 1];
  if (!last || last.tick !== estimate.tick) { // the error history is kept even when hidden
    state.history.push({ tick: estimate.tick, h: hErr, hs: u.horizontal_major_yd, d: depthErr, ds: u.depth_sigma_ft });
    if (state.history.length > MAX_HISTORY) state.history.shift();
  }
  if (!tabVisible("tab-compare")) return;
  const cards = [
    card("水平位置", `${fmt(hErr, 0)} <small>YD</small>`, judge(hErr, u.horizontal_major_yd), [
      ["推定", `${latText(ep.latitude)}<br>${lonText(ep.longitude)}`],
      ["真値", `${latText(tp.latitude)}<br>${lonText(tp.longitude)}`],
      ["東西/南北", `${signed(d.east / YD_TO_M, 0)} / ${signed(d.north / YD_TO_M, 0)} YD`],
      ["1σ 長×短", `${fmt(u.horizontal_major_yd, 0)} × ${fmt(u.horizontal_minor_yd, 0)} YD`],
    ], true),
    card("深度", `${signed(depthErr, 0)} <small>Ft</small>`, judge(depthErr, u.depth_sigma_ft), [
      ["推定", `${fmt(estimate.depth_ft, 0)} Ft`], ["真値", `${fmt(tp.depth_ft, 0)} Ft`], ["1σ", `${fmt(u.depth_sigma_ft, 0)} Ft`],
    ]),
    card("HDG", `${signed(hdgErr)}<small>°</small>`, judge(hdgErr, u.hdg_sigma_deg), [
      ["推定", `${fmt(estimate.hdg_deg)}°`], ["真値", `${fmt(target.hdg_deg)}°`], ["1σ", `${fmt(u.hdg_sigma_deg)}°`],
    ]),
    card("COG", `${signed(cogErr)}<small>°</small>`, judge(cogErr, u.cog_sigma_deg), [
      ["推定", `${fmt(estimate.cog_deg)}°`], ["真値", `${fmt(target.cog_deg)}°`], ["1σ", `${fmt(u.cog_sigma_deg)}°`],
    ]),
    card("対水速力", `${signed(stwErr, 2)} <small>kt</small>`, judge(stwErr, u.through_water_speed_sigma_kt), [
      ["推定", `${fmt(estimate.through_water_speed_kt)} kt`], ["真値", `${fmt(target.through_water_speed_kt)} kt`], ["1σ", `${fmt(u.through_water_speed_sigma_kt, 2)} kt`],
    ]),
    card("対地速力", `${signed(sogErr, 2)} <small>kt</small>`, judge(sogErr, u.ground_speed_sigma_kt), [
      ["推定", `${fmt(estimate.ground_speed_kt)} kt`], ["真値", `${fmt(target.ground_speed_kt)} kt`], ["1σ", `${fmt(u.ground_speed_sigma_kt, 2)} kt`],
    ]),
    card("深度変化率", `${signed(estimate.vertical_rate_fps - trueVertical, 2)} <small>Ft/s</small>`, { cls: "s-none", text: "" }, [
      ["推定", `${fmt(estimate.vertical_rate_fps, 2)} Ft/s`], ["真値", `${fmt(trueVertical, 2)} Ft/s`],
    ]),
    card("周波数偏り", `${signed(biasErr, 3)} <small>Hz</small>`, judge(biasErr, u.bias_sigma_hz), [
      ["推定", `${fmt(estimate.source_bias_hz, 3)} Hz`], ["真値", `${fmt(trueBias, 3)} Hz`], ["1σ", `${fmt(u.bias_sigma_hz, 3)} Hz`],
    ]),
  ];
  setHtml(container, cards.join(""));

  const region = estimate.presence_region;
  const inside = region.components.find((c) =>
    pointInPolygon(tp.longitude, tp.latitude, c.polygon) && tp.depth_ft >= c.min_depth_ft && tp.depth_ft <= c.max_depth_ft);
  setHtml($("region-check"), `<b>${escapeHtml(estimate.observability_status)}</b>　真値は推定存在圏（${fmt(region.probability_pct, 0)} %・${region.components.length} 領域）の` +
    (inside ? `<span class="s-good">内側</span>` : `<span class="s-bad">外側</span>`) +
    `<br><small>入力：${escapeHtml(estimate.metadata?.observation_inputs || "--")}</small>` +
    maneuverTimingText(estimate.metadata?.maneuver_timing));
}

// last maneuver-timing event: when each observer heard the change, relative to the first
function maneuverTimingText(timing) {
  if (!timing || !timing.enabled) return "";
  const last = timing.last_event;
  const head = `<br><small>変針・変速の到達時間差：採用 ${timing.events_applied} 件・不採用 ${timing.events_rejected} 件`;
  if (!last) return `${head}</small>`;
  const rows = last.observers.map((o) => `${escapeHtml(o.observer_id)} +${fmt(o.offset_s, 2)} s（±${fmt(o.sigma_s, 2)}）`).join("、");
  return `${head}<br>最新（${clockAt(last.tick)}${last.used ? "" : "・不採用"}）：${rows}</small>`;
}

function pair(a, b) {
  return `<span class="est">${a}</span><span class="tru">${b}</span>`;
}

function updateRelative(snapshot, estimate) {
  for (const t of snapshot.doppler?.truth || []) { // true CPA is accumulated even when hidden
    const best = state.trueCpa.get(t.observer_id);
    if (!best || t.slant_range_yd < best.range) state.trueCpa.set(t.observer_id, { range: t.slant_range_yd, tick: t.tick });
  }
  if (!tabVisible("tab-compare")) return;
  const truth = new Map((snapshot.doppler?.truth || []).map((t) => [t.observer_id, t]));
  const bearings = new Map((snapshot.bearings || []).map((b) => [b.observer_id, b]));
  const estRel = new Map((estimate?.relative || []).map((r) => [r.observer_id, r]));
  const detected = new Set((snapshot.doppler?.observations || []).filter((o) => o.detected).map((o) => o.observer_id));
  const rows = snapshot.observers
    .map((r) => r.state.observer_id)
    .map((id) => ({ id, est: estRel.get(id), tr: truth.get(id), brg: bearings.get(id), det: detected.has(id) }))
    .sort((a, b) => Number(b.det) - Number(a.det) || (a.tr?.slant_range_yd ?? 1e9) - (b.tr?.slant_range_yd ?? 1e9))
    .slice(0, 40)
    .map((r) => `<tr class="${r.det ? "det" : ""}"><td>${r.det ? "● " : "– "}${escapeHtml(r.id)}</td>` +
      `<td>${pair(`${fmt(r.est?.relative_speed_kt)} kt`, `${fmt(r.tr?.relative_speed_kt)} kt`)}</td>` +
      `<td>${pair(`${fmt(r.est?.slant_range_yd, 0)} YD`, `${fmt(r.tr?.slant_range_yd, 0)} YD`)}</td>` +
      `<td>${pair(r.brg ? `${fmt(r.brg.bearing_deg)}°` : "--", `${fmt(r.tr?.true_bearing_deg)}°`)}</td></tr>`);
  setHtml($("relative-table").querySelector("tbody"), rows.join("") || "<tr><td colspan='4'>観測者なし</td></tr>");
}

function updateCpa(cpa) {
  if (!tabVisible("tab-compare")) return;
  const rows = cpa
    .slice()
    .sort((a, b) => b.cpa_tick - a.cpa_tick)
    .slice(0, 30)
    .map((item) => {
      const tr = state.trueCpa.get(item.observer_id);
      return `<tr class="${item.final ? "" : "provisional"}"><td>${escapeHtml(item.observer_id)}#${item.pass_index}</td>` +
        `<td>${pair(`${clockAt(item.cpa_tick)}±${fmt(item.cpa_tick_sigma_s, 0)} s`, tr ? clockAt(tr.tick) : "--")}</td>` +
        `<td>${pair(`${fmt(item.cpa_slant_range_yd, 0)}±${fmt(item.cpa_slant_range_sigma_yd, 0)} YD`, `${tr ? fmt(tr.range, 0) : "--"} YD`)}</td>` +
        `<td>${fmt(item.relative_speed_kt, 2)}<br><small>±${fmt(item.relative_speed_sigma_kt, 2)} kt</small></td></tr>`;
    });
  setHtml($("cpa-table").querySelector("tbody"), rows.join("") || "<tr><td colspan='4'>最近接通過なし</td></tr>");
}

// ================================================================== Lloyd's mirror depth
const LLOYD_STATUS = {
  OK: "採用", AMBIGUOUS: "曖昧（縞の次数）", NO_FRINGES: "縞が不足", NO_PATTERN: "干渉が見えない", FEW_SAMPLES: "標本不足",
};

function updateLloyd(lloyd, config, target) {
  const enabled = Boolean(config?.lloyd?.enabled);
  if (!state.lloydPending && $("lloyd-enabled").checked !== enabled) $("lloyd-enabled").checked = enabled;
  const summary = $("lloyd-summary");
  const body = $("lloyd-table").querySelector("tbody");
  if (!enabled || !lloyd || !lloyd.enabled) {
    summary.textContent = enabled ? "準備中（受信レベルの蓄積を開始）" : "オフ（計算していません）";
    body.innerHTML = "";
    return;
  }
  if (lloyd.status === "OK") {
    const truth = target ? target.position.depth_ft : null;
    const error = truth == null ? "" : `、真値 ${fmt(truth, 0)}・誤差 ${lloyd.depth_ft - truth >= 0 ? "+" : ""}${fmt(lloyd.depth_ft - truth, 0)}`;
    summary.textContent = `深度 ${fmt(lloyd.depth_ft, 0)} Ft ±${fmt(lloyd.sigma_ft, 0)}${error}（観測者 ${lloyd.used_observers}、`
      + `${clockAt(lloyd.tick)} 時点、計算 ${fmt(lloyd.fit_ms, 0)} ms${lloyd.applied ? "、推定に反映" : ""}）`;
  } else if (lloyd.status === "WAITING") {
    summary.textContent = "待機中（追尾の水平精度、または受信レベルの蓄積を待っています）";
  } else {
    summary.textContent = "結果なし（干渉縞が不足、または縞の次数が曖昧）";
  }
  body.innerHTML = (lloyd.observers || []).map((o) => `<tr><td>${escapeHtml(o.observer_id)}</td>`
    + `<td>${LLOYD_STATUS[o.status] || escapeHtml(o.status)}</td>`
    + `<td>${o.depth_ft == null ? "--" : `${fmt(o.depth_ft, 0)} ±${fmt(o.sigma_ft, 0)}`}</td>`
    + `<td>${fmt(o.fringes, 1)}</td></tr>`).join("");
}

$("lloyd-enabled").addEventListener("change", () => {
  const on = checked("lloyd-enabled");
  state.lloydPending = true;
  putConfig((next) => { next.lloyd.enabled = on; })
    .then(() => setMessage(on ? "ロイドミラー深度の計算を開始しました（処理負荷が増えます）。" : "ロイドミラー深度の計算を停止しました。"))
    .catch((error) => {
      $("lloyd-enabled").checked = !on;
      setMessage(`設定エラー: ${error.message}`);
    })
    .finally(() => { state.lloydPending = false; });
});

function updateCurrent(current, config) {
  if (!tabVisible("tab-compare")) return;
  const tbody = $("current-table").querySelector("tbody");
  const t = config.current_field;
  const dirSpeed = (e, n) => `${fmt((Math.atan2(e, n) * 180 / Math.PI + 360) % 360, 0)}° ${fmt(Math.hypot(e, n), 2)} kt`;
  if (!current) {
    setHtml(tbody, `<tr><td>基準流速</td><td>--</td><td>${dirSpeed(t.base_velocity.east_kt, t.base_velocity.north_kt)}</td></tr>`);
    return;
  }
  const b = current.base_velocity;
  const g = current.gradient_per_nm;
  const tg = t.gradient_per_nm;
  setHtml(tbody, [
    `<tr><td>流速</td><td>${dirSpeed(b.east_kt, b.north_kt)}</td><td>${dirSpeed(t.base_velocity.east_kt, t.base_velocity.north_kt)}</td></tr>`,
    `<tr><td>∂u/∂x, ∂v/∂y<br><small>kt/NM</small></td><td>${fmt(g[0][0], 3)}, ${fmt(g[1][1], 3)}</td><td>${fmt(tg[0][0], 3)}, ${fmt(tg[1][1], 3)}</td></tr>`,
    `<tr><td>観測者 / 期間</td><td>${current.observer_count} / ${fmt(current.window_seconds / 60, 0)} 分</td><td>--</td></tr>`,
    `<tr><td>当てはめ残差</td><td>${fmt(current.residual_kt, 3)} kt</td><td>--</td></tr>`,
  ].join(""));
}

// ================================================================== charts (canvas)
const CHART = { series: "#3987e5", band: "rgba(160, 180, 200, 0.22)", grid: "rgba(143, 174, 203, 0.16)", axis: "#8faecb", text: "#c9dcef" };

function niceMax(value) {
  if (!(value > 0)) return 1;
  const exp = Math.pow(10, Math.floor(Math.log10(value)));
  const f = value / exp;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * exp;
}

function nearest(points, tick) {
  return points.reduce((best, q) => (Math.abs(q.tick - tick) < Math.abs(best.tick - tick) ? q : best));
}

function drawChart(canvas, points, opts) {
  const ctx = canvas.getContext && canvas.getContext("2d");
  if (!ctx) return;
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 340;
  const height = Number(canvas.getAttribute("height")) || 130;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  canvas.style.height = `${height}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const pad = { l: 44, r: 30, t: 8, b: 20 };
  const w = width - pad.l - pad.r;
  const h = height - pad.t - pad.b;
  canvas._chart = { points, pad, w, h, opts };
  ctx.font = "10px sans-serif";
  ctx.fillStyle = CHART.axis;
  if (points.length < 2) {
    ctx.fillText("推定開始後に表示", pad.l, pad.t + h / 2);
    return;
  }
  const t0 = points[0].tick;
  const t1 = points[points.length - 1].tick;
  const maxAbs = Math.max(...points.map((p) => Math.max(Math.abs(opts.value(p)), opts.sigma(p))));
  const yMax = niceMax(maxAbs * 1.05);
  const yMin = opts.symmetric ? -yMax : 0;
  const x = (t) => pad.l + ((t - t0) / Math.max(t1 - t0, 1)) * w;
  const y = (v) => pad.t + h - ((v - yMin) / (yMax - yMin)) * h;
  canvas._chart.x = x;
  ctx.strokeStyle = CHART.grid;
  ctx.lineWidth = 1;
  const steps = opts.symmetric ? [-yMax, -yMax / 2, 0, yMax / 2, yMax] : [0, yMax / 2, yMax];
  for (const v of steps) {
    ctx.beginPath();
    ctx.moveTo(pad.l, y(v));
    ctx.lineTo(pad.l + w, y(v));
    ctx.stroke();
    ctx.fillStyle = CHART.axis;
    ctx.textAlign = "right";
    ctx.fillText(String(Math.round(v)), pad.l - 4, y(v) + 3);
  }
  ctx.textAlign = "left";
  ctx.fillText(clockAt(t0), pad.l, height - 5);
  ctx.textAlign = "right";
  ctx.fillText(clockAt(t1), pad.l + w, height - 5);
  ctx.fillStyle = CHART.band;
  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(x(p.tick), y(opts.sigma(p))) : ctx.moveTo(x(p.tick), y(opts.sigma(p)))));
  for (let i = points.length - 1; i >= 0; i -= 1) ctx.lineTo(x(points[i].tick), y(opts.symmetric ? -opts.sigma(points[i]) : 0));
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = CHART.series;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(x(p.tick), y(opts.value(p))) : ctx.moveTo(x(p.tick), y(opts.value(p)))));
  ctx.stroke();
  const lastPoint = points[points.length - 1];
  ctx.fillStyle = CHART.text;
  ctx.textAlign = "left";
  ctx.fillText("誤差", pad.l + w + 3, y(opts.value(lastPoint)) + 3);
  ctx.fillStyle = CHART.axis;
  ctx.fillText("1σ", pad.l + w + 3, y(opts.sigma(lastPoint)) - 4);
  if (canvas._hoverTick != null) {
    const p = nearest(points, canvas._hoverTick);
    ctx.strokeStyle = CHART.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x(p.tick), pad.t);
    ctx.lineTo(x(p.tick), pad.t + h);
    ctx.stroke();
    ctx.fillStyle = CHART.series;
    ctx.beginPath();
    ctx.arc(x(p.tick), y(opts.value(p)), 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

const chartDefs = {
  "chart-horizontal": { value: (p) => p.h, sigma: (p) => p.hs, symmetric: false, unit: "YD" },
  "chart-depth": { value: (p) => p.d, sigma: (p) => p.ds, symmetric: true, unit: "Ft" },
};

function drawCharts() {
  if (!tabVisible("tab-compare")) return;
  for (const [id, opts] of Object.entries(chartDefs)) {
    const canvas = $(id);
    if (canvas) drawChart(canvas, state.history, opts);
  }
}

for (const [id, opts] of Object.entries(chartDefs)) {
  const canvas = $(id);
  if (!canvas || !canvas.addEventListener) continue;
  canvas.addEventListener("mousemove", (event) => {
    const chart = canvas._chart;
    if (!chart || !chart.x || chart.points.length < 2) return;
    const rect = canvas.getBoundingClientRect();
    const t0 = chart.points[0].tick;
    const t1 = chart.points[chart.points.length - 1].tick;
    canvas._hoverTick = t0 + ((event.clientX - rect.left - chart.pad.l) / chart.w) * (t1 - t0);
    const p = nearest(chart.points, canvas._hoverTick);
    const tip = $("chart-tooltip");
    tip.hidden = false;
    tip.textContent = `${clockAt(p.tick)}　誤差 ${signed(opts.value(p), 0)} ${opts.unit}　1σ ${fmt(opts.sigma(p), 0)} ${opts.unit}`;
    tip.style.left = `${event.clientX + 12}px`;
    tip.style.top = `${event.clientY - 28}px`;
    drawChart(canvas, state.history, opts);
  });
  canvas.addEventListener("mouseleave", () => {
    canvas._hoverTick = null;
    $("chart-tooltip").hidden = true;
    drawChart(canvas, state.history, opts);
  });
}

// ================================================================== forms
function populateForms(config) {
  state.latestConfig = config;
  if (state.formsLoaded) return;
  const t = config.target;
  const values = {
    "init-lat": t.initial_position.latitude.toFixed(4),
    "init-lon": t.initial_position.longitude.toFixed(4),
    "init-depth": t.initial_position.depth_ft,
    "init-hdg": t.initial_hdg_deg,
    "init-speed": t.initial_through_water_speed_kt,
    "desired-hdg": t.desired_hdg_deg,
    "hdg-rate": t.hdg_rate_deg_per_sec,
    "desired-speed": t.desired_through_water_speed_kt,
    "speed-rate": t.speed_rate_kt_per_sec,
    "desired-depth": t.desired_depth_ft,
    "depth-rate": t.depth_rate_ft_per_sec,
    "max-range": config.max_slant_range_yd,
    "observer-limit": config.observer_limit,
    "source-frequency": config.source.source_frequency_hz,
    "frequency-bias": config.source.shared_recognition_bias_hz,
    "source-bandwidth": config.source.bandwidth_hz,
    "source-stability": config.source.stability_hz,
    "source-stability-tau": config.source.stability_correlation_s,
    "additional-tonals": (config.source.additional_tonals || [])
      .map((t) => `${t.frequency_hz}, ${t.bandwidth_hz}, ${t.stability_hz}`).join("\n"),
    "est-stability": config.estimator.assumed_stability_hz,
    "bearing-sigma": config.bearing.sigma_deg,
    "bearing-interval": config.bearing.interval_s,
    "cur-e": config.current_field.base_velocity.east_kt,
    "cur-n": config.current_field.base_velocity.north_kt,
    g00: config.current_field.gradient_per_nm[0][0],
    g11: config.current_field.gradient_per_nm[1][1],
    "est-bearing-sigma": config.estimator.bearing_sigma_deg,
    "bias-sigma": config.estimator.assumed_bias_sigma_hz,
    probability: config.presence_probability_pct,
    "smoothing-window": config.smoothing_window_seconds,
    particles: config.estimator.particle_count,
    "place-lat": t.initial_position.latitude.toFixed(4),
    "place-lon": t.initial_position.longitude.toFixed(4),
  };
  for (const [id, value] of Object.entries(values)) if ($(id)) $(id).value = value;
  $("bearing-enabled").checked = config.bearing.enabled;
  const f = config.forward;
  $("fwd-enabled").checked = f.enabled;
  const fwdValues = { "fwd-lead": f.lead_time_s, "fwd-min": f.min_coverage, "fwd-ahead": f.ahead_distance_yd,
    "fwd-lateral": f.lateral_offset_yd, "fwd-count": f.observers_per_drop, "fwd-cooldown": f.cooldown_s };
  for (const [id, value] of Object.entries(fwdValues)) $(id).value = value;
  if (f.strategy) {
    $("fwd-strategy").value = f.strategy;
    $("fwd-horizon").value = f.horizon_s;
    $("fwd-target").value = f.target_error_yd;
    $("fwd-max").value = f.max_per_drop;
    $("fwd-depths").value = (f.depth_options_ft || []).join(", ");
    $("fwd-depth-weight").value = f.depth_weight;
    $("fwd-min-gain").value = Math.round((f.min_relative_gain || 0) * 100);
    $("fwd-maneuver").value = Math.round((f.maneuver_weight ?? 0.4) * 100);
    $("fwd-gate").checked = f.use_detection_gate ?? true;
  }
  $("use-bearing").checked = config.estimator.use_bearing;
  $("propagation-delay").checked = config.source.propagation_delay;
  $("use-maneuver-timing").checked = config.estimator.use_maneuver_timing;
  const lay = config.layer;
  if (lay) {
    $("layer-enabled").checked = lay.enabled;
    $("layer-paused").checked = Boolean(lay.paused);
    $("layer-turn").value = lay.preferred_turn || "left";
    const layValues = { "layer-speed": lay.speed_kt, "layer-spread": lay.speed_spread_kt, "layer-bank": lay.max_bank_deg,
      "layer-orbit": lay.orbit_radius_yd, "layer-turn-margin": lay.turn_margin_s ?? 10, "layer-timeout": lay.proposal_timeout_s,
      "layer-cruise-alt": lay.cruise_altitude_ft ?? 3000, "layer-drop-alt": lay.drop_altitude_ft ?? 1000, "layer-climb": lay.climb_rate_fpm ?? 2000 };
    for (const [id, value] of Object.entries(layValues)) $(id).value = value;
    $("layer-wind-correction").checked = lay.wind_correction !== false;
  }
  fillWindTable(config);
  const l = config.lloyd || {};
  const lloydValues = { "lloyd-noise": l.level_noise_db, "lloyd-corr": l.noise_correlation_s, "lloyd-wave": l.wave_height_rms_m,
    "lloyd-path-error": l.path_difference_error_pct, "est-lloyd-model-error": config.estimator.lloyd_model_error_pct,
    "est-lloyd-interval": config.estimator.lloyd_fit_interval_s };
  for (const [id, value] of Object.entries(lloydValues)) if ($(id) && value != null) $(id).value = value;
  state.formsLoaded = true;
}

async function putConfig(mutate) {
  const next = structuredClone(state.latestConfig);
  mutate(next);
  const response = await fetch("/api/config", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(next) });
  if (!response.ok) throw new Error(await response.text());
  state.latestConfig = await response.json();
}

const num = (id) => Number($(id).value);

// "周波数, 帯域幅, 安定度" per line -> additional tonals (bandwidth / stability default 0)
function parseTonals(text, correlation) {
  return text.split(/\n/).map((line) => line.trim()).filter(Boolean).map((line) => {
    const [frequency, bandwidth = 0, stability = 0] = line.split(/[,\s]+/).map(Number);
    if (!(frequency > 0) || !(bandwidth >= 0) || !(stability >= 0)) throw new Error(`音源周波数の行が不正です: ${line}`);
    return { frequency_hz: frequency, bandwidth_hz: bandwidth, stability_hz: stability, stability_correlation_s: correlation };
  });
}

$("initial-form").addEventListener("submit", (event) => {
  event.preventDefault();
  putConfig((next) => {
    const t = next.target;
    t.initial_position = { latitude: num("init-lat"), longitude: num("init-lon"), depth_ft: num("init-depth") };
    t.initial_hdg_deg = num("init-hdg") % 360;
    t.initial_through_water_speed_kt = num("init-speed");
    t.desired_hdg_deg = t.initial_hdg_deg; // base motion: constant course / speed / depth
    t.desired_through_water_speed_kt = t.initial_through_water_speed_kt;
    t.desired_depth_ft = t.initial_position.depth_ft;
    next.current_field.reference_position = { ...t.initial_position, depth_ft: 0 };
  })
    .then(() => fetch(`/api/reset?replace_observers=${checked("init-replace-observers")}`, { method: "POST" }))
    .then((response) => {
      if (!response.ok) throw new Error("reset failed");
      $("desired-hdg").value = num("init-hdg");
      $("desired-speed").value = num("init-speed");
      $("desired-depth").value = num("init-depth");
      state.firstFix = true;
      state.lastRegionKey = "";
      setMessage("目標の初期状態を適用し、シナリオを再スタートしました。");
    })
    .catch((error) => setMessage(`設定エラー: ${error.message}`));
});

$("target-form").addEventListener("submit", (event) => {
  event.preventDefault();
  putConfig((next) => {
    next.target.desired_hdg_deg = num("desired-hdg") % 360;
    next.target.hdg_rate_deg_per_sec = num("hdg-rate");
    next.target.desired_through_water_speed_kt = num("desired-speed");
    next.target.speed_rate_kt_per_sec = num("speed-rate");
    next.target.desired_depth_ft = num("desired-depth");
    next.target.depth_rate_ft_per_sec = num("depth-rate");
  })
    .then(() => setMessage("運動指令を反映しました（設定した変化率で変化します）。"))
    .catch((error) => setMessage(`設定エラー: ${error.message}`));
});

$("config-form").addEventListener("submit", (event) => {
  event.preventDefault();
  putConfig((next) => {
    next.max_slant_range_yd = num("max-range");
    next.observer_limit = num("observer-limit");
    next.source.source_frequency_hz = num("source-frequency");
    next.source.shared_recognition_bias_hz = num("frequency-bias");
    next.source.bandwidth_hz = num("source-bandwidth");
    next.source.stability_hz = num("source-stability");
    next.source.stability_correlation_s = num("source-stability-tau");
    next.source.additional_tonals = parseTonals($("additional-tonals").value, next.source.stability_correlation_s);
    next.source.propagation_delay = checked("propagation-delay");
    next.estimator.assumed_stability_hz = num("est-stability");
    next.estimator.use_maneuver_timing = checked("use-maneuver-timing");
    next.bearing.enabled = checked("bearing-enabled");
    next.bearing.sigma_deg = num("bearing-sigma");
    next.bearing.interval_s = num("bearing-interval");
    next.estimator.use_bearing = checked("use-bearing");
    next.estimator.bearing_sigma_deg = num("est-bearing-sigma");
    next.estimator.assumed_bias_sigma_hz = num("bias-sigma");
    next.presence_probability_pct = num("probability");
    next.smoothing_window_seconds = num("smoothing-window");
    next.estimator.particle_count = num("particles");
    next.lloyd.level_noise_db = num("lloyd-noise");
    next.lloyd.noise_correlation_s = num("lloyd-corr");
    next.lloyd.wave_height_rms_m = num("lloyd-wave");
    next.lloyd.path_difference_error_pct = num("lloyd-path-error");
    next.estimator.lloyd_model_error_pct = num("est-lloyd-model-error");
    next.estimator.lloyd_fit_interval_s = num("est-lloyd-interval");
  })
    .then(() => setMessage("観測・推定条件を反映しました。"))
    .catch((error) => setMessage(`設定エラー: ${error.message}`));
});

$("forward-form").addEventListener("submit", (event) => {
  event.preventDefault();
  putConfig((next) => {
    next.forward.enabled = checked("fwd-enabled");
    next.forward.lead_time_s = num("fwd-lead");
    next.forward.min_coverage = num("fwd-min");
    next.forward.ahead_distance_yd = num("fwd-ahead");
    next.forward.lateral_offset_yd = num("fwd-lateral");
    next.forward.observers_per_drop = num("fwd-count");
    next.forward.cooldown_s = num("fwd-cooldown");
    next.forward.strategy = $("fwd-strategy").value;
    next.forward.horizon_s = num("fwd-horizon");
    next.forward.target_error_yd = num("fwd-target");
    next.forward.max_per_drop = num("fwd-max");
    const depths = $("fwd-depths").value.split(/[,\s、]+/).map(Number).filter((d) => Number.isFinite(d) && d >= 0);
    if (depths.length) next.forward.depth_options_ft = depths;
    next.forward.depth_weight = num("fwd-depth-weight");
    next.forward.min_relative_gain = num("fwd-min-gain") / 100;
    next.forward.maneuver_weight = num("fwd-maneuver") / 100;
    next.forward.use_detection_gate = $("fwd-gate").checked;
  })
    .then(() => setMessage("前程配置の設定を反映しました。"))
    .catch((error) => setMessage(`設定エラー: ${error.message}`));
});

$("current-form").addEventListener("submit", (event) => {
  event.preventDefault();
  putConfig((next) => {
    next.current_field.base_velocity.east_kt = num("cur-e");
    next.current_field.base_velocity.north_kt = num("cur-n");
    next.current_field.gradient_per_nm[0][0] = num("g00");
    next.current_field.gradient_per_nm[1][1] = num("g11");
  })
    .then(() => setMessage("潮流を反映しました。"))
    .catch((error) => setMessage(`設定エラー: ${error.message}`));
});

$("placement-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = { position: { latitude: num("place-lat"), longitude: num("place-lon"), depth_ft: num("place-depth") } };
  if ($("place-time").value.trim() !== "") {
    body.planned_tick = tickFromClock($("place-time").value);
    if (body.planned_tick == null) { setMessage("投入時刻を 時:分:秒（例 14:05:30）で入力してください。"); return; }
  }
  const response = await fetch("/api/observers/placements", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (response.ok) {
    const result = await response.json();
    setMessage(state.latestConfig?.layer?.enabled !== false
      ? `配置点を設標者に指示しました（設標待ち ${result.pending_placements} 件）。到着時に観測者が投入されます。`
      : `配置を予約しました（待機 ${result.pending_placements} 件）。`);
  } else {
    setMessage(`配置エラー: ${await response.text()}`);
  }
});

const handler = new Cesium.ScreenSpaceEventHandler(scene.canvas);
handler.setInputAction((movement) => {
  const cartesian = viewer.camera.pickEllipsoid(movement.position, scene.globe.ellipsoid);
  if (!cartesian) return;
  const carto = Cesium.Cartographic.fromCartesian(cartesian);
  const lat = Cesium.Math.toDegrees(carto.latitude).toFixed(4);
  const lon = Cesium.Math.toDegrees(carto.longitude).toFixed(4);
  const purpose = document.querySelector("input[name='dblclick']:checked")?.value || "target";
  if (purpose === "target") {
    $("init-lat").value = lat;
    $("init-lon").value = lon;
    setMessage("目標の初期位置を入力しました。「初期状態を適用して再スタート」で反映します。");
  } else {
    $("place-lat").value = lat;
    $("place-lon").value = lon;
    setMessage("観測者配置の座標を入力しました。");
  }
}, Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);

$("depth-exaggeration").addEventListener("change", () => {
  // tracks are recomputed by the stream worker with the new vertical scale
  stream.setExaggeration(exaggeration());
  state.lastRegionKey = "";
  state.dropKey = "";
  for (const marker of [state.truth, ...Object.values(state.estimates || {}), ...[...state.observers.values()].map((o) => o.marker)]) {
    if (marker) marker.pos = null; // jump, do not glide, to the rescaled position
  }
  if (state.latestSnapshot) render(state.latestSnapshot);
});
for (const id of ["show-truth", "show-online", "show-smoothed", "show-region", "show-voxels", "show-range", "show-bearing", "show-error-line", "show-drops", "show-forces", "show-layer-hud"]) {
  $(id).addEventListener("change", () => {
    state.lastRegionKey = "";
    if (state.latestSnapshot) render(state.latestSnapshot);
  });
}
for (const input of document.querySelectorAll("input[name='panel-mode']")) {
  input.addEventListener("change", () => {
    state.lastRegionKey = "";
    state.history = [];
    if (state.latestSnapshot) render(state.latestSnapshot);
  });
}
if (window.addEventListener) window.addEventListener("resize", drawCharts);

// ================================================================== render loop
function render(snapshot) {
  state.latestSnapshot = snapshot;
  setText("tick", snapshot.tick);
  setText("clock-time", clockAt(snapshot.tick));
  if (snapshot.time_scale != null && snapshot.time_scale !== clock.scale) showClock(snapshot.time_scale);
  noteClockTick(snapshot.tick);
  populateForms(snapshot.config);
  updateRunState(snapshot);
  updateTruth(snapshot.target);
  updateObservers(snapshot.observers, snapshot.doppler, snapshot.config);
  updateBearings(snapshot.bearings || [], snapshot.doppler, snapshot.observers, snapshot.config);
  updateEstimateLayers(snapshot.estimates);
  const estimate = selectedEstimate(snapshot);
  updateRegion(estimate);
  setText("observability", estimate?.observability_status || (snapshot.estimation?.running ? "--" : "推定停止中"));
  setText("position-basis", estimate?.metadata?.position_basis || "--");
  updateComparison(snapshot, estimate);
  updateRelative(snapshot, estimate);
  updateCpa(snapshot.cpa || []);
  updateCurrent(snapshot.current_estimate, snapshot.config);
  updateLloyd(snapshot.lloyd, snapshot.config, snapshot.target);
  updateDeployment(snapshot.deployment);
  updateLayer(snapshot.deployment);
  updateWind(snapshot.deployment);
  updateForces(snapshot, estimate);
  drawCharts();
  if (state.firstFix && snapshot.target) {
    state.firstFix = false;
    initialView();
  }
  scene.requestRender();
}

// ================================================================== stream (Web Worker)
function handleUpdate(result, bytes, parseMs) {
  const t0 = performance.now();
  applyTrackUpdates(result.tracks);
  if (result.timeScale != null) showClock(result.timeScale); // speed changed while paused
  if (result.regionChanged) state.region = result.region;
  if (result.snapshot) {
    for (const estimate of result.snapshot.estimates) {
      estimate.presence_region = state.region || { probability_pct: 0, components: [], disconnected: false };
    }
    if (bytes) {
      const now = performance.now();
      if (anim.lastArrival) anim.interval = Math.min(2000, Math.max(400, 0.8 * anim.interval + 0.2 * (now - anim.lastArrival)));
      anim.lastArrival = now;
    }
    render(result.snapshot);
  }
  const took = performance.now() - t0;
  const ema = (old, value) => (telemetry.updates ? 0.85 * old + 0.15 * value : value);
  if (bytes) {
    telemetry.bytes = ema(telemetry.bytes, bytes);
    telemetry.parseMs = ema(telemetry.parseMs, parseMs);
    telemetry.updateMs = ema(telemetry.updateMs, took);
    telemetry.maxUpdateMs = telemetry.updates > 5 ? Math.max(telemetry.maxUpdateMs, took) : 0;
    telemetry.updates += 1;
  }
  if (!quality.requestedAt) quality.requestedAt = performance.now();
  scene.requestRender();
}

const stream = (() => {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const url = `${protocol}://${location.host}/ws`;
  const onStatus = (connected) => setText("connection-status", connected ? "接続済み" : "再接続中");
  const guard = (fn) => {
    try {
      fn();
    } catch (error) {
      setMessage(`描画エラー: ${error.message}`);
      if (window.console) console.error(error);
    }
  };
  if (typeof Worker !== "undefined") {
    // decoding, delta merge and coordinate conversion run off the rendering thread
    const worker = new Worker("/stream-worker.js");
    telemetry.worker = true;
    worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === "status") onStatus(msg.connected);
      else if (msg.type === "error") setMessage(`受信エラー: ${msg.message}`);
      else if (msg.type === "update") guard(() => handleUpdate(msg.result, msg.bytes, msg.parseMs));
    };
    worker.postMessage({ type: "connect", url, exaggeration: exaggeration() });
    return { setExaggeration: (value) => worker.postMessage({ type: "exaggeration", value }) };
  }
  // fallback: same decoder on the main thread
  const decoder = AquaDecoder.createDecoder();
  decoder.setExaggeration(exaggeration());
  function connect() {
    const socket = new WebSocket(url);
    socket.addEventListener("open", () => onStatus(true));
    socket.addEventListener("message", (event) => {
      const t0 = performance.now();
      guard(() => {
        const result = decoder.decode(JSON.parse(event.data));
        handleUpdate(result, event.data.length, performance.now() - t0);
      });
    });
    socket.addEventListener("close", () => {
      onStatus(false);
      setTimeout(connect, 1500);
    });
    socket.addEventListener("error", () => socket.close());
  }
  connect();
  return { setExaggeration: (value) => guard(() => handleUpdate({ snapshot: null, tracks: decoder.setExaggeration(value) }, 0, 0)) };
})();

window.aquaDrift = { viewer, state, applyView, centerOn, cameraInfo, renderFaults, quality, telemetry, gpu, baseMap, setBaseMap, tracks, anim, updateLayer }; // diagnostics / E2E
