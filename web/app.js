/* global Cesium */
"use strict";

const FT_TO_M = 0.3048;
const YD_TO_M = 0.9144;
const EARTH_R = 6371008.8;
const REGION_REBUILD_MS = 3000; // presence-region geometry rebuild throttle
const MAX_VOXEL_BOXES = 400; // densest voxels drawn as translucent boxes (overdraw budget)
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
  msaaSamples: 4, // hardware multisample anti-aliasing (WebGL2)
  contextOptions: { webgl: { powerPreference: "high-performance", alpha: false } },
});
const scene = viewer.scene;
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
  error: Cesium.Color.WHITE,
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
  followAnchor: null,
};

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

scene.preRender.addEventListener(() => {
  if (!anim.active.size) return;
  const now = performance.now();
  for (const marker of anim.active) if (!marker.step(now)) anim.active.delete(marker);
  updateErrorLine();
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

function updateBearings(bearings, config) {
  const show = checked("show-bearing");
  const seen = new Set();
  const length = config.max_slant_range_yd * YD_TO_M;
  for (const item of bearings) {
    seen.add(item.observer_id);
    let line = state.bearingLines.get(item.observer_id);
    if (!line) {
      line = { polyline: gpu.lines.add({ positions: [], width: 1.5, material: dashMaterial(COLORS.bearing.withAlpha(0.85), 8) }), key: "" };
      state.bearingLines.set(item.observer_id, line);
    }
    const key = `${item.tick}-${exaggeration()}`;
    if (line.key !== key) { // a bearing changes only every bearing interval
      const end = destination(item.observer_position, item.bearing_deg, length);
      line.polyline.positions = [cartOf(item.observer_position), cart(end.longitude, end.latitude, item.observer_position.depth_ft)];
      line.key = key;
    }
    line.polyline.show = show;
  }
  for (const [id, line] of state.bearingLines) if (!seen.has(id)) line.polyline.show = false;
}

// ================================================================== forward deployment
function updateDeployment(deployment) {
  const status = deployment || { history: [], standby_count: 0, pending_placements: 0 };
  const key = `${status.history.length}-${status.last_deploy_tick}-${checked("show-drops")}-${exaggeration()}`;
  if (key !== state.dropKey) {
    state.dropKey = key;
    gpu.drops.removeAll();
    gpu.dropLabels.removeAll();
    if (checked("show-drops")) {
      for (const record of status.history) {
        for (const position of record.positions) {
          gpu.drops.add({ position: cartOf(position), pixelSize: 9, color: COLORS.drop.withAlpha(0.25), outlineColor: COLORS.drop, outlineWidth: 2 });
        }
        if (record.positions.length) {
          gpu.dropLabels.add({
            position: cartOf(record.positions[0]),
            text: `前程 ${record.tick}s`,
            font: "11px sans-serif",
            fillColor: COLORS.drop,
            pixelOffset: new Cesium.Cartesian2(0, -14),
          });
        }
      }
    }
  }
  if (!tabVisible("tab-display")) return;
  const enabled = state.latestConfig?.forward?.enabled;
  const starting = status.pending_placements > 0 ? "（観測者コンテナを起動中）" : "";
  setHtml($("deploy-status"), `自動前程配置 <b>${enabled ? "有効" : "無効"}</b>　投入待ち ${status.pending_placements}${starting}　` +
    `待機中 ${status.standby_count}　最終配置 ${status.last_deploy_tick ?? "--"} s`);
  const rows = status.history.slice().reverse().map((r) =>
    `<tr><td>${r.tick} s</td><td>${r.positions.length}</td><td>${escapeHtml(r.reason)}</td></tr>`);
  setHtml($("deploy-table").querySelector("tbody"), rows.join("") || "<tr><td colspan='3'>配置なし</td></tr>");
}

$("deploy-now").addEventListener("click", async () => {
  const response = await fetch("/api/deployment/now", { method: "POST" });
  if (response.ok) {
    const result = await response.json();
    setMessage(`推定位置の前程に ${result.deployed} 点の配置を指示しました（待機観測者 ${result.standby}）。`);
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
  const voxelShare = Math.min(1, MAX_VOXEL_BOXES / Math.max(totalVoxels, 1));
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
  const settings = `${estimate?.mode}-${checked("show-region")}-${checked("show-voxels")}-${voxelStyle}-${exaggeration()}-${state.runKey}`;
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
  const boxes = checked("show-voxels") && voxelStyle === "box";
  const built = regionPrimitives(region, boxes);
  replacePrimitive(gpu.region, checked("show-region") ? built.fill : null);
  replacePrimitive(gpu.regionOutline, checked("show-region") ? built.outline : null);
  replacePrimitive(gpu.voxels, boxes ? built.voxels : null);
  if (checked("show-voxels") && voxelStyle === "point") {
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
    camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(p.longitude, p.latitude, heightOf(p.depth_ft) + radius * 2.6),
      orientation: { heading: 0, pitch: -Cesium.Math.PI_OVER_TWO, roll: 0 },
      duration,
    });
  } else if (view === "side") {
    const heading = sideHeading(p);
    const distance = radius * 2.4;
    const from = destination(p, (heading + 180) % 360, distance);
    camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(from.longitude, from.latitude, heightOf(p.depth_ft)),
      orientation: { heading: Cesium.Math.toRadians(heading), pitch: 0, roll: 0 },
      duration,
    });
  } else {
    camera.flyToBoundingSphere(new Cesium.BoundingSphere(cartOf(p), radius), {
      offset: new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-40), radius * 3.2),
      duration,
    });
  }
  for (const button of document.querySelectorAll(".vt[data-view]")) button.classList.toggle("active", button.dataset.view === view);
  state.followAnchor = null;
  const names = { oblique: "斜視", top: "真上（垂直）", side: "水平（側面）" };
  setMessage(`視点：${names[view]}（中心：${focus.source}）`);
  scene.requestRender();
}

function centerOn(prefer) {
  const focus = focusPosition(prefer);
  if (!focus) {
    setMessage("中心に置く対象がまだありません。");
    return;
  }
  const camera = viewer.camera;
  camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
  const target = cartOf(focus.position);
  const range = Cesium.Math.clamp(Cesium.Cartesian3.distance(camera.positionWC, target), 300, viewRadiusM() * 6);
  // keep the current viewing direction, move so that the target is in the screen centre
  camera.flyToBoundingSphere(new Cesium.BoundingSphere(target, 1), {
    offset: new Cesium.HeadingPitchRange(camera.heading, camera.pitch, range),
    duration: 0.8,
  });
  state.followAnchor = null;
  setMessage(`${focus.source}を画面中心にしました。`);
  scene.requestRender();
}

function followEstimate() {
  if (!checked("follow-estimate")) {
    state.followAnchor = null;
    return;
  }
  const focus = focusPosition(centreTarget());
  if (!focus) return;
  if (state.followSource !== focus.source) state.followAnchor = null; // target switched
  state.followSource = focus.source;
  const now = cartOf(focus.position);
  if (state.followAnchor) {
    // translate the camera by the followed position's displacement: orientation and zoom stay
    const delta = Cesium.Cartesian3.subtract(now, state.followAnchor, new Cesium.Cartesian3());
    viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
    Cesium.Cartesian3.add(viewer.camera.position, delta, viewer.camera.position);
  }
  state.followAnchor = now;
}

for (const button of document.querySelectorAll(".vt[data-view]")) {
  button.addEventListener("click", () => applyView(button.dataset.view));
}
$("center-estimate").addEventListener("click", () => centerOn("estimate"));
$("center-truth").addEventListener("click", () => centerOn("truth"));
$("follow-estimate").addEventListener("change", () => {
  state.followAnchor = null;
  if (checked("follow-estimate")) centerOn(centreTarget());
});
$("center-target").addEventListener("change", () => {
  state.followAnchor = null;
  centerOn(centreTarget());
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
  { name: "高", res: Math.min(window.devicePixelRatio || 1, 2), msaa: 4, fxaa: true },
  { name: "中", res: 1.0, msaa: 2, fxaa: true },
  { name: "低", res: 0.75, msaa: 1, fxaa: true },
  { name: "最低", res: 0.5, msaa: 1, fxaa: false },
];
const quality = { level: -1, lastFrame: 0, ema: 0, slowMs: 0, fastMs: 0 };
const TARGET_INTERVAL = 1000 / TARGET_FPS;

function applyQuality(level) {
  if (level === quality.level) return;
  quality.level = level;
  const q = QUALITY_LEVELS[level];
  viewer.resolutionScale = q.res;
  scene.msaaSamples = q.msaa;
  if (scene.postProcessStages && scene.postProcessStages.fxaa) scene.postProcessStages.fxaa.enabled = q.fxaa;
  telemetry.quality = q.name;
  scene.requestRender();
}
applyQuality(0);

scene.postRender.addEventListener(() => {
  const now = performance.now();
  const dt = now - quality.lastFrame;
  quality.lastFrame = now;
  if (dt > 250) return; // idle (on-demand rendering): no frame-rate information
  quality.ema = quality.ema ? 0.9 * quality.ema + 0.1 * dt : dt;
  telemetry.frameIntervalMs = quality.ema;
  quality.slowMs = quality.ema > 1.6 * TARGET_INTERVAL ? quality.slowMs + dt : 0;
  quality.fastMs = quality.ema < 1.15 * TARGET_INTERVAL ? quality.fastMs + dt : 0;
  if ($("quality").value === "auto") {
    if (quality.slowMs > 1500 && quality.level < QUALITY_LEVELS.length - 1) {
      applyQuality(quality.level + 1);
      quality.slowMs = 0;
    } else if (quality.fastMs > 6000 && quality.level > 0) {
      applyQuality(quality.level - 1);
      quality.fastMs = 0;
    }
  }
  // smooth display needs ~10 fps at least; otherwise fall back to 1 Hz jumps automatically
  if (anim.enabled && anim.active.size && quality.ema > 100) {
    anim.slowSince = anim.slowSince || now;
    if (now - anim.slowSince > 2000) setSmooth(false, "描画が追いつかないため、なめらか表示を自動で停止しました（品質を下げるか、GPUの有効化を確認してください）。");
  } else {
    anim.slowSince = null;
  }
  if (anim.active.size) scene.requestRender(); // keep animating until markers arrive
  telemetry.smooth = anim.enabled;
});

$("quality").addEventListener("change", () => {
  const value = $("quality").value;
  quality.slowMs = quality.fastMs = 0;
  applyQuality(value === "auto" ? 0 : Number(value));
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
  setText("run-info", `Run #${control.run_id ?? 0}　開始 ${control.started_tick ?? 0} s　経過 ${Math.max(0, end - (control.started_tick ?? 0))} s` +
    (control.running ? "" : `　停止 ${control.stopped_tick ?? "--"} s`));
  $("est-stop").disabled = !control.running;
  const key = `${snapshot.generation}-${control.run_id}`;
  if (key !== state.runKey) {
    state.runKey = key;
    state.history = [];
    state.trueCpa.clear();
    if (snapshot.generation !== state.generation) {
      state.generation = snapshot.generation;
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
    `<br><small>入力：${escapeHtml(estimate.metadata?.observation_inputs || "--")}</small>`);
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
        `<td>${pair(`${fmt(item.cpa_tick, 0)}±${fmt(item.cpa_tick_sigma_s, 0)} s`, `${tr ? tr.tick : "--"} s`)}</td>` +
        `<td>${pair(`${fmt(item.cpa_slant_range_yd, 0)}±${fmt(item.cpa_slant_range_sigma_yd, 0)} YD`, `${tr ? fmt(tr.range, 0) : "--"} YD`)}</td>` +
        `<td>${fmt(item.relative_speed_kt, 2)}<br><small>±${fmt(item.relative_speed_sigma_kt, 2)} kt</small></td></tr>`;
    });
  setHtml($("cpa-table").querySelector("tbody"), rows.join("") || "<tr><td colspan='4'>最近接通過なし</td></tr>");
}

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
  ctx.fillText(`${t0} s`, pad.l, height - 5);
  ctx.textAlign = "right";
  ctx.fillText(`${t1} s`, pad.l + w, height - 5);
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
    tip.textContent = `${p.tick} s　誤差 ${signed(opts.value(p), 0)} ${opts.unit}　1σ ${fmt(opts.sigma(p), 0)} ${opts.unit}`;
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
  $("use-bearing").checked = config.estimator.use_bearing;
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
    next.bearing.enabled = checked("bearing-enabled");
    next.bearing.sigma_deg = num("bearing-sigma");
    next.bearing.interval_s = num("bearing-interval");
    next.current_field.base_velocity.east_kt = num("cur-e");
    next.current_field.base_velocity.north_kt = num("cur-n");
    next.current_field.gradient_per_nm[0][0] = num("g00");
    next.current_field.gradient_per_nm[1][1] = num("g11");
    next.estimator.use_bearing = checked("use-bearing");
    next.estimator.bearing_sigma_deg = num("est-bearing-sigma");
    next.estimator.assumed_bias_sigma_hz = num("bias-sigma");
    next.presence_probability_pct = num("probability");
    next.smoothing_window_seconds = num("smoothing-window");
    next.estimator.particle_count = num("particles");
    next.forward.enabled = checked("fwd-enabled");
    next.forward.lead_time_s = num("fwd-lead");
    next.forward.min_coverage = num("fwd-min");
    next.forward.ahead_distance_yd = num("fwd-ahead");
    next.forward.lateral_offset_yd = num("fwd-lateral");
    next.forward.observers_per_drop = num("fwd-count");
    next.forward.cooldown_s = num("fwd-cooldown");
  })
    .then(() => setMessage("観測・推定条件を反映しました。"))
    .catch((error) => setMessage(`設定エラー: ${error.message}`));
});

$("placement-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = { position: { latitude: num("place-lat"), longitude: num("place-lon"), depth_ft: num("place-depth") } };
  const response = await fetch("/api/observers/placements", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (response.ok) {
    const result = await response.json();
    setMessage(`配置を予約しました（待機 ${result.pending_placements} 件）。観測者コンテナを追加してください。`);
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
for (const id of ["show-truth", "show-online", "show-smoothed", "show-region", "show-voxels", "show-range", "show-bearing", "show-error-line", "show-drops"]) {
  $(id).addEventListener("change", () => {
    state.lastRegionKey = "";
    state.followAnchor = null;
    if (state.latestSnapshot) render(state.latestSnapshot);
  });
}
for (const input of document.querySelectorAll("input[name='panel-mode']")) {
  input.addEventListener("change", () => {
    state.lastRegionKey = "";
    state.history = [];
    state.followAnchor = null;
    if (state.latestSnapshot) render(state.latestSnapshot);
  });
}
if (window.addEventListener) window.addEventListener("resize", drawCharts);

// ================================================================== render loop
function render(snapshot) {
  state.latestSnapshot = snapshot;
  setText("tick", snapshot.tick);
  populateForms(snapshot.config);
  updateRunState(snapshot);
  updateTruth(snapshot.target);
  updateObservers(snapshot.observers, snapshot.doppler, snapshot.config);
  updateBearings(snapshot.bearings || [], snapshot.config);
  updateEstimateLayers(snapshot.estimates);
  const estimate = selectedEstimate(snapshot);
  updateRegion(estimate);
  setText("observability", estimate?.observability_status || (snapshot.estimation?.running ? "--" : "推定停止中"));
  setText("position-basis", estimate?.metadata?.position_basis || "--");
  updateComparison(snapshot, estimate);
  updateRelative(snapshot, estimate);
  updateCpa(snapshot.cpa || []);
  updateCurrent(snapshot.current_estimate, snapshot.config);
  updateDeployment(snapshot.deployment);
  drawCharts();
  followEstimate();
  if (state.firstFix && snapshot.target) {
    state.firstFix = false;
    applyView("oblique");
  }
  scene.requestRender();
}

// ================================================================== stream (Web Worker)
function handleUpdate(result, bytes, parseMs) {
  const t0 = performance.now();
  applyTrackUpdates(result.tracks);
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

window.aquaDrift = { viewer, state, applyView, centerOn, telemetry, gpu, baseMap, setBaseMap, tracks, anim }; // diagnostics / E2E
