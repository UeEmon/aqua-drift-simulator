/* global Cesium */
"use strict";

const FT_TO_M = 0.3048;
const YD_TO_M = 0.9144;
const EARTH_R = 6371008.8;
const MAX_TRUTH_POINTS = 4000;
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

Cesium.TileMapServiceImageryProvider.fromUrl(Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII"))
  .then((provider) => {
    viewer.imageryLayers.addImageryProvider(provider);
    scene.requestRender();
  })
  .catch(() => setMessage("Natural Earth II の読込みに失敗しました。"));

const COLORS = {
  truth: Cesium.Color.CYAN,
  online: Cesium.Color.ORANGE,
  smoothed: Cesium.Color.LIME,
  region: Cesium.Color.fromCssColorString("#ff4fd8"),
  observer: Cesium.Color.GOLD,
  detecting: Cesium.Color.fromCssColorString("#5dff9d"),
  inactive: Cesium.Color.GRAY,
  bearing: Cesium.Color.fromCssColorString("#ffe680"),
  error: Cesium.Color.WHITE,
};

// GPU-batched primitive collections: one draw call per collection instead of one per entity
const gpu = {
  lines: scene.primitives.add(new Cesium.PolylineCollection()),
  points: scene.primitives.add(new Cesium.PointPrimitiveCollection()),
  labels: scene.primitives.add(new Cesium.LabelCollection()),
  regionLabels: scene.primitives.add(new Cesium.LabelCollection()),
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
  truthTrack: [],
  observerTracks: new Map(),
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
    drawCharts();
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

// ================================================================== truth
function updateTruth(target) {
  if (!target) return;
  const last = state.truthTrack[state.truthTrack.length - 1];
  if (!last || last.tick !== target.tick) {
    state.truthTrack.push({ tick: target.tick, ...target.position });
    if (state.truthTrack.length > MAX_TRUTH_POINTS) state.truthTrack.shift();
  }
  const show = checked("show-truth");
  if (!state.truth) {
    state.truth = {
      point: gpu.points.add({ pixelSize: 10, color: COLORS.truth, outlineColor: Cesium.Color.BLACK, outlineWidth: 2, id: "truth" }),
      label: gpu.labels.add({ text: "TRUTH", font: "12px sans-serif", fillColor: COLORS.truth, pixelOffset: new Cesium.Cartesian2(0, -18) }),
      track: gpu.lines.add({ positions: [], width: 2, material: colorMaterial(COLORS.truth.withAlpha(0.6)) }),
    };
  }
  const position = cartOf(target.position);
  state.truth.point.position = position;
  state.truth.label.position = position;
  state.truth.point.show = state.truth.label.show = state.truth.track.show = show;
  if (state.truthTrack.length > 1) state.truth.track.positions = state.truthTrack.map(cartOf);
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
    const track = state.observerTracks.get(id) || [];
    const last = track[track.length - 1];
    if (!last || last.tick !== observer.tick) track.push({ tick: observer.tick, ...observer.position });
    if (track.length > MAX_TRUTH_POINTS) track.shift();
    state.observerTracks.set(id, track);

    let item = state.observers.get(id);
    if (!item) {
      item = {
        point: gpu.points.add({ pixelSize: 8, outlineColor: Cesium.Color.BLACK, outlineWidth: 1, id: `observer-${id}` }),
        label: gpu.labels.add({ text: id, font: "11px sans-serif", fillColor: COLORS.observer, pixelOffset: new Cesium.Cartesian2(0, -14), scale: 0.9 }),
        track: gpu.lines.add({ positions: [], width: 1, material: colorMaterial(COLORS.observer.withAlpha(0.45)) }),
        range: null,
      };
      state.observers.set(id, item);
    }
    const position = cartOf(observer.position);
    item.point.position = position;
    item.label.position = position;
    item.point.color = detecting.has(id) ? COLORS.detecting : COLORS.observer;
    if (track.length > 1) item.track.positions = track.map(cartOf);
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
    if (active.has(id)) continue;
    item.point.color = COLORS.inactive; // evicted / expired: drift history stays visible
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
      line = gpu.lines.add({ positions: [], width: 1.5, material: dashMaterial(COLORS.bearing.withAlpha(0.85), 8) });
      state.bearingLines.set(item.observer_id, line);
    }
    const end = destination(item.observer_position, item.bearing_deg, length);
    line.positions = [cartOf(item.observer_position), cart(end.longitude, end.latitude, item.observer_position.depth_ft)];
    line.show = show;
  }
  for (const [id, line] of state.bearingLines) if (!seen.has(id)) line.show = false;
}

// ================================================================== estimates
function estimateGraphics(mode) {
  state.estimates = state.estimates || {};
  if (state.estimates[mode]) return state.estimates[mode];
  const color = mode === "ONLINE" ? COLORS.online : COLORS.smoothed;
  const graphics = {
    point: gpu.points.add({ pixelSize: 11, color, outlineColor: Cesium.Color.BLACK, outlineWidth: 2, id: `estimate-${mode}` }),
    label: gpu.labels.add({ text: mode === "ONLINE" ? "EST" : "", font: "12px sans-serif", fillColor: color, pixelOffset: new Cesium.Cartesian2(0, 20) }),
    track: gpu.lines.add({
      positions: [],
      width: mode === "ONLINE" ? 2 : 3,
      material: mode === "ONLINE" ? dashMaterial(color.withAlpha(0.9), 12) : colorMaterial(color.withAlpha(0.85)),
    }),
  };
  state.estimates[mode] = graphics;
  return graphics;
}

function updateEstimateLayers(estimates, target) {
  for (const mode of ["ONLINE", "SMOOTHED"]) {
    const graphics = estimateGraphics(mode);
    const estimate = estimates.find((e) => e.mode === mode);
    const visible = checked(mode === "ONLINE" ? "show-online" : "show-smoothed");
    const has = Boolean(estimate?.current_position);
    graphics.point.show = graphics.label.show = visible && has;
    graphics.track.show = visible && Boolean(estimate) && estimate.track.length > 1;
    if (has) {
      const position = cartOf(estimate.current_position);
      graphics.point.position = position;
      graphics.label.position = position;
    }
    if (estimate && estimate.track.length > 1) {
      graphics.track.positions = estimate.track.map((p) => cart(p.longitude, p.latitude, p.depth_ft));
    }
  }
  if (!state.errorLine) {
    state.errorLine = gpu.lines.add({ positions: [], width: 1.5, material: colorMaterial(COLORS.error.withAlpha(0.75)) });
  }
  const selected = estimates.find((e) => e.mode === selectedMode());
  const showError = Boolean(checked("show-error-line") && selected?.current_position && target);
  state.errorLine.show = showError;
  if (showError) state.errorLine.positions = [cartOf(selected.current_position), cartOf(target.position)];
}

// ---------------------------------------------------------------- presence region (batched GPU geometry)
const MAX_PENDING_MS = 4000;
const telemetry = { frames: 0, pendingDropped: 0, swaps: 0 };

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
  // async geometry is built in web workers; poll at a modest rate until it is uploaded, then swap
  const now = performance.now();
  const waiting = [gpu.region, gpu.regionOutline, gpu.voxels].map((slot) => swapWhenReady(slot, now)).some(Boolean);
  if (waiting && !pendingRenderScheduled) {
    pendingRenderScheduled = true;
    setTimeout(() => {
      pendingRenderScheduled = false;
      scene.requestRender();
    }, 50);
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

function regionPrimitives(region) {
  const fill = [];
  const outline = [];
  const voxels = [];
  const exag = exaggeration();
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
    const n = component.voxels.length;
    const size = component.voxel_size_yd * YD_TO_M * 0.9;
    const tall = component.voxel_height_ft * FT_TO_M * exag * 0.9;
    if (!(size > 0 && tall > 0)) return;
    component.voxels.forEach((voxel, rank) => {
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
  const key = `${estimate?.tick}-${estimate?.mode}-${checked("show-region")}-${checked("show-voxels")}-${exaggeration()}-${state.runKey}`;
  if (key === state.lastRegionKey) return;
  state.lastRegionKey = key;
  gpu.regionLabels.removeAll();
  if (!region || !region.components.length) {
    replacePrimitive(gpu.region, null);
    replacePrimitive(gpu.regionOutline, null);
    replacePrimitive(gpu.voxels, null);
    return;
  }
  const built = regionPrimitives(region);
  replacePrimitive(gpu.region, checked("show-region") ? built.fill : null);
  replacePrimitive(gpu.regionOutline, checked("show-region") ? built.outline : null);
  replacePrimitive(gpu.voxels, checked("show-voxels") ? built.voxels : null);
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

function focusPosition(prefer = "estimate") {
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
  const focus = focusPosition("estimate");
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
  const estimate = selectedEstimate();
  if (!estimate?.current_position) return;
  const now = cartOf(estimate.current_position);
  if (state.followAnchor) {
    // translate the camera by the estimate's displacement: orientation and zoom stay as set
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
  if (checked("follow-estimate")) centerOn("estimate");
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
      state.truthTrack = [];
      state.observerTracks.clear();
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
    container.innerHTML = `<p class="empty">${snapshot.estimation?.running ? "探知待ち（推定前）" : "推定停止中"}</p>`;
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
  container.innerHTML = cards.join("");

  const region = estimate.presence_region;
  const inside = region.components.find((c) =>
    pointInPolygon(tp.longitude, tp.latitude, c.polygon) && tp.depth_ft >= c.min_depth_ft && tp.depth_ft <= c.max_depth_ft);
  $("region-check").innerHTML = `<b>${escapeHtml(estimate.observability_status)}</b>　真値は推定存在圏（${fmt(region.probability_pct, 0)} %・${region.components.length} 領域）の` +
    (inside ? `<span class="s-good">内側</span>` : `<span class="s-bad">外側</span>`) +
    `<br><small>入力：${escapeHtml(estimate.metadata?.observation_inputs || "--")}</small>`;

  const last = state.history[state.history.length - 1];
  if (!last || last.tick !== estimate.tick) {
    state.history.push({ tick: estimate.tick, h: hErr, hs: u.horizontal_major_yd, d: depthErr, ds: u.depth_sigma_ft });
    if (state.history.length > MAX_HISTORY) state.history.shift();
  }
}

function pair(a, b) {
  return `<span class="est">${a}</span><span class="tru">${b}</span>`;
}

function updateRelative(snapshot, estimate) {
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
  $("relative-table").querySelector("tbody").innerHTML = rows.join("") || "<tr><td colspan='4'>観測者なし</td></tr>";
  for (const t of snapshot.doppler?.truth || []) {
    const best = state.trueCpa.get(t.observer_id);
    if (!best || t.slant_range_yd < best.range) state.trueCpa.set(t.observer_id, { range: t.slant_range_yd, tick: t.tick });
  }
}

function updateCpa(cpa) {
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
  $("cpa-table").querySelector("tbody").innerHTML = rows.join("") || "<tr><td colspan='4'>最近接通過なし</td></tr>";
}

function updateCurrent(current, config) {
  const tbody = $("current-table").querySelector("tbody");
  const t = config.current_field;
  const dirSpeed = (e, n) => `${fmt((Math.atan2(e, n) * 180 / Math.PI + 360) % 360, 0)}° ${fmt(Math.hypot(e, n), 2)} kt`;
  if (!current) {
    tbody.innerHTML = `<tr><td>基準流速</td><td>--</td><td>${dirSpeed(t.base_velocity.east_kt, t.base_velocity.north_kt)}</td></tr>`;
    return;
  }
  const b = current.base_velocity;
  const g = current.gradient_per_nm;
  const tg = t.gradient_per_nm;
  tbody.innerHTML = [
    `<tr><td>流速</td><td>${dirSpeed(b.east_kt, b.north_kt)}</td><td>${dirSpeed(t.base_velocity.east_kt, t.base_velocity.north_kt)}</td></tr>`,
    `<tr><td>∂u/∂x, ∂v/∂y<br><small>kt/NM</small></td><td>${fmt(g[0][0], 3)}, ${fmt(g[1][1], 3)}</td><td>${fmt(tg[0][0], 3)}, ${fmt(tg[1][1], 3)}</td></tr>`,
    `<tr><td>観測者 / 期間</td><td>${current.observer_count} / ${fmt(current.window_seconds / 60, 0)} 分</td><td>--</td></tr>`,
    `<tr><td>当てはめ残差</td><td>${fmt(current.residual_kt, 3)} kt</td><td>--</td></tr>`,
  ].join("");
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

for (const id of ["show-truth", "show-online", "show-smoothed", "show-region", "show-voxels", "show-range", "show-bearing", "show-error-line", "depth-exaggeration"]) {
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
  updateEstimateLayers(snapshot.estimates, snapshot.target);
  const estimate = selectedEstimate(snapshot);
  updateRegion(estimate);
  setText("observability", estimate?.observability_status || (snapshot.estimation?.running ? "--" : "推定停止中"));
  setText("position-basis", estimate?.metadata?.position_basis || "--");
  updateComparison(snapshot, estimate);
  updateRelative(snapshot, estimate);
  updateCpa(snapshot.cpa || []);
  updateCurrent(snapshot.current_estimate, snapshot.config);
  drawCharts();
  followEstimate();
  if (state.firstFix && snapshot.target) {
    state.firstFix = false;
    applyView("oblique");
  }
  scene.requestRender();
}

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws`);
  socket.addEventListener("open", () => setText("connection-status", "接続済み"));
  socket.addEventListener("message", (event) => {
    try {
      render(JSON.parse(event.data));
    } catch (error) {
      setMessage(`描画エラー: ${error.message}`);
      if (window.console) console.error(error);
    }
  });
  socket.addEventListener("close", () => {
    setText("connection-status", "再接続中");
    setTimeout(connect, 1500);
  });
  socket.addEventListener("error", () => socket.close());
}

window.aquaDrift = { viewer, state, applyView, centerOn, telemetry, gpu }; // diagnostics / E2E
connect();
