/* global Cesium */
"use strict";

const FT_TO_M = 0.3048;
const YD_TO_M = 0.9144;
const EARTH_R = 6371008.8;
const MAX_TRUTH_POINTS = 4000;
const MAX_HISTORY = 3600;

const $ = (id) => document.getElementById(id);
const checked = (id) => Boolean($(id) && $(id).checked);

const viewer = new Cesium.Viewer("cesiumContainer", {
  animation: false,
  timeline: false,
  baseLayerPicker: false,
  geocoder: false,
  homeButton: true,
  sceneModePicker: true,
  navigationHelpButton: false,
  infoBox: true,
  selectionIndicator: true,
  terrainProvider: new Cesium.EllipsoidTerrainProvider(),
  baseLayer: false,
});
viewer.scene.globe.depthTestAgainstTerrain = false;
if (viewer.scene.globe.translucency) {
  viewer.scene.globe.translucency.enabled = true;
  viewer.scene.globe.translucency.frontFaceAlpha = 0.55;
  viewer.scene.globe.translucency.backFaceAlpha = 0.55;
}
viewer.scene.screenSpaceCameraController.enableCollisionDetection = false;

Cesium.TileMapServiceImageryProvider.fromUrl(
  Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII"),
)
  .then((provider) => viewer.imageryLayers.addImageryProvider(provider))
  .catch(() => setMessage("Natural Earth II の読込みに失敗しました。"));

const COLORS = {
  truth: Cesium.Color.CYAN,
  online: Cesium.Color.ORANGE,
  smoothed: Cesium.Color.LIME,
  region: Cesium.Color.fromCssColorString("#ff4fd8"),
  observer: Cesium.Color.GOLD,
  detecting: Cesium.Color.fromCssColorString("#5dff9d"),
  bearing: Cesium.Color.fromCssColorString("#ffe680"),
  error: Cesium.Color.WHITE,
};

const state = {
  truthTrack: [],
  observerTracks: new Map(),
  observerEntities: new Map(),
  bearingEntities: [],
  regionEntities: [],
  voxelPoints: viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection()),
  estimateEntities: {},
  latestConfig: null,
  latestSnapshot: null,
  formsLoaded: false,
  firstFix: true,
  lastRegionKey: "",
  runKey: "",
  history: [],
  trueCpa: new Map(),
};

// ------------------------------------------------------------------ helpers
function exaggeration() {
  const value = Number($("depth-exaggeration").value);
  return Number.isFinite(value) && value >= 1 ? value : 1;
}
function cart(longitude, latitude, depthFt) {
  return Cesium.Cartesian3.fromDegrees(longitude, latitude, -depthFt * FT_TO_M * exaggeration());
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
function latLonText(position) {
  const ns = position.latitude >= 0 ? "N" : "S";
  const ew = position.longitude >= 0 ? "E" : "W";
  return `${Math.abs(position.latitude).toFixed(5)}°${ns} ${Math.abs(position.longitude).toFixed(5)}°${ew}`;
}
function offsetM(a, b) {
  const meanLat = ((a.latitude + b.latitude) / 2) * Math.PI / 180;
  const north = (b.latitude - a.latitude) * Math.PI / 180 * EARTH_R;
  const east = (b.longitude - a.longitude) * Math.PI / 180 * EARTH_R * Math.cos(meanLat);
  return { east, north };
}
function horizontalYd(a, b) {
  const d = offsetM(a, b);
  return Math.hypot(d.east, d.north) / YD_TO_M;
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

// ------------------------------------------------------------------ tabs
for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    for (const other of document.querySelectorAll(".tab")) other.classList.toggle("active", other === tab);
    for (const panel of document.querySelectorAll(".tab-panel")) {
      panel.classList.toggle("active", panel.id === tab.dataset.tab);
    }
    drawCharts();
  });
}

// ------------------------------------------------------------------ truth
function updateTruth(target) {
  if (!target) return;
  const last = state.truthTrack[state.truthTrack.length - 1];
  if (!last || last.tick !== target.tick) {
    state.truthTrack.push({ tick: target.tick, ...target.position });
    if (state.truthTrack.length > MAX_TRUTH_POINTS) state.truthTrack.shift();
  }
  const show = checked("show-truth");
  const position = cartOf(target.position);
  if (!state.truthEntity) {
    state.truthEntity = viewer.entities.add({
      id: "target-truth",
      name: "目標（真値）",
      position,
      point: { pixelSize: 10, color: COLORS.truth, outlineColor: Cesium.Color.BLACK, outlineWidth: 2 },
      label: { text: "TRUTH", font: "12px sans-serif", fillColor: COLORS.truth, pixelOffset: new Cesium.Cartesian2(0, -18) },
    });
    state.truthLine = viewer.entities.add({
      id: "target-truth-track",
      polyline: { positions: [], width: 2, material: COLORS.truth.withAlpha(0.6) },
    });
  }
  state.truthEntity.position = position;
  state.truthEntity.show = show;
  state.truthLine.polyline.positions = state.truthTrack.map(cartOf);
  state.truthLine.show = show;
}

// ------------------------------------------------------------------ observers
function updateObservers(records, batch, config) {
  const detecting = new Set((batch?.observations || []).filter((o) => o.detected).map((o) => o.observer_id));
  const active = new Set();
  for (const record of records) {
    const observer = record.state;
    const id = observer.observer_id;
    active.add(id);
    const track = state.observerTracks.get(id) || [];
    const last = track[track.length - 1];
    if (!last || last.tick !== observer.tick) track.push({ tick: observer.tick, ...observer.position });
    if (track.length > MAX_TRUTH_POINTS) track.shift();
    state.observerTracks.set(id, track);

    let entities = state.observerEntities.get(id);
    if (!entities) {
      entities = {
        marker: viewer.entities.add({
          id: `observer-${id}`,
          name: `観測者 ${id}`,
          point: { pixelSize: 8, outlineColor: Cesium.Color.BLACK, outlineWidth: 1 },
          label: { text: id, font: "11px sans-serif", fillColor: COLORS.observer, pixelOffset: new Cesium.Cartesian2(0, -14), scale: 0.9 },
        }),
        line: viewer.entities.add({
          id: `observer-track-${id}`,
          polyline: { positions: [], width: 1, material: COLORS.observer.withAlpha(0.45) },
        }),
        range: viewer.entities.add({
          id: `observer-range-${id}`,
          ellipsoid: {
            radii: new Cesium.Cartesian3(1, 1, 1),
            material: COLORS.observer.withAlpha(0.05),
            outline: true,
            outlineColor: COLORS.observer.withAlpha(0.25),
            slicePartitions: 12,
            stackPartitions: 8,
          },
        }),
      };
      state.observerEntities.set(id, entities);
    }
    const position = cartOf(observer.position);
    entities.marker.show = true;
    entities.line.show = true;
    entities.marker.position = position;
    entities.marker.point.color = detecting.has(id) ? COLORS.detecting : COLORS.observer;
    entities.marker.description = `深度 ${fmt(observer.position.depth_ft, 0)} Ft<br>登録 ${record.registered_tick} s / 最終 ${record.last_tick} s`;
    entities.line.polyline.positions = track.map(cartOf);
    const r = config.max_slant_range_yd * YD_TO_M;
    entities.range.position = position;
    entities.range.ellipsoid.radii = new Cesium.Cartesian3(r, r, r * exaggeration());
    entities.range.show = checked("show-range");
  }
  for (const [id, entities] of state.observerEntities) {
    if (active.has(id)) continue;
    entities.marker.point.color = Cesium.Color.GRAY; // evicted/expired: history stays visible
    entities.range.show = false;
  }
  setText("observer-count", String(records.length));
  setText("detecting-count", String(detecting.size));
}

function updateBearings(bearings, config) {
  for (const entity of state.bearingEntities) viewer.entities.remove(entity);
  state.bearingEntities = [];
  if (!checked("show-bearing")) return;
  const length = config.max_slant_range_yd * YD_TO_M;
  for (const item of bearings) {
    const end = destination(item.observer_position, item.bearing_deg, length);
    const depth = item.observer_position.depth_ft;
    state.bearingEntities.push(viewer.entities.add({
      id: `bearing-${item.observer_id}`,
      name: `方位 ${item.observer_id}`,
      description: `観測方位 ${fmt(item.bearing_deg)}°T（${item.tick} s）`,
      polyline: {
        positions: [cartOf(item.observer_position), cart(end.longitude, end.latitude, depth)],
        width: 1.5,
        material: new Cesium.PolylineDashMaterialProperty({ color: COLORS.bearing.withAlpha(0.8), dashLength: 8 }),
      },
    }));
  }
}

// ------------------------------------------------------------------ estimates on the map
function estimateEntities(mode) {
  if (state.estimateEntities[mode]) return state.estimateEntities[mode];
  const color = mode === "ONLINE" ? COLORS.online : COLORS.smoothed;
  const entities = {
    point: viewer.entities.add({
      id: `estimate-${mode}`,
      name: mode === "ONLINE" ? "推定（非更新）" : "推定（更新）",
      point: { pixelSize: 11, color, outlineColor: Cesium.Color.BLACK, outlineWidth: 2 },
      label: { text: mode === "ONLINE" ? "EST" : "", font: "12px sans-serif", fillColor: color, pixelOffset: new Cesium.Cartesian2(0, 20) },
    }),
    track: viewer.entities.add({
      id: `estimate-track-${mode}`,
      polyline: {
        positions: [],
        width: mode === "ONLINE" ? 2 : 3,
        material: mode === "ONLINE"
          ? new Cesium.PolylineDashMaterialProperty({ color: color.withAlpha(0.9), dashLength: 12 })
          : color.withAlpha(0.85),
      },
    }),
  };
  state.estimateEntities[mode] = entities;
  return entities;
}

function updateEstimateLayers(estimates, target) {
  for (const estimate of estimates) {
    const entities = estimateEntities(estimate.mode);
    const visible = checked(estimate.mode === "ONLINE" ? "show-online" : "show-smoothed");
    const has = Boolean(estimate.current_position);
    entities.point.show = visible && has;
    entities.track.show = visible && estimate.track.length > 1;
    if (has) entities.point.position = cartOf(estimate.current_position);
    entities.track.polyline.positions = estimate.track.map((p) => cart(p.longitude, p.latitude, p.depth_ft));
  }
  const selected = estimates.find((e) => e.mode === selectedMode());
  if (!state.errorLine) {
    state.errorLine = viewer.entities.add({
      id: "estimate-error-line",
      name: "推定–真値",
      polyline: { positions: [], width: 1.5, material: COLORS.error.withAlpha(0.7) },
    });
  }
  const showError = checked("show-error-line") && selected?.current_position && target;
  state.errorLine.show = Boolean(showError);
  if (showError) state.errorLine.polyline.positions = [cartOf(selected.current_position), cartOf(target.position)];
}

function updateRegion(estimate) {
  const region = estimate?.presence_region;
  const key = `${estimate?.tick}-${estimate?.mode}-${checked("show-region")}-${checked("show-voxels")}-${exaggeration()}`;
  if (key === state.lastRegionKey) return;
  state.lastRegionKey = key;
  for (const entity of state.regionEntities) viewer.entities.remove(entity);
  state.regionEntities = [];
  state.voxelPoints.removeAll();
  if (!region) return;
  region.components.forEach((component, index) => {
    if (checked("show-region") && component.polygon.length >= 3) {
      state.regionEntities.push(viewer.entities.add({
        id: `region-${index}`,
        name: `推定存在圏 ${index + 1}`,
        description: `確率 ${fmt(component.probability_mass_pct)} %<br>深度 ${fmt(component.min_depth_ft, 0)}–${fmt(component.max_depth_ft, 0)} Ft`,
        polygon: {
          hierarchy: Cesium.Cartesian3.fromDegreesArray(component.polygon.flat()),
          height: -component.min_depth_ft * FT_TO_M * exaggeration(),
          extrudedHeight: -component.max_depth_ft * FT_TO_M * exaggeration(),
          material: COLORS.region.withAlpha(0.12),
          outline: true,
          outlineColor: COLORS.region.withAlpha(0.8),
        },
        label: { text: `${fmt(component.probability_mass_pct, 0)}%`, font: "12px sans-serif", fillColor: COLORS.region },
        position: cartOf(component.centroid),
      }));
    }
    if (checked("show-voxels")) {
      for (const voxel of component.voxels) {
        state.voxelPoints.add({ position: cart(voxel[0], voxel[1], voxel[2]), pixelSize: 3, color: COLORS.region.withAlpha(0.45) });
      }
    }
  });
}

// ------------------------------------------------------------------ estimation control
function updateRunState(snapshot) {
  const control = snapshot.estimation || {};
  const badge = $("run-badge");
  badge.textContent = control.running ? "推定中" : "停止中";
  badge.className = `badge ${control.running ? "running" : "stopped"}`;
  const elapsed = control.running ? snapshot.tick - control.started_tick : (control.stopped_tick ?? snapshot.tick) - control.started_tick;
  setText("run-info", `Run #${control.run_id ?? 0}　開始 ${control.started_tick ?? 0} s　経過 ${Math.max(0, elapsed || 0)} s` +
    (control.running ? "" : `　停止 ${control.stopped_tick ?? "--"} s`));
  $("est-start").disabled = false;
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
  postControl("/api/estimation/start", "推定を開始しました（現在時刻以降の観測を使用）。")
    .catch((error) => setMessage(`推定開始エラー: ${error.message}`));
});
$("est-stop").addEventListener("click", () => {
  postControl("/api/estimation/stop", "推定を停止しました（最後の推定結果を保持表示）。")
    .catch((error) => setMessage(`推定停止エラー: ${error.message}`));
});

// ------------------------------------------------------------------ comparison
function rowHtml(label, est, truth, err, sigma) {
  return `<tr><td>${label}</td><td>${est}</td><td>${truth}</td><td>${err}</td><td>${sigma}</td></tr>`;
}

function updateComparison(snapshot, estimate) {
  const target = snapshot.target;
  const tbody = $("compare-table").querySelector("tbody");
  if (!estimate || !estimate.current_position || !target) {
    tbody.innerHTML = `<tr><td colspan="5">${snapshot.estimation?.running ? "探知待ち（推定前）" : "推定停止中"}</td></tr>`;
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
  const rows = [
    rowHtml("位置", latLonText(ep), latLonText(tp), `${fmt(hErr, 0)} YD`, `${fmt(u.horizontal_major_yd, 0)}×${fmt(u.horizontal_minor_yd, 0)} YD`),
    rowHtml("　東西 / 南北", "", "", `${signed(d.east / YD_TO_M, 0)} / ${signed(d.north / YD_TO_M, 0)} YD`, ""),
    rowHtml("深度", `${fmt(estimate.depth_ft, 0)} Ft`, `${fmt(tp.depth_ft, 0)} Ft`, `${signed(estimate.depth_ft - tp.depth_ft, 0)} Ft`, `${fmt(u.depth_sigma_ft, 0)} Ft`),
    rowHtml("HDG", `${fmt(estimate.hdg_deg)}°`, `${fmt(target.hdg_deg)}°`, `${signed(angleDiff(estimate.hdg_deg, target.hdg_deg))}°`, `${fmt(u.hdg_sigma_deg)}°`),
    rowHtml("COG", `${fmt(estimate.cog_deg)}°`, `${fmt(target.cog_deg)}°`, `${signed(angleDiff(estimate.cog_deg, target.cog_deg))}°`, `${fmt(u.cog_sigma_deg)}°`),
    rowHtml("対水速力", `${fmt(estimate.through_water_speed_kt)} kt`, `${fmt(target.through_water_speed_kt)} kt`, `${signed(estimate.through_water_speed_kt - target.through_water_speed_kt, 2)} kt`, `${fmt(u.through_water_speed_sigma_kt, 2)} kt`),
    rowHtml("対地速力", `${fmt(estimate.ground_speed_kt)} kt`, `${fmt(target.ground_speed_kt)} kt`, `${signed(estimate.ground_speed_kt - target.ground_speed_kt, 2)} kt`, `${fmt(u.ground_speed_sigma_kt, 2)} kt`),
    rowHtml("深度変化率", `${fmt(estimate.vertical_rate_fps, 2)} Ft/s`, `${fmt(trueVertical, 2)} Ft/s`, `${signed(estimate.vertical_rate_fps - trueVertical, 2)} Ft/s`, ""),
    rowHtml("周波数偏り", `${fmt(estimate.source_bias_hz, 3)} Hz`, `${fmt(trueBias, 3)} Hz`, `${signed(estimate.source_bias_hz - trueBias, 3)} Hz`, `${fmt(u.bias_sigma_hz, 3)} Hz`),
  ];
  tbody.innerHTML = rows.join("");

  const region = estimate.presence_region;
  const inside = region.components.find((c) =>
    pointInPolygon(tp.longitude, tp.latitude, c.polygon) && tp.depth_ft >= c.min_depth_ft && tp.depth_ft <= c.max_depth_ft);
  setText("region-check", `真値は推定存在圏（${fmt(region.probability_pct, 0)} %、${region.components.length} 領域）の` +
    (inside ? `内側（確率 ${fmt(inside.probability_mass_pct, 0)} % の領域）` : "外側") +
    `　｜　状態 ${estimate.observability_status}　｜　入力: ${estimate.metadata?.observation_inputs || "--"}`);

  const last = state.history[state.history.length - 1];
  if (!last || last.tick !== estimate.tick) {
    state.history.push({
      tick: estimate.tick,
      h: hErr,
      hs: u.horizontal_major_yd,
      d: estimate.depth_ft - tp.depth_ft,
      ds: u.depth_sigma_ft,
    });
    if (state.history.length > MAX_HISTORY) state.history.shift();
  }
}

function updateRelative(snapshot, estimate) {
  const truth = new Map((snapshot.doppler?.truth || []).map((t) => [t.observer_id, t]));
  const bearings = new Map((snapshot.bearings || []).map((b) => [b.observer_id, b]));
  const estRel = new Map((estimate?.relative || []).map((r) => [r.observer_id, r]));
  const detected = new Set((snapshot.doppler?.observations || []).filter((o) => o.detected).map((o) => o.observer_id));
  const ids = snapshot.observers.map((r) => r.state.observer_id);
  const rows = ids
    .map((id) => ({ id, est: estRel.get(id), tr: truth.get(id), brg: bearings.get(id), det: detected.has(id) }))
    .sort((a, b) => Number(b.det) - Number(a.det) || (a.tr?.slant_range_yd ?? 1e9) - (b.tr?.slant_range_yd ?? 1e9))
    .slice(0, 40)
    .map((r) => `<tr class="${r.det ? "det" : ""}"><td>${r.id}</td><td>${r.det ? "●" : "–"}</td>` +
      `<td>${fmt(r.est?.relative_speed_kt)} / ${fmt(r.tr?.relative_speed_kt)} kt</td>` +
      `<td>${fmt(r.est?.slant_range_yd, 0)} / ${fmt(r.tr?.slant_range_yd, 0)} YD</td>` +
      `<td>${r.brg ? fmt(r.brg.bearing_deg) : "--"} / ${fmt(r.tr?.true_bearing_deg)}°</td></tr>`);
  $("relative-table").querySelector("tbody").innerHTML = rows.join("") || "<tr><td colspan='5'>観測者なし</td></tr>";

  // true CPA (minimum true slant range since the run started)
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
      return `<tr class="${item.final ? "" : "provisional"}"><td>${item.observer_id}#${item.pass_index}</td>` +
        `<td>${fmt(item.cpa_tick, 0)}±${fmt(item.cpa_tick_sigma_s, 0)} / ${tr ? tr.tick : "--"} s</td>` +
        `<td>${fmt(item.cpa_slant_range_yd, 0)}±${fmt(item.cpa_slant_range_sigma_yd, 0)} / ${tr ? fmt(tr.range, 0) : "--"} YD</td>` +
        `<td>${fmt(item.relative_speed_kt, 2)}±${fmt(item.relative_speed_sigma_kt, 2)} kt</td></tr>`;
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
    `<tr><td>流速（推定基準点）</td><td>${dirSpeed(b.east_kt, b.north_kt)}</td><td>${dirSpeed(t.base_velocity.east_kt, t.base_velocity.north_kt)}（基準点）</td></tr>`,
    `<tr><td>∂u/∂x, ∂v/∂y (kt/NM)</td><td>${fmt(g[0][0], 3)}, ${fmt(g[1][1], 3)}</td><td>${fmt(tg[0][0], 3)}, ${fmt(tg[1][1], 3)}</td></tr>`,
    `<tr><td>使用観測者 / 期間</td><td>${current.observer_count} / ${fmt(current.window_seconds / 60, 0)} 分</td><td></td></tr>`,
    `<tr><td>当てはめ残差</td><td>${fmt(current.residual_kt, 3)} kt</td><td></td></tr>`,
  ].join("");
}

// ------------------------------------------------------------------ charts (canvas)
const CHART = {
  series: "#3987e5",
  band: "rgba(160, 180, 200, 0.22)",
  grid: "rgba(143, 174, 203, 0.16)",
  axis: "#8faecb",
  text: "#c9dcef",
};

function niceMax(value) {
  if (!(value > 0)) return 1;
  const exp = Math.pow(10, Math.floor(Math.log10(value)));
  const f = value / exp;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * exp;
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
  canvas._chart.y = y;
  // grid + y labels
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
  // sigma band
  ctx.fillStyle = CHART.band;
  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(x(p.tick), y(opts.sigma(p))) : ctx.moveTo(x(p.tick), y(opts.sigma(p)))));
  for (let i = points.length - 1; i >= 0; i -= 1) {
    const p = points[i];
    ctx.lineTo(x(p.tick), y(opts.symmetric ? -opts.sigma(p) : 0));
  }
  ctx.closePath();
  ctx.fill();
  // series line
  ctx.strokeStyle = CHART.series;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(x(p.tick), y(opts.value(p))) : ctx.moveTo(x(p.tick), y(opts.value(p)))));
  ctx.stroke();
  // direct labels at the right end (text in text ink)
  const lastPoint = points[points.length - 1];
  ctx.fillStyle = CHART.text;
  ctx.textAlign = "left";
  ctx.fillText("誤差", pad.l + w + 3, y(opts.value(lastPoint)) + 3);
  ctx.fillStyle = CHART.axis;
  ctx.fillText("1σ", pad.l + w + 3, y(opts.sigma(lastPoint)) - 4);
  // hover crosshair
  if (canvas._hoverTick != null) {
    const p = points.reduce((best, q) => (Math.abs(q.tick - canvas._hoverTick) < Math.abs(best.tick - canvas._hoverTick) ? q : best));
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
    const px = event.clientX - rect.left;
    const t0 = chart.points[0].tick;
    const t1 = chart.points[chart.points.length - 1].tick;
    canvas._hoverTick = t0 + ((px - chart.pad.l) / chart.w) * (t1 - t0);
    const p = chart.points.reduce((best, q) => (Math.abs(q.tick - canvas._hoverTick) < Math.abs(best.tick - canvas._hoverTick) ? q : best));
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

// ------------------------------------------------------------------ forms
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
  const response = await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(next),
  });
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
    // the base motion starts as constant course/speed/depth = the initial state
    t.desired_hdg_deg = t.initial_hdg_deg;
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
  const response = await fetch("/api/observers/placements", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (response.ok) {
    const result = await response.json();
    setMessage(`配置を予約しました（待機 ${result.pending_placements} 件）。観測者コンテナを追加してください。`);
  } else {
    setMessage(`配置エラー: ${await response.text()}`);
  }
});

const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
handler.setInputAction((movement) => {
  const cartesian = viewer.camera.pickEllipsoid(movement.position, viewer.scene.globe.ellipsoid);
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

for (const id of ["show-truth", "show-online", "show-smoothed", "show-region", "show-voxels",
  "show-range", "show-bearing", "show-error-line", "depth-exaggeration"]) {
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

// ------------------------------------------------------------------ render loop
function render(snapshot) {
  state.latestSnapshot = snapshot;
  setText("tick", snapshot.tick);
  populateForms(snapshot.config);
  updateRunState(snapshot);
  updateTruth(snapshot.target);
  updateObservers(snapshot.observers, snapshot.doppler, snapshot.config);
  updateBearings(snapshot.bearings || [], snapshot.config);
  updateEstimateLayers(snapshot.estimates, snapshot.target);
  const estimate = snapshot.estimates.find((item) => item.mode === selectedMode()) || snapshot.estimates[0];
  updateRegion(estimate);
  setText("observability", estimate?.observability_status || (snapshot.estimation?.running ? "--" : "推定停止中"));
  setText("position-basis", estimate?.metadata?.position_basis || "--");
  updateComparison(snapshot, estimate);
  updateRelative(snapshot, estimate);
  updateCpa(snapshot.cpa || []);
  updateCurrent(snapshot.current_estimate, snapshot.config);
  drawCharts();
  if (state.firstFix && snapshot.target) {
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(snapshot.target.position.longitude, snapshot.target.position.latitude - 0.12, 16000),
      orientation: { heading: 0, pitch: Cesium.Math.toRadians(-45), roll: 0 },
    });
    state.firstFix = false;
  }
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
    }
  });
  socket.addEventListener("close", () => {
    setText("connection-status", "再接続中");
    setTimeout(connect, 1500);
  });
  socket.addEventListener("error", () => socket.close());
}

connect();
