/* global Cesium */
"use strict";

const FT_TO_M = 0.3048;
const YD_TO_M = 0.9144;
const MAX_TRUTH_POINTS = 4000;

const $ = (id) => document.getElementById(id);
const checked = (id) => $(id).checked;

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
};

const state = {
  truthTrack: [],
  observerTracks: new Map(),
  observerEntities: new Map(),
  regionEntities: [],
  voxelPoints: viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection()),
  estimateEntities: {},
  latestConfig: null,
  latestSnapshot: null,
  formsLoaded: false,
  firstFix: true,
  lastRegionKey: "",
};

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

function setText(id, text) {
  const element = $(id);
  if (element) element.textContent = text;
}

function setMessage(text) {
  setText("message", text);
}

function latLonText(position) {
  const lat = position.latitude;
  const lon = position.longitude;
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(5)}°${ns} ${Math.abs(lon).toFixed(5)}°${ew}`;
}

function horizontalYd(a, b) {
  const meanLat = ((a.latitude + b.latitude) / 2) * Math.PI / 180;
  const north = (b.latitude - a.latitude) * Math.PI / 180 * 6371008.8;
  const east = (b.longitude - a.longitude) * Math.PI / 180 * 6371008.8 * Math.cos(meanLat);
  return Math.hypot(east, north) / YD_TO_M;
}

function angleDiff(a, b) {
  return ((a - b + 540) % 360) - 180;
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
      label: {
        text: "TRUTH",
        font: "12px sans-serif",
        fillColor: COLORS.truth,
        pixelOffset: new Cesium.Cartesian2(0, -18),
      },
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

  setText("hdg", `${fmt(target.hdg_deg)}°T`);
  setText("cog", `${fmt(target.cog_deg)}°T`);
  setText("water-speed", `${fmt(target.through_water_speed_kt)} kt`);
  setText("ground-speed", `${fmt(target.ground_speed_kt)} kt`);
  setText("depth", `${fmt(target.position.depth_ft, 0)} Ft`);
}

// ------------------------------------------------------------------ observers
function updateObservers(records, batch, config) {
  const detecting = new Set(
    (batch?.observations || []).filter((item) => item.detected).map((item) => item.observer_id),
  );
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
          label: {
            text: id,
            font: "11px sans-serif",
            fillColor: COLORS.observer,
            pixelOffset: new Cesium.Cartesian2(0, -14),
            scale: 0.9,
          },
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
    const isDetecting = detecting.has(id);
    entities.marker.position = position;
    entities.marker.point.color = isDetecting ? COLORS.detecting : COLORS.observer;
    entities.marker.description = `深度 ${fmt(observer.position.depth_ft, 0)} Ft<br>` +
      `登録 ${record.registered_tick} s / 最終 ${record.last_tick} s`;
    entities.line.polyline.positions = track.map(cartOf);
    const r = config.max_slant_range_yd * YD_TO_M;
    entities.range.position = position;
    entities.range.ellipsoid.radii = new Cesium.Cartesian3(r, r, r * exaggeration());
    entities.range.show = checked("show-range");
  }
  for (const [id, entities] of state.observerEntities) {
    if (active.has(id)) continue;
    // evicted / expired: keep the drift history visible, dim the marker
    entities.marker.point.color = Cesium.Color.GRAY;
    entities.range.show = false;
  }
  setText("observer-count", String(records.length));
  setText("detecting-count", String(detecting.size));
}

// ------------------------------------------------------------------ estimates
function estimateEntities(mode) {
  if (state.estimateEntities[mode]) return state.estimateEntities[mode];
  const color = mode === "ONLINE" ? COLORS.online : COLORS.smoothed;
  const label = mode === "ONLINE" ? "推定（非更新）" : "推定（更新）";
  const entities = {
    point: viewer.entities.add({
      id: `estimate-${mode}`,
      name: label,
      point: { pixelSize: 11, color, outlineColor: Cesium.Color.BLACK, outlineWidth: 2 },
      label: {
        text: mode === "ONLINE" ? "EST" : "",
        font: "12px sans-serif",
        fillColor: color,
        pixelOffset: new Cesium.Cartesian2(0, 20),
      },
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

function updateEstimateLayers(estimates) {
  for (const estimate of estimates) {
    const entities = estimateEntities(estimate.mode);
    const visible = checked(estimate.mode === "ONLINE" ? "show-online" : "show-smoothed");
    const hasPosition = Boolean(estimate.current_position);
    entities.point.show = visible && hasPosition;
    entities.track.show = visible && estimate.track.length > 1;
    if (hasPosition) {
      entities.point.position = cartOf(estimate.current_position);
      entities.point.description = describeEstimate(estimate);
    }
    entities.track.polyline.positions = estimate.track.map((p) => cart(p.longitude, p.latitude, p.depth_ft));
  }
}

function describeEstimate(estimate) {
  const u = estimate.uncertainty;
  return [
    `状態 ${estimate.observability_status}`,
    `深度 ${fmt(estimate.depth_ft, 0)} Ft ±${fmt(u?.depth_sigma_ft, 0)}`,
    `HDG ${fmt(estimate.hdg_deg)}° / COG ${fmt(estimate.cog_deg)}°`,
    `対水 ${fmt(estimate.through_water_speed_kt)} kt / 対地 ${fmt(estimate.ground_speed_kt)} kt`,
  ].join("<br>");
}

function updateRegion(estimate) {
  const region = estimate?.presence_region;
  const showRegion = checked("show-region");
  const showVoxels = checked("show-voxels");
  const key = `${estimate?.tick}-${estimate?.mode}-${showRegion}-${showVoxels}-${exaggeration()}`;
  if (key === state.lastRegionKey) return;
  state.lastRegionKey = key;
  for (const entity of state.regionEntities) viewer.entities.remove(entity);
  state.regionEntities = [];
  state.voxelPoints.removeAll();
  if (!region) return;
  region.components.forEach((component, index) => {
    if (showRegion && component.polygon.length >= 3) {
      const flat = component.polygon.flat();
      state.regionEntities.push(viewer.entities.add({
        id: `region-${index}`,
        name: `推定存在圏 ${index + 1}`,
        description: `確率 ${fmt(component.probability_mass_pct)} %<br>` +
          `深度 ${fmt(component.min_depth_ft, 0)}–${fmt(component.max_depth_ft, 0)} Ft`,
        polygon: {
          hierarchy: Cesium.Cartesian3.fromDegreesArray(flat),
          height: -component.min_depth_ft * FT_TO_M * exaggeration(),
          extrudedHeight: -component.max_depth_ft * FT_TO_M * exaggeration(),
          material: COLORS.region.withAlpha(0.12),
          outline: true,
          outlineColor: COLORS.region.withAlpha(0.8),
        },
        label: {
          text: `${fmt(component.probability_mass_pct, 0)}%`,
          font: "12px sans-serif",
          fillColor: COLORS.region,
        },
        position: cartOf(component.centroid),
      }));
    }
    if (showVoxels) {
      for (const voxel of component.voxels) {
        state.voxelPoints.add({
          position: cart(voxel[0], voxel[1], voxel[2]),
          pixelSize: 3,
          color: COLORS.region.withAlpha(0.45),
        });
      }
    }
  });
}

function selectedMode() {
  const input = document.querySelector("input[name='panel-mode']:checked");
  return input ? input.value : "ONLINE";
}

function updateEstimatePanel(snapshot) {
  const estimate = snapshot.estimates.find((item) => item.mode === selectedMode()) || snapshot.estimates[0];
  updateRegion(estimate);
  if (!estimate) return;
  setText("observability", estimate.observability_status);
  setText("position-basis", estimate.metadata?.position_basis || "--");
  const u = estimate.uncertainty;
  if (!estimate.current_position) {
    ["est-position", "est-depth", "est-vrate", "est-hdg", "est-cog", "est-stw", "est-sog",
      "est-hsigma", "est-region", "est-bias", "est-error"].forEach((id) => setText(id, "--"));
  } else {
    setText("est-position", latLonText(estimate.current_position));
    setText("est-depth", `${fmt(estimate.depth_ft, 0)} ±${fmt(u.depth_sigma_ft, 0)} Ft`);
    setText("est-vrate", `${fmt(estimate.vertical_rate_fps, 2)} Ft/s`);
    setText("est-hdg", `${fmt(estimate.hdg_deg)}° ±${fmt(u.hdg_sigma_deg)}`);
    setText("est-cog", `${fmt(estimate.cog_deg)}° ±${fmt(u.cog_sigma_deg)}`);
    setText("est-stw", `${fmt(estimate.through_water_speed_kt)} ±${fmt(u.through_water_speed_sigma_kt)} kt`);
    setText("est-sog", `${fmt(estimate.ground_speed_kt)} ±${fmt(u.ground_speed_sigma_kt)} kt`);
    setText("est-hsigma", `${fmt(u.horizontal_major_yd, 0)} × ${fmt(u.horizontal_minor_yd, 0)} YD`);
    const region = estimate.presence_region;
    setText("est-region", `${fmt(region.probability_pct, 0)} % / ${region.components.length} 領域`);
    setText("est-bias", `${fmt(estimate.source_bias_hz, 3)} ±${fmt(u.bias_sigma_hz, 3)} Hz`);
    const target = snapshot.target;
    if (target) {
      const dh = horizontalYd(target.position, estimate.current_position);
      const dd = estimate.depth_ft - target.position.depth_ft;
      setText("est-error", `水平 ${fmt(dh, 0)} YD / 深度 ${fmt(dd, 0)} Ft / COG ${fmt(angleDiff(estimate.cog_deg, target.cog_deg))}°`);
    }
  }
  const rows = (estimate.relative || [])
    .slice()
    .sort((a, b) => Number(b.detected) - Number(a.detected) || a.slant_range_yd - b.slant_range_yd)
    .slice(0, 30)
    .map((item) => `<tr class="${item.detected ? "det" : ""}"><td>${item.observer_id}</td>` +
      `<td>${item.detected ? "●" : "–"}</td><td>${fmt(item.relative_speed_kt)} kt</td>` +
      `<td>${fmt(item.slant_range_yd, 0)} YD</td></tr>`);
  $("relative-table").querySelector("tbody").innerHTML = rows.join("") ||
    "<tr><td colspan='4'>推定前</td></tr>";
}

function updateCpa(cpa) {
  const rows = cpa
    .slice()
    .sort((a, b) => b.cpa_tick - a.cpa_tick)
    .slice(0, 30)
    .map((item) => `<tr class="${item.final ? "" : "provisional"}"><td>${item.observer_id}#${item.pass_index}</td>` +
      `<td>${fmt(item.cpa_tick, 0)} ±${fmt(item.cpa_tick_sigma_s, 0)} s</td>` +
      `<td>${fmt(item.cpa_slant_range_yd, 0)} ±${fmt(item.cpa_slant_range_sigma_yd, 0)} YD</td>` +
      `<td>${fmt(item.relative_speed_kt, 2)} ±${fmt(item.relative_speed_sigma_kt, 2)} kt</td></tr>`);
  $("cpa-table").querySelector("tbody").innerHTML = rows.join("") ||
    "<tr><td colspan='4'>最近接通過なし</td></tr>";
}

function updateCurrent(current) {
  if (!current) return;
  const b = current.base_velocity;
  const speed = Math.hypot(b.east_kt, b.north_kt);
  const dir = (Math.atan2(b.east_kt, b.north_kt) * 180 / Math.PI + 360) % 360;
  setText("cur-base", `${fmt(dir, 0)}° ${fmt(speed, 2)} kt`);
  setText("cur-window", `${current.observer_count} / ${fmt(current.window_seconds / 60, 0)} 分`);
  const g = current.gradient_per_nm;
  setText("cur-grad", `[${g[0].slice(0, 2).map((v) => fmt(v, 3)).join(", ")}; ` +
    `${g[1].slice(0, 2).map((v) => fmt(v, 3)).join(", ")}]  残差 ${fmt(current.residual_kt, 3)} kt`);
}

// ------------------------------------------------------------------ forms
function populateForms(config) {
  state.latestConfig = config;
  if (state.formsLoaded) return;
  const t = config.target;
  $("desired-hdg").value = t.desired_hdg_deg;
  $("hdg-rate").value = t.hdg_rate_deg_per_sec;
  $("desired-speed").value = t.desired_through_water_speed_kt;
  $("speed-rate").value = t.speed_rate_kt_per_sec;
  $("desired-depth").value = t.desired_depth_ft;
  $("depth-rate").value = t.depth_rate_ft_per_sec;
  $("max-range").value = config.max_slant_range_yd;
  $("probability").value = config.presence_probability_pct;
  $("smoothing-window").value = config.smoothing_window_seconds;
  $("observer-limit").value = config.observer_limit;
  $("source-frequency").value = config.source.source_frequency_hz;
  $("frequency-bias").value = config.source.shared_recognition_bias_hz;
  $("bias-sigma").value = config.estimator.assumed_bias_sigma_hz;
  $("particles").value = config.estimator.particle_count;
  $("cur-e").value = config.current_field.base_velocity.east_kt;
  $("cur-n").value = config.current_field.base_velocity.north_kt;
  $("g00").value = config.current_field.gradient_per_nm[0][0];
  $("g11").value = config.current_field.gradient_per_nm[1][1];
  $("place-lat").value = t.initial_position.latitude.toFixed(4);
  $("place-lon").value = t.initial_position.longitude.toFixed(4);
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

$("target-form").addEventListener("submit", (event) => {
  event.preventDefault();
  putConfig((next) => {
    next.target.desired_hdg_deg = Number($("desired-hdg").value) % 360;
    next.target.hdg_rate_deg_per_sec = Number($("hdg-rate").value);
    next.target.desired_through_water_speed_kt = Number($("desired-speed").value);
    next.target.speed_rate_kt_per_sec = Number($("speed-rate").value);
    next.target.desired_depth_ft = Number($("desired-depth").value);
    next.target.depth_rate_ft_per_sec = Number($("depth-rate").value);
  })
    .then(() => setMessage("目標運動を反映しました（設定した変化率で変化します）。"))
    .catch((error) => setMessage(`設定エラー: ${error.message}`));
});

$("config-form").addEventListener("submit", (event) => {
  event.preventDefault();
  putConfig((next) => {
    next.max_slant_range_yd = Number($("max-range").value);
    next.presence_probability_pct = Number($("probability").value);
    next.smoothing_window_seconds = Number($("smoothing-window").value);
    next.observer_limit = Number($("observer-limit").value);
    next.source.source_frequency_hz = Number($("source-frequency").value);
    next.source.shared_recognition_bias_hz = Number($("frequency-bias").value);
    next.estimator.assumed_bias_sigma_hz = Number($("bias-sigma").value);
    next.estimator.particle_count = Number($("particles").value);
    next.current_field.base_velocity.east_kt = Number($("cur-e").value);
    next.current_field.base_velocity.north_kt = Number($("cur-n").value);
    next.current_field.gradient_per_nm[0][0] = Number($("g00").value);
    next.current_field.gradient_per_nm[1][1] = Number($("g11").value);
  })
    .then(() => setMessage("観測・推定条件を反映しました。"))
    .catch((error) => setMessage(`設定エラー: ${error.message}`));
});

$("placement-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = {
    position: {
      latitude: Number($("place-lat").value),
      longitude: Number($("place-lon").value),
      depth_ft: Number($("place-depth").value),
    },
  };
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

$("reset").addEventListener("click", async () => {
  const response = await fetch("/api/reset", { method: "POST" });
  if (response.ok) {
    state.truthTrack = [];
    state.observerTracks.clear();
    state.lastRegionKey = "";
    setMessage("ランタイムをリセットしました。履歴は保持されています。");
  }
});

const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
handler.setInputAction((movement) => {
  const cartesian = viewer.camera.pickEllipsoid(movement.position, viewer.scene.globe.ellipsoid);
  if (!cartesian) return;
  const carto = Cesium.Cartographic.fromCartesian(cartesian);
  $("place-lat").value = Cesium.Math.toDegrees(carto.latitude).toFixed(4);
  $("place-lon").value = Cesium.Math.toDegrees(carto.longitude).toFixed(4);
  setMessage("観測者配置の座標を入力しました。");
}, Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);

for (const id of ["show-truth", "show-online", "show-smoothed", "show-region", "show-voxels",
  "show-range", "depth-exaggeration"]) {
  $(id).addEventListener("change", () => {
    state.lastRegionKey = "";
    if (state.latestSnapshot) render(state.latestSnapshot);
  });
}
for (const input of document.querySelectorAll("input[name='panel-mode']")) {
  input.addEventListener("change", () => {
    state.lastRegionKey = "";
    if (state.latestSnapshot) render(state.latestSnapshot);
  });
}

// ------------------------------------------------------------------ render loop
function render(snapshot) {
  state.latestSnapshot = snapshot;
  setText("tick", snapshot.tick);
  populateForms(snapshot.config);
  updateTruth(snapshot.target);
  updateObservers(snapshot.observers, snapshot.doppler, snapshot.config);
  updateEstimateLayers(snapshot.estimates);
  updateEstimatePanel(snapshot);
  updateCpa(snapshot.cpa || []);
  updateCurrent(snapshot.current_estimate);
  if (state.firstFix && snapshot.target) {
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(
        snapshot.target.position.longitude,
        snapshot.target.position.latitude - 0.12,
        16000,
      ),
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
