/* global Cesium */

const FT_TO_M = 0.3048;
const YD_TO_M = 0.9144;
const MAX_TRACK_POINTS = 1200;

const viewer = new Cesium.Viewer("cesiumContainer", {
  animation: true,
  timeline: true,
  baseLayerPicker: false,
  geocoder: false,
  homeButton: true,
  sceneModePicker: true,
  navigationHelpButton: false,
  infoBox: true,
  selectionIndicator: true,
  terrainProvider: new Cesium.EllipsoidTerrainProvider(),
  imageryProvider: false,
});

viewer.scene.globe.depthTestAgainstTerrain = false;
if (viewer.scene.globe.translucency) {
  viewer.scene.globe.translucency.enabled = true;
  viewer.scene.globe.translucency.frontFaceAlpha = 0.72;
}

Cesium.TileMapServiceImageryProvider.fromUrl("/cesium/Assets/Textures/NaturalEarthII")
  .then((provider) => viewer.imageryLayers.addImageryProvider(provider))
  .catch(() => setMessage("Natural Earth IIの読込みに失敗しました。"));

const targetTrack = [];
const observerTracks = new Map();
const observerEntities = new Map();
const presencePoints = viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
let targetEntity;
let targetTrackEntity;
let configLoaded = false;
let firstFix = true;
let latestConfig;

function positionOf(position) {
  return Cesium.Cartesian3.fromDegrees(
    position.longitude,
    position.latitude,
    -position.depth_ft * FT_TO_M,
  );
}

function fmt(value, digits = 1) {
  return value == null ? "--" : Number(value).toFixed(digits);
}

function setText(id, text) {
  document.getElementById(id).textContent = text;
}

function setMessage(text) {
  setText("message", text);
}

function updateTarget(target) {
  if (!target) return;
  const position = positionOf(target.position);
  targetTrack.push(position);
  if (targetTrack.length > MAX_TRACK_POINTS) targetTrack.shift();

  if (!targetEntity) {
    targetEntity = viewer.entities.add({
      id: "target-truth",
      name: "Synthetic target truth",
      position,
      point: { pixelSize: 12, color: Cesium.Color.CYAN, outlineColor: Cesium.Color.BLACK, outlineWidth: 2 },
      label: {
        text: "TARGET",
        font: "13px sans-serif",
        fillColor: Cesium.Color.CYAN,
        pixelOffset: new Cesium.Cartesian2(0, -20),
      },
    });
    targetTrackEntity = viewer.entities.add({
      id: "target-track",
      polyline: { positions: targetTrack, width: 2, material: Cesium.Color.CYAN.withAlpha(0.7) },
    });
  } else {
    targetEntity.position = position;
    targetTrackEntity.polyline.positions = targetTrack.slice();
  }
  targetEntity.show = document.getElementById("show-truth").checked;
  targetTrackEntity.show = targetEntity.show;

  setText("hdg", `${fmt(target.hdg_deg)}°T`);
  setText("cog", `${fmt(target.cog_deg)}°T`);
  setText("water-speed", `${fmt(target.through_water_speed_kt)} kt`);
  setText("ground-speed", `${fmt(target.ground_speed_kt)} kt`);
  setText("depth", `${fmt(target.position.depth_ft, 0)} Ft`);

  if (firstFix) {
    viewer.camera.flyTo({ destination: Cesium.Cartesian3.fromDegrees(
      target.position.longitude,
      target.position.latitude,
      18000,
    ) });
    firstFix = false;
  }
}

function updateObservers(records) {
  for (const record of records) {
    const observer = record.state;
    const id = observer.observer_id;
    const position = positionOf(observer.position);
    const track = observerTracks.get(id) || [];
    track.push(position);
    if (track.length > MAX_TRACK_POINTS) track.shift();
    observerTracks.set(id, track);

    let entities = observerEntities.get(id);
    if (!entities) {
      const marker = viewer.entities.add({
        id: `observer-${id}`,
        name: `Observer ${id}`,
        position,
        point: { pixelSize: 8, color: Cesium.Color.GOLD, outlineColor: Cesium.Color.BLACK, outlineWidth: 1 },
        label: {
          text: id,
          font: "11px sans-serif",
          fillColor: Cesium.Color.GOLD,
          pixelOffset: new Cesium.Cartesian2(0, -14),
        },
      });
      const line = viewer.entities.add({
        id: `observer-track-${id}`,
        polyline: { positions: track, width: 1, material: Cesium.Color.GOLD.withAlpha(0.45) },
      });
      entities = { marker, line };
      observerEntities.set(id, entities);
    } else {
      entities.marker.position = position;
      entities.line.polyline.positions = track.slice();
    }
  }
  setText("observer-count", String(records.length));
}

function sampleShell(center, radiusYd, color) {
  const origin = positionOf(center);
  const transform = Cesium.Transforms.eastNorthUpToFixedFrame(origin);
  const radius = radiusYd * YD_TO_M;
  for (let latitudeIndex = -3; latitudeIndex <= 3; latitudeIndex += 1) {
    const latitude = latitudeIndex * Math.PI / 12;
    const ringRadius = radius * Math.cos(latitude);
    const up = radius * Math.sin(latitude);
    for (let longitudeIndex = 0; longitudeIndex < 18; longitudeIndex += 1) {
      const longitude = longitudeIndex * 2 * Math.PI / 18;
      const local = new Cesium.Cartesian3(
        ringRadius * Math.cos(longitude),
        ringRadius * Math.sin(longitude),
        up,
      );
      const world = Cesium.Matrix4.multiplyByPoint(transform, local, new Cesium.Cartesian3());
      presencePoints.add({ position: world, pixelSize: 2.5, color });
    }
  }
}

function updateEstimates(estimates) {
  presencePoints.removeAll();
  let status = "NO_OBSERVATION";
  let relativeSpeed;
  for (const estimate of estimates) {
    const visible = estimate.mode === "ONLINE"
      ? document.getElementById("show-online").checked
      : document.getElementById("show-smoothed").checked;
    if (!visible) continue;
    status = estimate.observability_status;
    relativeSpeed = estimate.relative_speed_kt ?? relativeSpeed;
    const color = estimate.mode === "ONLINE"
      ? Cesium.Color.ORANGE.withAlpha(0.55)
      : Cesium.Color.LIME.withAlpha(0.45);
    for (const component of estimate.presence_region.components.slice(0, 12)) {
      sampleShell(component.center, component.radius_yd, color);
    }
  }
  setText("observability", status);
  setText("relative-speed", relativeSpeed == null ? "--" : `${fmt(relativeSpeed)} kt`);
}

function populateConfig(config) {
  latestConfig = config;
  if (configLoaded) return;
  document.getElementById("max-range").value = config.max_slant_range_yd;
  document.getElementById("desired-hdg").value = config.target.desired_hdg_deg;
  document.getElementById("desired-speed").value = config.target.desired_through_water_speed_kt;
  document.getElementById("desired-depth").value = config.target.desired_depth_ft;
  document.getElementById("probability").value = config.presence_probability_pct;
  document.getElementById("frequency-bias").value = config.source.shared_recognition_bias_hz;
  document.getElementById("smoothing-window").value = config.smoothing_window_seconds;
  configLoaded = true;
}

function render(snapshot) {
  setText("tick", snapshot.tick);
  populateConfig(snapshot.config);
  updateTarget(snapshot.target);
  updateObservers(snapshot.observers);
  updateEstimates(snapshot.estimates);
}

async function updateConfig(event) {
  event.preventDefault();
  const next = structuredClone(latestConfig);
  next.max_slant_range_yd = Number(document.getElementById("max-range").value);
  next.target.desired_hdg_deg = Number(document.getElementById("desired-hdg").value);
  next.target.desired_through_water_speed_kt = Number(document.getElementById("desired-speed").value);
  next.target.desired_depth_ft = Number(document.getElementById("desired-depth").value);
  next.presence_probability_pct = Number(document.getElementById("probability").value);
  next.source.shared_recognition_bias_hz = Number(document.getElementById("frequency-bias").value);
  next.smoothing_window_seconds = Number(document.getElementById("smoothing-window").value);
  const response = await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(next),
  });
  if (!response.ok) throw new Error(await response.text());
  latestConfig = await response.json();
  setMessage("設定を反映しました。");
}

document.getElementById("config-form").addEventListener("submit", (event) => {
  updateConfig(event).catch((error) => setMessage(`設定エラー: ${error.message}`));
});

document.getElementById("reset").addEventListener("click", async () => {
  const response = await fetch("/api/reset", { method: "POST" });
  if (response.ok) setMessage("ランタイムをリセットしました。履歴は保持されています。");
});

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws`);
  socket.addEventListener("open", () => setText("connection-status", "接続済み"));
  socket.addEventListener("message", (event) => render(JSON.parse(event.data)));
  socket.addEventListener("close", () => {
    setText("connection-status", "再接続中");
    setTimeout(connect, 1500);
  });
  socket.addEventListener("error", () => socket.close());
}

connect();
