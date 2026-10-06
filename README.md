# AQUA-DRIFT Simulator

潮流による外力を受ける潜没目標を、漂流する観測者（1〜100）の**ドップラー観測と方位観測**から
3次元で航跡処理する合成データ・シミュレーターです。Docker 上で目標・観測者・音源演算・推定器を
独立コンテナとして動かし、CesiumJS（オープンソース GIS）で表示します。

Synthetic 3-D simulator for Doppler + bearing tracking of a submerged target under tidal current,
built on Docker, PostgreSQL/PostGIS and CesiumJS.

## What it computes

From Doppler frequency, horizontal bearing (σ 15°, every 15 s), detection/non-detection at the
common maximum slant range, and the exact time/position/depth of each drifting observer:

- current position and depth, HDG and COG
- through-water, over-ground and observer-relative speed
- past track in two variants: **not updated** (ONLINE) and **updated** within a configurable
  recomputation window (SMOOTHED)
- uncertainty (1σ) and the **estimated presence region** at a user-set probability (%)
  (highest-density region, not an ellipse/ellipsoid; split regions are shown separately)
- per-observer closest-approach time, slant range and relative speed with error from the
  source-frequency error and the speed error
- the common source-frequency recognition bias and the linear current field

Documentation:

- [docs/system-requirements.md](docs/system-requirements.md) – **動作環境の要件**（サーバー・ブラウザ・GPU・ストレージ・ネットワーク）
- [docs/requirements.md](docs/requirements.md) – requirements and item-by-item traceability
- [docs/estimation-methods.md](docs/estimation-methods.md) – Kalman filter explained, alternatives, chosen method
- [docs/observation-mode-comparison.md](docs/observation-mode-comparison.md) – position / range-bearing / bearing-only / Doppler / combinations
- [docs/architecture.md](docs/architecture.md) and [docs/uml/](docs/uml/) – PlantUML design

## GIS panel

- **Estimation bar**: start / stop the estimator (a start begins a new run from the current time)
- **推定と真値**: estimate vs truth side by side (value, truth, error, 1σ), error-over-time charts,
  relative speed / range / bearing per observer, CPA (estimate vs truth), current field
- **目標設定**: target initial position, depth, HDG and speed → restart; live maneuver commands
- **観測・推定条件**: max slant range, bearing σ / interval, bearing use on/off, frequency bias,
  current field, probability %, recompute window, particles
- **表示・観測者**: layer toggles (truth, both tracks, presence region, bearing lines, error line)
  and observer placement

- **View toolbar (map, top right)**: oblique / top-down (vertical) / horizontal side view
  (direction selectable), orthographic top view, **centre on truth (default centre target)**,
  centre on estimate, **follow** on/off (default **on, following the truth**) with the follow target truth / estimate
  (the target is kept at the view centre every frame; altitude, depression and heading are kept), FPS overlay
- **Camera readout (map, bottom right)**: camera latitude / longitude, altitude above the sea
  surface in ft, depression angle, heading and the view-centre coordinates. The camera starts at
  10000 ft without a fly-in or automatic zoom

By default four observers surround the target's initial position; the remaining observer
containers wait in standby. The `deployer` service places them **ahead (前程) of the estimated
target** whenever the estimate predicts it will leave the current observer field (estimate
only, never the truth); the 表示・観測者 tab shows the status and has a "deploy now" button.

The background map (Natural Earth II) is **off by default** to prioritise rendering; switch it
on with 「背景地図」 in the view toolbar when needed.

Rendering uses GPU-batched Cesium primitives, on-demand rendering and 4x MSAA. The GIS stream
(WebSocket protocol v2) sends a full update once and compact deltas afterwards; a Web Worker
decodes it and converts coordinates off the rendering thread; tracks are drawn in chunks so
only changed chunks reach the GPU; quality adapts to the frame time; markers glide between
the 1 Hz updates ("なめらか", switched off automatically on very slow GPUs). The 「性能」 toggle
shows frame rate, update time, received bytes, quality level and the GPU in use. A hardware
GPU with WebGL2 is recommended; the GIS also runs on software WebGL (used in CI).

## System requirements (summary)

| | Minimum (default, 12 observer containers) | Recommended |
|---|---|---|
| Server | Docker Engine 24+ / Compose v2.20+, x86-64, 4 cores, 3 GB RAM for Docker, 10 GB free | 4–8 cores, 4–8 GB, 20 GB SSD |
| Browser | Chrome / Edge 98+, Firefox 94+, Safari 15.4+, WebGL 2, hardware acceleration on, 1280 px wide | Latest Chrome / Edge, recent GPU, 1600 px+ |

Internet is needed only for the first build; the system runs offline. There is no
authentication — keep it on a trusted network or set `WEB_BIND=127.0.0.1`. Details, measured
figures and storage growth: [docs/system-requirements.md](docs/system-requirements.md).

## Start with Docker

```bash
cp .env.example .env
docker compose up --build
```

Observers are numbered obs-01 .. obs-99. The `orchestrator` service starts an observer
container only when an observer is needed (the initial four around the target, each forward
or manual placement) and the container is removed when the observer ends, so memory is used
only for observers in service; a freed number is reused (history kept as `obs-03#1`). It needs
the Docker socket; without it, use the static mode
`docker compose --profile static-observers up --build --scale observer=12`.
Stop with `docker compose down --remove-orphans`.

Open <http://localhost:8090>. API health: <http://localhost:8091/health>.

To add an observer at a specific point, double-click the map (or enter coordinates) and press
「配置を予約」: the orchestrator starts a container for it automatically (1–99 observers).
When all 99 numbers are in use, the oldest observer is removed to free a number; its history
stays in the database. Each observer observes for at most three hours.

## Configuration (GIS panel or `PUT /api/config`)

- Target: HDG (deg) and HDG rate (deg/s), through-water speed (kt) and rate (kt/s),
  depth (Ft) and rate (Ft/s)
- Common maximum slant range (YD), presence probability (%), past-track recomputation window (s)
- Observer limit, source frequency, common frequency-recognition bias (Hz)
- Truth current field (base vector and gradient), estimator settings (particles, bias prior …)

## Offline runs and analysis

```bash
cd backend
pip install -e '.[dev]'
python -m aqua_drift.scenario --seconds 3000 --observers 4 --bias-hz 0.2   # closed-loop run
python -m aqua_drift.analysis.compare_modes --observers 4 --runs 20        # observation-mode study
ruff check . && pytest
```

CI also starts the full stack with Docker and checks the GIS in headless Chromium
(`e2e/ui_check.py`); screenshots are uploaded as the `gis-e2e` artifact.

## Display conventions

YD for distance, Ft for depth, kt for speed, degrees true for HDG/COG. A depth exaggeration factor
(default ×10) is applied in the 3-D view only.
