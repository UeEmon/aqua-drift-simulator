# AQUA-DRIFT Simulator

潮流による外力を受ける潜没目標を、漂流する観測者（1〜100）の**ドップラー観測のみ**から
3次元で航跡処理する合成データ・シミュレーターです。Docker 上で目標・観測者・音源演算・推定器を
独立コンテナとして動かし、CesiumJS（オープンソース GIS）で表示します。

Synthetic 3-D simulator for Doppler-only tracking of a submerged target under tidal current,
built on Docker, PostgreSQL/PostGIS and CesiumJS. **No bearing information is used.**

## What it computes

From Doppler frequency, detection/non-detection at the common maximum slant range, and the
exact time/position/depth of each drifting observer:

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

- [docs/requirements.md](docs/requirements.md) – requirements and item-by-item traceability
- [docs/estimation-methods.md](docs/estimation-methods.md) – Kalman filter explained, alternatives, chosen method
- [docs/observation-mode-comparison.md](docs/observation-mode-comparison.md) – position / range-bearing / bearing-only / Doppler / combinations
- [docs/architecture.md](docs/architecture.md) and [docs/uml/](docs/uml/) – PlantUML design

## Start with Docker

```bash
cp .env.example .env
docker compose up --build --scale observer=4
```

Open <http://localhost:8090>. API health: <http://localhost:8091/health>.

Add observers (1–100). To place the next one at a specific point, double-click the map (or enter
coordinates) and press 「配置を予約」, then scale up:

```bash
docker compose up -d --scale observer=5
```

When a new observer exceeds the limit, the oldest active observer is removed; its history stays
in the database. Each observer observes for at most three hours.

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

## Display conventions

YD for distance, Ft for depth, kt for speed, degrees true for HDG/COG. A depth exaggeration factor
(default ×10) is applied in the 3-D view only.
