# AQUA-DRIFT Simulator

Synthetic three-dimensional marine drift and Doppler simulation environment using Docker,
PostgreSQL/PostGIS, CesiumJS, and bundled Natural Earth II imagery.

The simulator intentionally does **not** use bearing information. When direction data is absent,
the estimator reports the state as unobservable instead of inventing an absolute horizontal
position.

## Included services

- Shared one-second simulation clock
- Rate-limited target motion simulator
- Independently scalable passive-drift observer containers (1–100)
- Common affine current-field service
- Synthetic Doppler engine with common time-invariant frequency-recognition bias
- Online and retrospective estimate branches
- Immutable observation history in PostgreSQL/PostGIS
- CesiumJS 3D GIS with offline Natural Earth II imagery
- PlantUML component, class, sequence, and observer lifecycle diagrams

## Start with Docker Desktop

```bash
cp .env.example .env
docker compose up --build --scale observer=3
```

Open <http://localhost:8090>. The API health endpoint is available at
<http://localhost:8091/health> on the local machine.

Scale observers between 1 and 100:

```bash
docker compose up -d --scale observer=20
```

When the 101st active observer is registered, the oldest active observer is evicted. Its event
history remains in the database. Each observer can remain active for at most three hours.

## Configuration

The GIS control panel can change:

- Common maximum slant range in yards
- Desired target HDG, through-water speed, and depth
- Presence probability percentage
- Retrospective recomputation window

Target changes are rate limited by degrees/second, knots/second, and feet/second. Additional
parameters, including the current-field matrix and common source-frequency bias, are available
through `PUT /api/config`.

## Display conventions

- Horizontal/slant distance: YD
- Depth: Ft
- Speed: kt
- HDG/COG: degrees true
- Presence regions: sampled range-only point clouds, not forced ellipses or ellipsoids

## Development tests

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e './backend[dev]'
ruff check backend
pytest backend/tests
```

Architecture notes and PlantUML sources are under [`docs/`](docs/).
