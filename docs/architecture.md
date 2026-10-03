# AQUA-DRIFT architecture

AQUA-DRIFT is a synthetic, three-dimensional simulator for tracking a submerged target that is
moved by tidal current, using Doppler and horizontal bearing observations from passively
drifting observers. By default four observers surround the target's initial position.
The operator starts/stops the estimator; the GIS shows estimate and truth side by side.

## Containers

| Service | Role |
|---|---|
| `clock` | Shared 1 s time base |
| `target` | Constant HDG / through-water speed / depth; changes follow the configured rates (deg/s, kt/s, Ft/s); moved by the current |
| `observer` (×1..100) | One observer per container; drifts with the water; exact time/position/depth. Start position from env, a queued placement, or the default pattern |
| `current-field` | Truth affine current field `v(p) = a + G (p − p_ref)` |
| `doppler` | Acoustic source / Doppler + bearing engine. Waits until target and all observers reach the tick, then emits one synchronized batch: frequency when slant range ≤ R_max, explicit non-detection otherwise. Truth is attached separately |
| `estimator` | Pulls `/internal/estimator-feed` (observations only, truth stripped) and publishes ONLINE / SMOOTHED estimates, CPA results and the current estimate |
| `api` | FastAPI + WebSocket, single writer of the immutable PostgreSQL/PostGIS event history |
| `web` | nginx + CesiumJS with bundled Natural Earth II imagery |

## Truth separation

`DopplerObservation` carries only what an observer measures (time, exact own position/depth,
detected flag, received frequency, recognized source frequency, noisy bearing). `DopplerTruth` (slant range,
relative speed) and the target state are used for display and evaluation only. The estimator
feed endpoint removes truth, and `EstimatorSettings` exposes no truth parameters (source
frequency, bias magnitude, current field).

## Estimation

See [estimation-methods.md](estimation-methods.md). In short:

1. Linear current field fitted from observer drift over the detectable time `R_max / V_target`.
2. 7-state regularized particle filter `[p, u_through_water, bias]` with Doppler likelihood,
   range-gate likelihood (detected ⇒ inside, not detected ⇒ outside), progressive correction,
   per-cluster kernel jitter and a maneuver mixture.
3. Resample-move every `move_interval_s`: window likelihood, multi-start Levenberg–Marquardt,
   Gaussian-mixture independence MH and random-walk MH; the window is cut when a maneuver
   makes the constant-velocity fit inconsistent.
4. Fixed-lag smoothing by ancestor tracing gives the updated past track (SMOOTHED); the
   filtered track is frozen (ONLINE).
5. CPA analysis per observer pass: time (zero crossing vs recognized frequency), relative
   speed (pre/post frequency change), slant range (CPA slope) and their errors from the
   frequency bias and the speed error.
6. Presence region: highest-density region at the configured probability; disconnected parts
   are reported separately with their probability mass.

## Units

Internal kinematics use metres and seconds. The interface uses YD for distance, Ft for depth,
kt for speed, kt/s for speed change, Ft/s for depth change and degrees true for HDG/COG.

## Diagrams

PlantUML sources: `docs/uml/components.puml`, `classes.puml`, `sequence.puml`,
`estimation-activity.puml`, `observer-state.puml`.
