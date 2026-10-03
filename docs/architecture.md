# AQUA-DRIFT architecture

AQUA-DRIFT is a synthetic, three-dimensional marine drift and Doppler simulation environment.
It deliberately has no bearing observation field, endpoint, message, or estimator input.

## Runtime boundaries

- `simulation-clock` produces the shared one-second time base.
- `target-simulator` advances a rate-limited synthetic target motion profile.
- `observer` is a horizontally scalable service; each replica owns one observer identity.
- `current-field` exposes the common affine current field.
- `doppler-engine` emits one-second synthetic Doppler observations inside the common slant-range gate.
- `estimator` produces online and retrospective outputs without asserting an absolute horizontal
  position when direction information is absent.
- `api` is the only database writer and retains immutable event history.
- `web` serves an offline CesiumJS viewer with bundled Natural Earth II imagery.

## Truth separation

Target truth is published only for simulation display and evaluation. The estimator receives
observer state and synthetic Doppler observations. Its output carries an explicit observability
status. Range-only candidate shells are rendered as point clouds rather than ellipsoidal confidence
approximations.

## Units

Internal kinematics use metres and seconds where needed. The external interface displays distance
in yards, depth in feet, speed in knots, speed change in knots per second, depth change in feet per
second, and direction in degrees true.
