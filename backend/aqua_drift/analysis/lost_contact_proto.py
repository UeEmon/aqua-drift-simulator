"""Prototype of the lost-contact response (designs/lost-contact-response.md), for the study in
lost_contact.py only. Installed on one ScenarioRun; nothing in the services uses it.

Variants (letters as in the design note; combine with "+"):
  C2  COASTING (no detection >= LOST_DEBOUNCE_S, not LOST): the forward planner keeps running
      on the coasted estimate (its own sigma gates still apply); replanning stays frozen
  C3  LOST (sigma >= 0.5 R or no detection >= LOST_TIMEOUT_S): open forward drops the
      replanner may revise (aqua_drift.replanning.is_revisable) are replaced and search drops are laid where an R-disc covers the most
      particle mass at the expected drop time (greedy, until SEARCH_PD or SEARCH_MAX_DROPS)
  B2  while not detecting, a larger share of the particles maneuvers (COAST_MANEUVER_FRACTION)
"""
from __future__ import annotations

import math

import numpy as np

from aqua_drift.models import DropTask

YD_TO_M = 0.9144
LOST_DEBOUNCE_S = 10
LOST_SIGMA_FRACTION = 0.5
LOST_TIMEOUT_S = 300
SEARCH_PD = 0.7
SEARCH_MAX_DROPS = 4
SEARCH_RADIUS_FRACTION = 0.8
SEARCH_INTERVAL_S = 60
COAST_MANEUVER_FRACTION = 0.3
FALL_S = 40.0


def stage(run) -> str:
    engine = run.engine
    if not engine.pf.initialized or engine.last_detect_tick is None:
        return "NONE"
    since = engine.tick - engine.last_detect_tick
    if since < LOST_DEBOUNCE_S:
        return "TRACKING"
    pf = engine.pf
    mean = pf.w @ pf.x[:, 0:2]
    diff = pf.x[:, 0:2] - mean
    cov = (diff * pf.w[:, None]).T @ diff
    sigma = math.sqrt(max(np.linalg.eigvalsh(cov)[-1], 0.0))
    r = run.config.max_slant_range_yd * YD_TO_M
    if sigma >= LOST_SIGMA_FRACTION * r or since >= LOST_TIMEOUT_S:
        return "LOST"
    return "COASTING"


def install(run, variant: str) -> None:
    parts = set(variant.upper().split("+"))
    original_step = run.step
    original_deploy = run._forward_deploy
    base_fraction = run.config.estimator.maneuver_fraction
    memo = {"last_search": None, "stages": []}
    run.lost_stages = memo["stages"]

    def step():
        if "B2" in parts and run.engine.pf.initialized and run.engine.last_detect_tick is not None:
            coasting = run.engine.tick - run.engine.last_detect_tick >= LOST_DEBOUNCE_S
            run.engine.pf.maneuver_fraction = COAST_MANEUVER_FRACTION if coasting else base_fraction
        return original_step()

    def deploy():
        now = stage(run)
        memo["stages"].append((run.tick, now))
        if now == "COASTING" and "C2" in parts:
            engine = run.engine
            real_output = engine.output

            def output():
                out = real_output()
                for est in out.estimates:
                    est.observability_status = est.observability_status.removeprefix("COASTING_")
                return out

            engine.output = output
            forward = run.config.forward
            replan = forward.replan_enabled
            forward.replan_enabled = False  # replanning stays frozen while coasting
            try:
                original_deploy()
            finally:
                del engine.output
                forward.replan_enabled = replan
            return
        if now == "LOST" and "C3" in parts:
            last = memo["last_search"]
            searching = any(t.status == "APPROVED" and t.source == "search" for t in run.tasks)
            if (last is None or run.tick - last >= SEARCH_INTERVAL_S) and not searching:
                _search(run)
                memo["last_search"] = run.tick
            return
        original_deploy()

    run.step = step
    run._forward_deploy = deploy


def _search(run) -> None:
    engine = run.engine
    pf, frame = engine.pf, engine.frame
    from aqua_drift.replanning import flying_task_id, is_revisable

    layer = run.layer_state
    flying = flying_task_id(layer)
    for task in run.tasks:  # forward drops planned on the lost track: same freeze rule as replanning
        if task.status == "APPROVED" and is_revisable(task, run.tick, run.config.forward, flying):
            task.status = "REPLACED"
    free = run.config.observer_limit - len(run.observers) - sum(t.status == "APPROVED" for t in run.tasks)
    count = min(SEARCH_MAX_DROPS, free)
    if count <= 0:
        return
    w = pf.w / pf.w.sum()
    mean = w @ pf.x[:, 0:3]
    eta = 120.0
    if layer is not None:
        from aqua_drift.layer import eta_to
        eta = eta_to(layer, run.config.layer, frame.to_geo(*mean))
    t_drop = eta + FALL_S
    # water frame: drop points drift with the water like the target, so only the
    # through-water velocity moves the target relative to them
    pred = pf.x[:, 0:2] + pf.x[:, 3:5] * t_drop
    depth = float(np.clip(np.sum(w * pf.x[:, 2]), 50 * 0.3048, 1500 * 0.3048))
    radius = SEARCH_RADIUS_FRACTION * run.config.max_slant_range_yd * YD_TO_M
    rng = np.random.default_rng(run.tick)
    candidates = pred[rng.choice(len(pred), size=min(400, len(pred)), p=w)]
    left = w.copy()
    covered = 0.0
    chosen = []
    for _ in range(count):
        inside = np.linalg.norm(pred[None, :, :] - candidates[:, None, :], axis=2) <= radius
        gain = inside @ left
        best = int(np.argmax(gain))
        if gain[best] <= 1e-3:
            break
        chosen.append(candidates[best])
        covered += float(gain[best])
        left[inside[best]] = 0.0
        if covered >= SEARCH_PD:
            break
    for e, n in chosen:
        run.tasks.append(DropTask(
            task_id=len(run.tasks) + 1, created_tick=run.tick, source="search",
            reason=f"search ({100 * covered:.0f} % of the cloud)",
            position=frame.to_geo(float(e), float(n), depth), status="APPROVED",
            approved_tick=run.tick, planned_tick=None,
        ))
    if chosen:
        run.deployments.append((run.tick, len(chosen)))
        run.last_deploy_tick = run.tick
