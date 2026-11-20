"""設標者の投入レート: observers the layer (設標者) lays per unit time, and when it reaches its
drop limit (投入上限数) at that rate.

The rate is measured over the last `window_s` seconds (or the run so far when it is shorter):
drops laid in that time / that time. The estimate of the limit assumes the layer goes on laying
at the measured rate: remaining / rate from now. A run measured for less than MIN_MEASURED_S
gives no rate yet (one early drop would read as a very high rate)."""
from __future__ import annotations

from collections.abc import Sequence

from aqua_drift.models import DropRate

MIN_MEASURED_S = 60.0


def drop_rate(
    drop_ticks: Sequence[int], tick: int, start_tick: int, window_s: int, limit: int = 0, open_count: int = 0,
) -> DropRate:
    """drop_ticks: when each observer of this run was laid (ascending); start_tick: start of the
    run; limit: the drop limit (0 = none); open_count: drops still to be laid."""
    laid = len(drop_ticks)
    measured = float(min(window_s, max(tick - start_tick, 0)))
    since = tick - measured
    window_laid = laid if since <= start_tick else sum(1 for t in drop_ticks if t > since)
    rate = window_laid / measured * 3600.0 if measured >= MIN_MEASURED_S else None
    result = DropRate(
        tick=tick, laid=laid, open=open_count, window_s=window_s, measured_s=measured,
        window_laid=window_laid, rate_per_hour=rate,
        last_drop_tick=drop_ticks[-1] if drop_ticks else None,
    )
    if limit > 0:
        result.limit = limit
        result.remaining = max(limit - laid, 0)
        result.reached = laid >= limit
        if result.reached:
            result.limit_eta_s = 0.0
            result.limit_tick = drop_ticks[limit - 1]
        elif rate:
            result.limit_eta_s = result.remaining / rate * 3600.0
            result.limit_tick = tick + round(result.limit_eta_s)
    return result
