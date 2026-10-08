from aqua_drift.analysis.depth_doppler import (
    FT,
    Pass,
    crlb,
    lloyd_mirror_beat,
    surround,
    vertical_pairs,
    with_overflight,
)


def test_overflight_observer_makes_depth_observable() -> None:
    base = surround()
    current = crlb(Pass(observers=base))["depth_ft"]
    over = crlb(Pass(observers=with_overflight(base, 0.0)))["depth_ft"]
    beside = crlb(Pass(observers=with_overflight(base, 1500 * FT)))["depth_ft"]
    # the vertical share of the slant miss distance carries the depth information
    assert over < current / 10
    assert over < beside < current


def test_vertical_baseline_beats_current_depth_cycle() -> None:
    current = crlb(Pass(observers=surround()))["depth_ft"]
    pairs = crlb(Pass(observers=vertical_pairs(surround(depths_ft=(200,) * 4), 1000)))["depth_ft"]
    spread = crlb(Pass(observers=surround(depths_ft=(60, 1000, 60, 1000))))["depth_ft"]
    assert pairs < current / 2
    assert spread < current / 2


def test_lloyd_mirror_beat_matches_far_field_formula() -> None:
    zs, zr, r, rdot = 500 * FT, 200 * FT, 2000.0, 4.0
    beat = lloyd_mirror_beat(zs, zr, r, rdot)
    # far field: path difference ~ 2 zs zr / R, its rate ~ -2 zs zr rdot / R^2
    assert abs(beat["path_difference_m"] - 2 * zs * zr / r) < 0.05 * beat["path_difference_m"]
    approx = 400.0 * (2 * zs * zr * rdot / r**2) / 1500.0
    assert abs(beat["beat_hz"] - approx) < 0.05 * approx
