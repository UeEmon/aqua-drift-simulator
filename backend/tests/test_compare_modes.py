from aqua_drift.analysis.compare_modes import MODES, Noise, build_geometry, crlb

YD = 0.9144


def _pos_bound(cov) -> float:
    return (cov[0, 0] + cov[1, 1]) ** 0.5 / YD


def test_observation_mode_bounds_are_ordered() -> None:
    geo = build_geometry(4, 1500)
    noise = Noise()
    bound = {name: _pos_bound(crlb(geo, kinds, noise)) for name, kinds in MODES.items()}
    assert bound["POS"] < bound["RB"] < bound["BRG"]
    # detection gating at the known common max range is strong range information
    assert bound["DOP+GATE"] < bound["DOP"]
    assert bound["DOP+GATE"] < bound["RB"]
    # adding bearings never hurts
    assert bound["BRG+DOP"] <= bound["DOP"] + 1e-6
