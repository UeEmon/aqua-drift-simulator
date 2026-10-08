from aqua_drift.orchestration import ManagedContainer, free_slots, plan, slot_name, slot_number


def container(slot: str, running: bool = True, age: float = 5.0) -> ManagedContainer:
    return ManagedContainer(slot=slot, container_id=f"id-{slot}", running=running, age_s=age)


def test_slot_names_are_01_to_99() -> None:
    assert slot_name(1) == "obs-01" and slot_name(99) == "obs-99"
    assert slot_number("obs-07") == 7
    assert slot_number("obs-100") is None and slot_number("obs-00") is None
    assert slot_number("4f2a9c1b") is None
    assert len(free_slots(set(), 99)) == 99
    assert free_slots(set(), 150)[-1] == "obs-99"


def test_initial_observers_start_lowest_slots() -> None:
    result = plan(limit=99, active_ids=[], pending_placements=0, initial_remaining=4, containers=[])
    assert result.start == ["obs-01", "obs-02", "obs-03", "obs-04"]
    assert result.stop == [] and result.evict_oldest == 0


def test_no_idle_containers_without_demand() -> None:
    active = ["obs-01", "obs-02", "obs-03", "obs-04"]
    running = [container(s) for s in active]
    result = plan(limit=99, active_ids=active, pending_placements=0, initial_remaining=0, containers=running)
    assert result.start == [] and result.stop == []


def test_placements_start_containers_and_reuse_freed_numbers() -> None:
    # obs-02 ended (its container exited and was removed): its number is reused first
    active = ["obs-01", "obs-03", "obs-04"]
    running = [container(s) for s in active]
    result = plan(limit=99, active_ids=active, pending_placements=2, initial_remaining=0, containers=running)
    assert result.start == ["obs-02", "obs-05"]


def test_containers_still_starting_count_towards_demand() -> None:
    running = [container("obs-01"), container("obs-05", age=2.0)]  # obs-05 not registered yet
    result = plan(limit=99, active_ids=["obs-01"], pending_placements=1, initial_remaining=0, containers=running)
    assert result.start == []


def test_idle_surplus_is_stopped_after_grace() -> None:
    running = [container("obs-01"), container("obs-02", age=90.0), container("obs-03", age=10.0)]
    result = plan(limit=99, active_ids=["obs-01"], pending_placements=0, initial_remaining=0, containers=running)
    assert result.stop == ["id-obs-02"]  # obs-03 is still within its start-up grace


def test_full_slots_evict_oldest() -> None:
    active = [slot_name(n) for n in range(1, 100)]
    running = [container(s) for s in active]
    result = plan(limit=99, active_ids=active, pending_placements=2, initial_remaining=0, containers=running)
    assert result.start == [] and result.evict_oldest == 2


def test_warm_standby_keeps_spare_containers() -> None:
    result = plan(limit=99, active_ids=["obs-01"], pending_placements=0, initial_remaining=0,
                  containers=[container("obs-01")], warm_standby=2)
    assert result.start == ["obs-02", "obs-03"]
