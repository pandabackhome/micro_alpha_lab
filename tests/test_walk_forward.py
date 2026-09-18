from research_engine.ml.walk_forward import day_splits


def test_expanding_day_splits_are_ordered_and_disjoint():
    days = ["2026-09-{:02d}".format(i) for i in range(1, 17)]
    splits = list(day_splits(days, 10, 2, 1, expanding=True))
    assert splits[0].train == tuple(days[:10])
    assert splits[0].validation == tuple(days[10:12])
    assert splits[0].test == (days[12],)
    assert splits[1].train == tuple(days[:11])
    assert splits[1].validation == tuple(days[11:13])
    assert splits[1].test == (days[13],)
    assert set(splits[0].train).isdisjoint(splits[0].test)


def test_sliding_window_does_not_reuse_older_days():
    days = [str(i) for i in range(20)]
    splits = list(day_splits(days, 3, 2, 1, expanding=False))
    assert len(splits[0].train) == 3
    assert len(splits[1].train) == 3
    assert splits[0].train != splits[1].train
