import numpy as np
import pandas as pd
import pytest

from research_engine.ml.baseline import make_model
from research_engine.ml.sequence import SequenceClassifier, build_windows, window_bounds

torch = pytest.importorskip("torch")


def test_a_window_never_reaches_into_the_previous_session():
    """The failure this module exists to prevent.

    Rows are stored day after day in one frame. A window that ran off the start
    of a session would read the previous close as though it were the minutes
    before the open -- not a lookback, a different day's outcome.
    """
    groups = ["d1", "d1", "d1", "d2", "d2", "d2", "d2"]
    bounds = window_bounds(groups, lookback=3)
    # First row of d2 is index 3; nothing at or before index 2 may be readable.
    assert bounds[3] == (3, 4)
    assert bounds[4] == (3, 5)
    assert bounds[5] == (3, 6)
    # Only once the session has produced three rows is the window full.
    assert bounds[6] == (4, 7)
    for i, (start, stop) in enumerate(bounds):
        assert all(groups[j] == groups[i] for j in range(start, stop))


def test_training_drops_short_windows_and_prediction_pads_them():
    values = np.arange(10, dtype=float).reshape(5, 2)
    groups = ["d1", "d1", "d1", "d2", "d2"]

    full, kept = build_windows(values, groups, lookback=3, full_only=True)
    # Only d1's third row has three rows of its own session behind it.
    assert kept.tolist() == [2]
    assert full.shape == (1, 3, 2)

    padded, kept_all = build_windows(values, groups, lookback=3, full_only=False)
    assert kept_all.tolist() == [0, 1, 2, 3, 4]
    assert padded.shape == (5, 3, 2)
    # d2's first row pads by repeating itself, never by borrowing from d1.
    assert np.array_equal(padded[3], np.repeat(values[3:4], 3, axis=0))


def _frame(n_days=3, per_day=40, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    rows, dates, labels = [], [], []
    for d in range(n_days):
        x = rng.normal(size=(per_day, n_features))
        rows.append(x)
        dates += ["2026-09-{:02d}".format(10 + d)] * per_day
        # A weak but real dependence on the previous row, so the model has
        # something a sequence could in principle use.
        drift = np.r_[0.0, x[:-1, 0]]
        labels += np.where(drift > 0.6, "UP", np.where(drift < -0.6, "DOWN", "FLAT")).tolist()
    return pd.DataFrame(np.vstack(rows), columns=["f{}".format(i) for i in range(n_features)]), \
        np.array(dates), np.array(labels)


def test_predicts_one_row_per_input_row_including_the_start_of_a_session():
    x, dates, y = _frame()
    model = SequenceClassifier(lookback=5, hidden=8, epochs=2)
    model.fit(x, y, groups=dates)
    proba = model.predict_proba(x, groups=dates)
    assert proba.shape == (len(x), len(model.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert len(model.predict(x, groups=dates)) == len(x)


def test_refuses_to_run_without_session_keys():
    """Silently assuming one session would be the leak; an error is the fix."""
    x, dates, y = _frame()
    model = SequenceClassifier(lookback=5, hidden=8, epochs=1)
    with pytest.raises(ValueError, match="groups"):
        model.fit(x, y)
    model.fit(x, y, groups=dates)
    with pytest.raises(ValueError, match="groups"):
        model.predict_proba(x)


def test_refuses_a_lookback_no_session_can_fill():
    x, dates, y = _frame(per_day=6)
    model = SequenceClassifier(lookback=50, hidden=8, epochs=1)
    with pytest.raises(ValueError, match="lookback"):
        model.fit(x, y, groups=dates)


def test_same_seed_same_probabilities():
    x, dates, y = _frame()
    first = SequenceClassifier(lookback=5, hidden=8, epochs=3, seed=7).fit(x, y, groups=dates)
    second = SequenceClassifier(lookback=5, hidden=8, epochs=3, seed=7).fit(x, y, groups=dates)
    assert np.allclose(first.predict_proba(x, groups=dates), second.predict_proba(x, groups=dates))


def test_validation_is_used_for_stopping_and_never_for_fitting():
    """Early stopping may read the validation split; a gradient may not.

    Checked by giving the validation rows labels the training split never sees:
    if they reached the loss, the class list would have grown.
    """
    x, dates, y = _frame()
    vx, vdates, _ = _frame(seed=1)
    model = SequenceClassifier(lookback=5, hidden=8, epochs=3)
    model.fit(x, y, groups=dates,
              eval_set=(vx, np.array(["UNSEEN"] * len(vx)), vdates))
    assert "UNSEEN" not in set(model.classes_.tolist())


def test_factory_exposes_it_and_declares_what_it_needs():
    model = make_model("lstm", "classification")
    assert isinstance(model, SequenceClassifier)
    # walk_forward_train reads these rather than special-casing a model name.
    assert model.requires_groups and model.uses_validation
    with pytest.raises(ValueError, match="classification only"):
        make_model("lstm", "regression")
