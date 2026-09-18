"""A recurrent classifier that keeps the row-wise contract the rest of the ML
layer is written against.

Every other model here is an sklearn pipeline: `fit(X, y)`, `predict_proba(X)`,
`classes_`, one row in and one probability out. A recurrent net wants
`(batch, timesteps, features)` instead, so the windowing has to live somewhere.
It lives here rather than in the dataset layer, because putting it there would
force every model to carry a sequence axis it does not use, and because the one
rule that matters -- **a window never spans two sessions** -- is easier to keep
true in one place than at every call site.

That rule is not a detail. Rows are stored day after day in one frame; a window
that ran off the start of 09-16 would read the close of 09-15 as though it were
the minutes before the open, which is not a lookback, it is a different day's
outcome. `fit` and `predict` therefore take `groups` (the session date per row)
and build windows strictly inside each group.

Two properties inherited from the surrounding pipeline, stated because they are
easy to misread:

* Rows arrive already strided (`load_dataset(stride=...)`, default 10), so a
  `lookback` of 12 is 120 seconds of history at 10-second resolution, not 12
  seconds. The stride exists to stop adjacent 1-second rows from being counted
  as independent examples, and sequences do not get to opt out of it -- feeding
  overlapping 1-second windows would put back exactly the dependence the stride
  removes, and the significance estimates are fragile enough already.
* The first `lookback - 1` rows of each session have no full history. Training
  drops them; prediction cannot, because the caller needs one probability per
  test row, so those windows are left-padded with the session's first row. The
  padded rows are real predictions made on degraded input, not placeholders.

Validation days are used for early stopping and for nothing else. The baselines
ignore them; a net with tens of thousands of parameters trained on ten sessions
will fit them perfectly if nothing stops it, and the validation split is what
that decision is for. It never enters a gradient.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by the import guard test
        raise ImportError("Install optional dependency: pip install '.[torch]'") from exc
    return torch


def window_bounds(groups: Sequence, lookback: int) -> List[Tuple[int, int]]:
    """For each row, the half-open slice of rows its window may read.

    Returned per row so the caller can tell a full window from a padded one:
    `stop - start < lookback` means the session had not run long enough yet.
    Rows are assumed to be in time order within each group, which is how
    `load_dataset` builds them.
    """
    bounds: List[Tuple[int, int]] = []
    start_of_group = 0
    previous = object()
    for i, key in enumerate(groups):
        if key != previous:
            start_of_group = i
            previous = key
        bounds.append((max(start_of_group, i - lookback + 1), i + 1))
    return bounds


def build_windows(values: np.ndarray, groups: Sequence, lookback: int,
                  *, full_only: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Stack each row's window into `(n, lookback, features)`.

    `full_only` drops rows whose session has not produced `lookback` rows yet,
    which is what training wants: a padded window is a fabricated observation
    and there is no reason to fit on one. Prediction passes False and pads by
    repeating the earliest row available in that session, because the caller
    requires one output per input row.
    """
    bounds = window_bounds(groups, lookback)
    keep = [i for i, (a, b) in enumerate(bounds) if not full_only or b - a == lookback]
    out = np.empty((len(keep), lookback, values.shape[1]), dtype=np.float32)
    for row, i in enumerate(keep):
        a, b = bounds[i]
        chunk = values[a:b]
        if len(chunk) < lookback:
            chunk = np.vstack([np.repeat(chunk[:1], lookback - len(chunk), axis=0), chunk])
        out[row] = chunk
    return out, np.asarray(keep, dtype=int)


class SequenceClassifier:
    """LSTM over a fixed lookback, exposing the estimator interface used here.

    Kept deliberately small. Ten sessions of strided rows is a few tens of
    thousands of examples that are not independent of one another; a wide net
    would memorise them. The defaults are sized so the parameter count stays
    within an order of magnitude of the example count, and anything larger
    should be justified against a held-out fold rather than a training curve.
    """

    #: `walk_forward_train` reads these to decide what to hand `fit`.
    requires_groups = True
    uses_validation = True

    def __init__(self, lookback: int = 12, hidden: int = 32, layers: int = 1,
                 dropout: float = 0.2, epochs: int = 40, patience: int = 6,
                 batch_size: int = 256, learning_rate: float = 1e-3,
                 seed: int = 42, task: str = "classification") -> None:
        if task != "classification":
            raise ValueError("SequenceClassifier handles classification only; "
                             "a regression head would need its own loss and metric")
        self.lookback = int(lookback)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.classes_: Optional[np.ndarray] = None

    # -- preprocessing -----------------------------------------------------
    # Median impute then standardise, fitted on training rows only. The same
    # two steps the sklearn baselines get from their pipeline; a recurrent net
    # needs them more, not less, because unscaled inputs saturate the gates.
    def _fit_scaler(self, x: np.ndarray) -> None:
        self._median = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
        self._median = np.where(np.isfinite(self._median), self._median, 0.0)
        filled = self._impute(x)
        self._mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        # A column that never moves carries no information and would divide by
        # zero; leaving its scale at 1 turns it into a constant the net ignores.
        self._scale = np.where(std > 1e-12, std, 1.0)

    def _impute(self, x: np.ndarray) -> np.ndarray:
        out = np.where(np.isfinite(x), x, np.nan)
        return np.where(np.isnan(out), self._median, out)

    def _prepare(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.to_numpy(dtype=np.float64, copy=True)
        return ((self._impute(x) - self._mean) / self._scale).astype(np.float32)

    # -- fit ---------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y, groups=None, eval_set=None):
        torch = _require_torch()
        if groups is None:
            raise ValueError("SequenceClassifier needs `groups` (one session key per row); "
                             "without it a window would silently span two sessions")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y.tolist())))
        index = {label: i for i, label in enumerate(self.classes_)}

        self._fit_scaler(X.to_numpy(dtype=np.float64, copy=True))
        windows, kept = build_windows(self._prepare(X), list(groups), self.lookback, full_only=True)
        if len(kept) == 0:
            raise ValueError("no session in the training split reached {} rows; "
                             "lower `lookback` or raise the stride".format(self.lookback))
        targets = np.array([index[label] for label in y[kept]], dtype=np.int64)

        # The FLAT class is 92% of rows. Unweighted, the net reaches that
        # accuracy by never predicting anything else, which is both a good loss
        # and a useless model. Inverse-frequency weights are what
        # `class_weight="balanced"` does for the logistic baseline.
        counts = np.bincount(targets, minlength=len(self.classes_)).astype(np.float64)
        weights = np.where(counts > 0, len(targets) / (len(self.classes_) * np.maximum(counts, 1)), 0.0)

        self._model = _build_net(torch, windows.shape[2], self.hidden, self.layers,
                                 self.dropout, len(self.classes_))
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        evaluation = None
        if eval_set is not None:
            vx, vy, vgroups = eval_set
            if len(vx):
                vw, vkept = build_windows(self._prepare(vx), list(vgroups), self.lookback, full_only=True)
                if len(vkept):
                    vy = np.asarray(vy)[vkept]
                    known = np.array([label in index for label in vy])
                    if known.any():
                        evaluation = (torch.from_numpy(vw[known]),
                                      torch.tensor([index[label] for label in vy[known]], dtype=torch.int64))

        tensor_x = torch.from_numpy(windows)
        tensor_y = torch.from_numpy(targets)
        best_state, best_loss, since_best = None, float("inf"), 0
        for _ in range(self.epochs):
            self._model.train()
            order = torch.randperm(len(tensor_x))
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                optimiser.zero_grad()
                loss = loss_fn(self._model(tensor_x[batch]), tensor_y[batch])
                loss.backward()
                # Recurrent nets on noisy financial windows produce occasional
                # very large gradients; clipping keeps one bad batch from
                # undoing an epoch.
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimiser.step()
            if evaluation is None:
                continue
            self._model.eval()
            with torch.no_grad():
                current = float(loss_fn(self._model(evaluation[0]), evaluation[1]))
            if current < best_loss - 1e-5:
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
                best_loss, since_best = current, 0
            else:
                since_best += 1
                if since_best >= self.patience:
                    break
        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()
        return self

    # -- predict -----------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame, groups=None) -> np.ndarray:
        torch = _require_torch()
        if groups is None:
            raise ValueError("SequenceClassifier needs `groups` at prediction time too")
        windows, kept = build_windows(self._prepare(X), list(groups), self.lookback, full_only=False)
        assert len(kept) == len(X), "prediction must return one row per input row"
        with torch.no_grad():
            logits = self._model(torch.from_numpy(windows))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X: pd.DataFrame, groups=None) -> np.ndarray:
        return self.classes_[self.predict_proba(X, groups=groups).argmax(axis=1)]


def _build_net(torch, n_features: int, hidden: int, layers: int, dropout: float, n_classes: int):
    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = torch.nn.LSTM(
                input_size=n_features, hidden_size=hidden, num_layers=layers,
                batch_first=True, dropout=dropout if layers > 1 else 0.0,
            )
            self.drop = torch.nn.Dropout(dropout)
            self.head = torch.nn.Linear(hidden, n_classes)

        def forward(self, x):
            output, _ = self.lstm(x)
            # The last timestep is the row being predicted; earlier ones are
            # its history and are read only through the hidden state.
            return self.head(self.drop(output[:, -1, :]))

    return Net()
