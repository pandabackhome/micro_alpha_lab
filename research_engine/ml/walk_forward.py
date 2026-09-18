from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, Tuple


@dataclass(frozen=True)
class DaySplit:
    train: Tuple[str, ...]
    validation: Tuple[str, ...]
    test: Tuple[str, ...]


def day_splits(days: Sequence[str], train_days: int, validation_days: int,
               test_days: int, expanding: bool = True) -> Iterator[DaySplit]:
    ordered = sorted(set(days))
    minimum = train_days + validation_days + test_days
    for end in range(minimum, len(ordered) + 1, test_days):
        test_start = end - test_days
        validation_start = test_start - validation_days
        train_start = 0 if expanding else validation_start - train_days
        yield DaySplit(
            train=tuple(ordered[train_start:validation_start]),
            validation=tuple(ordered[validation_start:test_start]),
            test=tuple(ordered[test_start:end]),
        )

