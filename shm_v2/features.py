"""ASTM E1049 rainflow cycles and the fixed-exponent Miner moment."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

CYCLE_COLUMNS = ["range", "mean", "count", "start_index", "end_index", "sigma_a"]


def reversals(series: Iterable[float]) -> Iterator[tuple[int, float]]:
    """Yield turning points, treating the first and last samples as reversals."""
    values = iter(series)
    first = next(values, None)
    current = next(values, None)
    if first is None or current is None:
        return

    previous_difference = current - first
    yield 0, float(first)
    final_index = 1
    final_value = current
    for index, following in enumerate(values, start=1):
        final_index = index + 1
        final_value = following
        if following == current:
            continue
        difference = following - current
        if previous_difference * difference < 0:
            yield index, float(current)
        first, current = current, following
        previous_difference = difference
    yield final_index, float(final_value)


def extract_cycles(series: Iterable[float]) -> Iterator[tuple[float, float, float, int, int]]:
    """Yield ``(range, mean, count, start, end)`` per ASTM E1049-85.

    Closed cycles have count 1.0; residual half cycles have count 0.5.
    """
    points: deque[tuple[int, float]] = deque()

    def cycle(a: tuple[int, float], b: tuple[int, float], count: float) -> tuple[float, float, float, int, int]:
        return abs(a[1] - b[1]), 0.5 * (a[1] + b[1]), count, a[0], b[0]

    for point in reversals(series):
        points.append(point)
        while len(points) >= 3:
            newer_range = abs(points[-1][1] - points[-2][1])
            older_range = abs(points[-2][1] - points[-3][1])
            if newer_range < older_range:
                break
            if len(points) == 3:
                yield cycle(points[0], points[1], 0.5)
                points.popleft()
            else:
                yield cycle(points[-3], points[-2], 1.0)
                newest = points.pop()
                points.pop()
                points.pop()
                points.append(newest)

    while len(points) > 1:
        yield cycle(points[0], points[1], 0.5)
        points.popleft()


def rainflow_cycles(signal: np.ndarray) -> pd.DataFrame:
    """Return the complete rainflow cycle table for a one-dimensional signal."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Rainflow input must be a one-dimensional signal with at least two samples")
    if not np.isfinite(values).all():
        raise ValueError("Rainflow input contains NaN or infinite values")
    rows = list(extract_cycles(values))
    cycles = pd.DataFrame(rows, columns=CYCLE_COLUMNS[:-1])
    cycles["sigma_a"] = cycles["range"] / 2.0
    return cycles[CYCLE_COLUMNS]


def cache_cycle_table(signal: np.ndarray, path: str | Path) -> pd.DataFrame:
    """Count and atomically cache one cycle table as Parquet."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cycles = rainflow_cycles(signal)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    cycles.to_parquet(temporary, index=False)
    temporary.replace(destination)
    return cycles


def rainflow_log_moment(cycles: pd.DataFrame, m: float = 5.0) -> float:
    """Compute ``log(sum(count * sigma_a**m))`` stably."""
    amplitude = cycles["sigma_a"].to_numpy(dtype=float)
    count = cycles["count"].to_numpy(dtype=float)
    valid = (amplitude > 0) & (count > 0)
    if not valid.any():
        raise ValueError("Signal has no positive-amplitude rainflow cycles")
    terms = m * np.log(amplitude[valid]) + np.log(count[valid])
    peak = float(np.max(terms))
    return float(peak + np.log(np.exp(terms - peak).sum()))


def rainflow_moment(cycles: pd.DataFrame, m: float = 5.0) -> float:
    """Compute the Miner rainflow moment ``S(m)``."""
    return float(np.exp(rainflow_log_moment(cycles, m=m)))


def cumulative_damage(cycles: pd.DataFrame, *, m: float, c: float, signal_length: int) -> pd.DataFrame:
    """Return cumulative cycle damage ordered by cycle end position for plotting."""
    if c <= 0:
        raise ValueError("C must be positive")
    ordered = cycles.loc[cycles["sigma_a"] > 0].sort_values("end_index").copy()
    ordered["damage_increment"] = ordered["count"] * np.power(ordered["sigma_a"], m) / c
    ordered["cumulative_damage"] = ordered["damage_increment"].cumsum()
    denominator = max(signal_length - 1, 1)
    ordered["progress"] = ordered["end_index"] / denominator
    return ordered[["progress", "cumulative_damage"]]
