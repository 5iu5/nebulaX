import numpy as np
import pytest

from shm_v2.features import rainflow_cycles, rainflow_moment


def test_astm_e1049_cycle_count_example() -> None:
    # Standard ASTM E1049-style sequence used to verify range-pair counting.
    signal = np.array([0, -2, 1, -3, 5, -1, 3, -4, 4, -2, 0], dtype=float)
    cycles = rainflow_cycles(signal)
    counts = cycles.groupby("range")["count"].sum().to_dict()

    assert counts == {2.0: 1.0, 3.0: 0.5, 4.0: 1.5, 6.0: 0.5, 8.0: 1.0, 9.0: 0.5}
    assert set(cycles["count"]) == {0.5, 1.0}
    assert cycles.loc[cycles["count"] == 0.5, "count"].sum() == pytest.approx(4.0)


def test_rainflow_moment_uses_amplitude_not_range() -> None:
    signal = np.array([0.0, 4.0, 0.0])
    cycles = rainflow_cycles(signal)
    # Two half cycles of range 4 make one equivalent cycle at amplitude 2.
    assert rainflow_moment(cycles, m=5.0) == pytest.approx(2.0**5)
