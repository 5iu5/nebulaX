import numpy as np
import pandas as pd
import pytest
from acv.data import from_wide, _fill_short
from acv.features import extract_features, thermal_scores, longest_minutes


def test_shuffled_columns_and_ids(wide):
    a = extract_features(from_wide(wide))
    b = extract_features(from_wide(wide.sample(frac=1, axis=1, random_state=9)))
    pd.testing.assert_series_equal(thermal_scores(a), thermal_scores(b))
    assert a.case.cars == [f"{i:02d}" for i in range(1, 9)]


def test_renaming_cars_is_equivariant(wide):
    mapping = {f"{i:02d}": f"{9 - i:02d}" for i in range(1, 9)}
    renamed = wide.rename(columns=lambda c: "Car " + mapping[c[4:6]] + c[6:] if c.startswith("Car ") and " - " in c else c)
    a = thermal_scores(extract_features(from_wide(wide)))
    b = thermal_scores(extract_features(from_wide(renamed)))
    for c in a.index:
        assert a[c] == pytest.approx(b[mapping[c]])


def test_quality_masks_are_distinct(wide):
    wide["Car 01 - Indoor Average Temperature"] = wide["Car 01 - Indoor Average Temperature"].astype(object)
    wide.loc[0, "Car 01 - Indoor Average Temperature"] = "None"
    wide.loc[1, "Car 01 - Indoor Average Temperature"] = "Invalid"
    wide.loc[2, "Car 01 - Indoor Average Temperature"] = -50
    wide = wide.drop(columns="Car 02 - Outdoor Average Temperature")
    c = from_wide(wide)
    f = c.native.loc[c.native.car.eq("01")]
    assert f.cabin_missing.iloc[0] and not f.cabin_invalid.iloc[0]
    assert f.cabin_invalid.iloc[1:3].all()
    assert c.native.loc[c.native.car.eq("02"), "ambient_absent"].all()
    assert c.frame.loc[c.frame.car.eq("02"), "ambient_shared"].notna().all()


def test_validity_masks_zero_temperature(wide):
    wide.loc[10, "Car 03 - Indoor Average Temperature"] = 0
    wide.loc[10, "Car 03 - ACV Information Valid"] = "Invalid"
    f = from_wide(wide).native.query("car == '03'")
    assert not f.eligible.iloc[10] and pd.isna(f.err.iloc[10])


def test_fill_only_bounded_short_same_state():
    s = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, 8.0])
    state = pd.DataFrame({"mode": ["cool"] * 8, "target": [24] * 8})
    out = _fill_short(s, state)
    assert out.iloc[1:3].tolist() == [2.0, 3.0] and out.iloc[4:7].isna().all()
    state.loc[2, "mode"] = "stop"
    assert _fill_short(s, state).iloc[1:3].isna().all()


def test_long_gap_splits_episode_and_slope(wide):
    wide = wide.drop(index=range(100, 140))
    f = from_wide(wide).frame.query("car == '03'")
    assert f.loc[f.index[140], "since_transition"] == 0
    assert pd.isna(f.loc[f.index[140], "slope_5"])


def test_unavailable_car_is_retained(wide):
    for col in wide:
        if col.startswith("Car 08 - "):
            wide[col] = "None"
    f = extract_features(from_wide(wide))
    assert "08" in f.table.index and not f.table.loc["08", "available"]


def test_rich_operating_invalid_does_not_discard(wide):
    wide = wide.rename(
        columns=lambda c: c.replace("Indoor Average Temperature", "Passenger Cabin Temperature Detected Value").replace(
            "ACV Control Temperature (Cooling)", "Target Temperature Value"
        )
    )
    wide = wide.drop(columns=[c for c in wide if c.endswith("ACV Information Valid")])
    wide["Car 03 - ACV Operating Mode"] = "Invalid"
    c = from_wide(wide)
    assert c.frame.query("car == '03'").eligible.all()


def test_ten_second_aggregation(wide):
    wide.Time = pd.date_range("2026-01-01", periods=len(wide), freq="10s")
    case = from_wide(wide)
    assert len(case.frame.query("car == '01'")) == 120
    assert case.metadata["dominant_interval_seconds"] == 10


def test_longest_run_handles_datetime_units():
    index = pd.date_range("2026-01-01", periods=5, freq="30s")
    assert longest_minutes(pd.Series([True] * 5), index) == 2.5
    assert longest_minutes(pd.Series([True] * 5), index.as_unit("us")) == 2.5


def test_reject_bad_time_and_duplicate_headers(wide):
    bad = wide.copy()
    bad.loc[1, "Time"] = bad.loc[0, "Time"]
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        from_wide(bad)
    with pytest.raises(ValueError, match="Time column"):
        from_wide(wide.drop(columns="Time"))


def test_missing_ambient_spike_and_short_gap_keep_complete_ranking(wide, baseline):
    from acv.inference import rank_case

    for col in [c for c in wide if c.endswith("Outdoor Average Temperature")]:
        wide[col] = np.nan
    wide.loc[40, "Car 03 - Indoor Average Temperature"] = 500.0
    wide.loc[70:71, "Car 03 - Indoor Average Temperature"] = np.nan
    features = extract_features(from_wide(wide))
    frame = features.case.frame.query("car == '03'")
    assert frame.ambient_shared.isna().all()
    assert frame.cabin_invalid.iloc[40] and pd.isna(frame.cabin.iloc[40])
    assert frame.cabin.iloc[70:72].notna().all()
    result = rank_case(features, baseline)
    assert result.ranked_cars[0] == "03"
    assert set(result.ranked_cars) == set(features.case.cars)
