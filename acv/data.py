from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import io
import re
import numpy as np
import pandas as pd

HEADER = re.compile(r"^Car (\d+) - (.+)$")
COOLING = {"Automatic Cooling", "Full Cooling", "Half Cooling"}
ALIASES = {
    "cabin": ["Indoor Average Temperature", "Passenger Cabin Temperature Detected Value"],
    "target": ["ACV Control Temperature (Cooling)", "Target Temperature Value"],
    "ambient": ["Outdoor Average Temperature", "Outside Temperature Sensor Reading", "Fresh Air Temperature Detected Value"],
    "mode": ["ACV Running Mode"],
    "control": ["ACV Setting Mode", "ACV Control Mode"],
    "validity": ["ACV Information Valid"],
    "load": ["Load Halved", "Load Shedding"],
    "observation": ["Observation Area Temperature Detected Value"],
    **{f"p{k}_{side.lower()}": [f"Refrigeration System {k} {side} Pressure Value"] for k in [1, 2] for side in ["High", "Low"]},
    **{f"compressor{k}": [f"Compressor {k} Running"] for k in [1, 2]},
}
NUMERIC = [k for k in ALIASES if k not in {"mode", "control", "validity", "load"}]


@dataclass
class Case:
    file_id: str
    cars: list[str]
    frame: pd.DataFrame
    native: pd.DataFrame
    metadata: dict


def _fill_short(series, states, limit=2):
    """Fill only complete short holes bounded by valid, identical states."""
    s = series.copy()
    missing = s.isna().to_numpy()
    edges = np.diff(np.r_[False, missing, False].astype(int))
    for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True):
        if start == 0 or end == len(s) or end - start > limit:
            continue
        block = states.iloc[start - 1 : end + 1]
        if block.isna().any().any() or not block.eq(block.iloc[0]).all().all():
            continue
        s.iloc[start:end] = np.linspace(s.iloc[start - 1], s.iloc[end], end - start + 2)[1:-1]
    return s


def _mode(s):
    v = s.dropna()
    return v.mode().iloc[0] if len(v) else np.nan


def _resample_modes(frame):
    result = {}
    for col in frame:
        s = frame[col].dropna()
        if s.empty:
            result[col] = pd.Series(index=pd.DatetimeIndex([], name="time"), dtype=object)
            continue
        counts = s.groupby([s.index.floor("30s"), s], sort=True).size()
        # Stable sorting preserves lexical tie-breaking, identical to Series.mode().iloc[0].
        chosen = counts.sort_values(ascending=False, kind="stable").reset_index(name="count").drop_duplicates("time")
        result[col] = chosen.set_index("time")[col]
    out = pd.DataFrame(result)
    out.index = pd.DatetimeIndex(out.index, name="time")
    return out


def from_wide(df, file_id="case.xlsx", source_hash="memory"):
    if "Time" not in df:
        raise ValueError("Workbook must contain a Time column.")
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column headers are ambiguous.")
    mapping = {}
    for col in df:
        m = HEADER.match(str(col))
        if m:
            mapping.setdefault(m[1], {})[m[2]] = col
    if not mapping:
        raise ValueError("No 'Car <ID> - <parameter>' columns found.")
    time = pd.to_datetime(df.Time, errors="coerce")
    if time.isna().any():
        raise ValueError("Time contains missing or unparseable timestamps.")
    if time.duplicated().any():
        raise ValueError("Duplicate timestamps require source correction.")
    order = np.argsort(time.to_numpy(), kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    time = pd.DatetimeIndex(time.iloc[order], name="time")
    native = []
    for car, columns in sorted(mapping.items()):
        out = pd.DataFrame(index=time)
        for key, names in ALIASES.items():
            found = [columns[name] for name in names if name in columns]
            if len(found) > 1:
                raise ValueError(f"Ambiguous aliases for car {car}: {key}")
            raw = df[found[0]].astype(str).reset_index(drop=True) if found else pd.Series("", index=range(len(df)))
            raw.index = time
            out[key + "_absent"] = not bool(found)
            out[key + "_missing"] = raw.isna() | raw.isin(["", "None", "nan", "NaN", "NaT"])
            out[key + "_invalid"] = raw.eq("Invalid")
            if key in NUMERIC:
                val = pd.to_numeric(raw, errors="coerce")
                if key in {"cabin", "observation"}:
                    val = val.where(val.between(5, 50))
                if key == "target":
                    val = val.where(val.between(10, 35))
                if key == "ambient":
                    val = val.where(val.between(-10, 65))
                if key.startswith("p"):
                    val = val.where(val > 0)
                out[key + "_invalid"] |= val.isna() & ~out[key + "_missing"] & ~out[key + "_absent"]
                out[key] = val
            else:
                out[key] = raw.mask(out[key + "_missing"] | out[key + "_invalid"])
        out["car"] = car
        valid = out.validity_absent | out.validity.eq("Valid")
        out["valid"] = valid & out.cabin.notna() & out.target.notna()
        out["eligible"] = out.valid & out["mode"].isin(COOLING)
        out["err"] = (out.cabin - out.target).where(out.eligible)
        native.append(out)
    native = pd.concat(native)
    frames = []
    for car, f in native.groupby("car", sort=True):
        nums = f[NUMERIC].resample("30s").median()
        states = _resample_modes(f[["mode", "control", "validity", "load"]])
        masks = f[[c for c in f if c.endswith(("_missing", "_invalid", "_absent"))]].resample("30s").max()
        for c in masks:
            masks[c] = masks[c].fillna(bool(f[c].iloc[0]) if c.endswith("_absent") else c.endswith("_missing")).astype(bool)
        q = nums.join(states).join(masks)
        q["observed"] = f.valid.astype(int).resample("30s").max().reindex(q.index).fillna(0).astype(bool)
        q["invalid_events"] = (f.cabin_invalid | f.target_invalid | f.validity_invalid).astype(int).resample("30s").sum()
        q["mode_events"] = (f["mode"].ne(f["mode"].shift()) & f["mode"].notna()).astype(int).resample("30s").sum()
        match = q[["mode", "control", "target"]]
        for c in NUMERIC:
            # Never interpolate explicit invalid observations or setpoints/modes.
            if c == "target" or c.startswith("compressor"):
                continue
            filled = _fill_short(q[c], match)
            q[c] = filled.mask(q[c + "_invalid"])
        q["car"] = car
        valid = q.validity_absent | q.validity.eq("Valid")
        q["valid"] = valid & q.cabin.notna() & q.target.notna()
        q["eligible"] = q.valid & q["mode"].isin(COOLING)
        change = q["mode"].ne(q["mode"].shift()) | q.target.ne(q.target.shift()) | ~q.valid | ~q.valid.shift(fill_value=False)
        q["episode"] = change.cumsum()
        starts = pd.Series(q.index, index=q.index).groupby(q.episode).transform("min")
        q["since_transition"] = (pd.Series(q.index, index=q.index) - starts).dt.total_seconds()
        q["steady"] = q.eligible & q.since_transition.ge(300)
        q["err"] = (q.cabin - q.target).where(q.eligible)
        for minutes in (5, 15):
            lag = minutes * 2
            same = q.episode.eq(q.episode.shift(lag))
            q[f"slope_{minutes}"] = ((q.cabin - q.cabin.shift(lag)) / minutes).where(same & q.eligible)
        frames.append(q)
    frame = pd.concat(frames)
    meta = {
        "source_hash": source_hash,
        "rows": len(df),
        "columns": len(df.columns),
        "model_family": str(df["Car model"].iloc[0]) if "Car model" in df else "unknown",
        "train": str(df["Train number"].iloc[0]) if "Train number" in df else "unknown",
        "start": str(time.min()),
        "end": str(time.max()),
        "dominant_interval_seconds": float(time.to_series().diff().dt.total_seconds().mode().iloc[0]) if len(time) > 1 else None,
        "schema": "rich" if any("High Pressure" in str(c) for c in df) else "basic",
        "sorted_input": bool(np.array_equal(order, np.arange(len(df)))),
        "cars": sorted(mapping),
    }
    return enrich(Case(file_id, sorted(mapping), frame, native, meta))


def enrich(case):
    for attr in ("native", "frame"):
        f = getattr(case, attr).copy()
        error = f.pivot(columns="car", values="err")
        ambient = f.pivot(columns="car", values="ambient").median(axis=1)
        modes = f.pivot(columns="car", values="mode")
        for car in case.cars:
            loc = f.car.eq(car)
            own = f.loc[loc]
            peers = error.drop(columns=car)
            if attr == "frame":
                peers = peers.where(modes.drop(columns=car).eq(modes[car], axis=0))
            count = peers.count(axis=1)
            median = peers.median(axis=1).where(count >= 2)
            f.loc[loc, "peer_err"] = median.reindex(own.index).to_numpy()
            f.loc[loc, "peer_count"] = count.reindex(own.index).to_numpy()
            f.loc[loc, "ambient_shared"] = ambient.reindex(own.index).to_numpy()
        f["peer"] = f.err - f.peer_err
        setattr(case, attr, f)
    case.metadata["unavailable_cars"] = [c for c in case.cars if not case.frame.loc[case.frame.car.eq(c), "valid"].any()]
    return case


def load_case(source, file_id=None):
    if isinstance(source, (str, Path)):
        path = Path(source)
        content = path.read_bytes()
        name = file_id or path.name
    else:
        content = source.getvalue() if hasattr(source, "getvalue") else source.read()
        name = file_id or Path(getattr(source, "name", "uploaded.xlsx")).name
    book = pd.ExcelFile(io.BytesIO(content), engine="openpyxl")
    matches = []
    for sheet in book.sheet_names:
        headers = pd.read_excel(book, sheet_name=sheet, nrows=0).columns
        if "Time" in headers and any(HEADER.match(str(c)) for c in headers):
            matches.append(sheet)
    if len(matches) != 1:
        raise ValueError("Expected exactly one telemetry worksheet with Time and car headers.")
    df = pd.read_excel(book, sheet_name=matches[0], keep_default_na=False, dtype={"Train number": str})
    return from_wide(df, name, hashlib.sha256(content).hexdigest())
