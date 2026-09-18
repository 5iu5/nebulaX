from __future__ import annotations
import hashlib
import json
from pathlib import Path
import importlib.metadata
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

def config(path="configs/acv.yaml"):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    result = yaml.safe_load(path.read_text())
    for key in ("data_root", "output_root", "artifact_root"):
        result[key] = str(ROOT / result[key])
    return result

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def code_hash():
    h = hashlib.sha256()
    for p in sorted((ROOT / "acv").glob("*.py")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    h.update((ROOT / "configs/acv.yaml").read_bytes())
    return h.hexdigest()

def jsonable(x):
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return jsonable(x.tolist())
    if isinstance(x, np.generic): return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    return x

def save_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True) + "\n")

def versions():
    result = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "torch", "streamlit"]:
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = "not installed"
    return result
