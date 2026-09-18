import numpy as np
import pandas as pd
import pytest
from acv.data import from_wide
from acv.features import extract_features

@pytest.fixture
def wide():
    n=360;t=pd.date_range("2026-01-01",periods=n,freq="30s")
    d={"Car model":["S"]*n,"Train number":["001"]*n,"Time":t}
    for i in range(1,9):
        c=f"{i:02d}"
        for key,value in {"Indoor Average Temperature":24+(2 if i==3 else 0)+.1*np.sin(np.arange(n)/20),
                "ACV Control Temperature (Cooling)":24,"Outdoor Average Temperature":30,
                "ACV Running Mode":"Automatic Cooling","ACV Setting Mode":"Centralized Control",
                "ACV Information Valid":"Valid","Load Halved":"Normal"}.items():d[f"Car {c} - {key}"]=value
    return pd.DataFrame(d)

@pytest.fixture
def features(wide):return extract_features(from_wide(wide))

@pytest.fixture
def baseline(features):
    from acv.models import Model
    return {"baseline":Model("thermal",{"aggregation":"mean"}).fit({"case.xlsx":features},{"case.xlsx":"03"}),"challenger":None,"weight":0.}
