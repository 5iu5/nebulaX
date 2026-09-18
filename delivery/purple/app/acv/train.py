import argparse
import json
from pathlib import Path
import joblib
from .common import config, save_json, versions, code_hash, digest
from .prepare import training_data
from .models import Model
from .features import LINEAR_FEATURES, TREE_FEATURES

def train(selection,cfg,device="auto"):
    data,labels=training_data(cfg)
    actual={n:f.case.metadata["source_hash"] for n,f in data.items()}
    if actual!=selection["case_hashes"]:raise ValueError("Training inputs differ from evaluated data")
    if selection["code_hash"]!=code_hash():raise ValueError("Code changed after evaluation; reevaluate before final training")
    base=Model("thermal",selection["baseline_params"]).fit(data,labels)
    challenger=None
    if selection["model"]!="thermal":challenger=Model(selection["model"],selection["params"]).fit(data,labels,device)
    bundle={"baseline":base,"challenger":challenger,"weight":selection["weight"],"selection":selection}
    root=Path(cfg["artifact_root"])/"final";root.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,root/"model.joblib",compress=3)
    manifest={"format_version":1,"code_hash":code_hash(),"model_sha256":digest(root/"model.joblib"),"versions":versions(),
              "training_files":sorted(data),"training_hashes":actual,"selection":selection,
              "feature_catalogue":{"linear":LINEAR_FEATURES,"tree":TREE_FEATURES},
              "quality_policy":"Unavailable cars retain neutral percentile 0.5; scores are not probabilities.",
              "pressure_policy":"Diagnostic only; no independently validated rich-schema cohort."}
    save_json(root/"manifest.json",manifest)
    print(f"Saved {root}")
    return root

def main():
    p=argparse.ArgumentParser();p.add_argument("--selection",default="outputs/acv/selection.json");p.add_argument("--config",default="configs/acv.yaml");p.add_argument("--device",default="auto")
    a=p.parse_args();train(json.loads(Path(a.selection).read_text()),config(a.config),a.device)
if __name__=="__main__":main()
