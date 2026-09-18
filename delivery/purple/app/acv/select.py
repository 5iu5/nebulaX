import argparse
import json
from pathlib import Path
from .common import config, save_json, code_hash
from .metrics import summary
from .models import MODEL_NAMES

def key(choice):
    m=choice["inner_summary"]
    return (m["primary_metric"],-m["worst_rank"],m["top1_accuracy"],-MODEL_NAMES.index(choice["model"]))

def select(results,cfg):
    reports=[];excluded=[]
    for path in sorted(Path(results).glob("*.json")):
        r=json.loads(path.read_text())
        if "outer_folds" not in r: continue
        if r.get("status")!="complete" or r.get("quick"):
            excluded.append({"file":path.name,"status":r.get("status"),"reason":r.get("reason","smoke run")});continue
        reports.append(r)
    if not any(r["model"]=="thermal" for r in reports): raise ValueError("Complete thermal nested evaluation is required")
    baseline=next(r for r in reports if r["model"]=="thermal")
    for r in reports:
        if r["case_hashes"]!=baseline["case_hashes"] or r["code_hash"]!=baseline["code_hash"]:
            raise ValueError("Evaluation reports have different data/code versions; rerun evaluation before selection")
    # Select model families using INNER evidence for each outer fold, never the outer label.
    outer=[]
    for b in baseline["outer_folds"]:
        contenders=[next(f for f in r["outer_folds"] if f["file_id"]==b["file_id"]) for r in reports]
        winner=max(contenders,key=lambda f:key(f["selection"]))
        outer.append(winner)
    final=max([r["final_selection"] for r in reports],key=key)
    selection={**final,"validation_summary":summary(outer),"nested_selector_cases":outer,
               "baseline_summary":baseline["summary"],"available_models":[r["model"] for r in reports],
               "excluded_models":excluded,"case_hashes":baseline["case_hashes"],"code_hash":baseline["code_hash"],
               "interpretation":"Nested development validation; six independent cases. Hidden-test accuracy is unknown."}
    save_json(Path(cfg["output_root"])/"selection.json",selection)
    print(json.dumps({k:selection[k] for k in ["model","params","weight","validation_summary","excluded_models"]},indent=2))
    return selection

def main():
    p=argparse.ArgumentParser();p.add_argument("--results",default="outputs/acv/evaluation");p.add_argument("--config",default="configs/acv.yaml")
    a=p.parse_args();select(a.results,config(a.config))
if __name__=="__main__":main()
