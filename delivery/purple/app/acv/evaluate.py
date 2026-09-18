from __future__ import annotations
import argparse
import hashlib
import json
import time
import tempfile
from pathlib import Path
import joblib
from .common import config, code_hash, save_json, versions
from .prepare import training_data
from .models import Model, candidates, MODEL_NAMES
from .features import percentiles, ordered
from .metrics import case_result, summary, selection_key

class Evaluator:
    def __init__(self,data,labels,cfg,device="auto",deadline=None):
        self.data=data;self.labels=labels;self.cfg=cfg;self.device=device;self.deadline=deadline
        self.cache=Path(cfg["output_root"])/"fit_cache";self.cache.mkdir(parents=True,exist_ok=True)
        self.hash=code_hash(); self.predictions={}

    def get_model(self,name,params,train):
        provenance={n:self.data[n].case.metadata["source_hash"] for n in sorted(train)}
        key=hashlib.sha256(json.dumps([self.hash,name,params,provenance,{n:self.labels[n] for n in train}],sort_keys=True).encode()).hexdigest()
        path=self.cache/(key+".joblib")
        if path.exists(): return joblib.load(path)
        model=Model(name,params).fit({n:self.data[n] for n in train},self.labels,self.device,self.deadline)
        # Concurrent experiment processes must never see a partially written pickle.
        with tempfile.NamedTemporaryFile(dir=self.cache,suffix=".joblib",delete=False) as temp:
            temporary=Path(temp.name)
        try:
            joblib.dump(model,temporary,compress=3)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return model

    def predict(self,name,params,train,test):
        assert set(train).isdisjoint(test), "Case leakage detected"
        key=json.dumps([name,params,sorted(train),sorted(test)],sort_keys=True)
        if key not in self.predictions:
            model=self.get_model(name,params,train)
            assert set(model.training_files)==set(train)
            self.predictions[key]={n:percentiles(model.predict(self.data[n]),self.data[n].table.available) for n in test}
        return self.predictions[key]

    def cv(self,name,params,names):
        result={}
        for test in names:
            train=[n for n in names if n!=test]
            result.update(self.predict(name,params,train,[test]))
        return result

    def rows(self,preds):
        return [case_result(n,ordered(s),self.labels[n]) for n,s in sorted(preds.items())]

    def choose(self,name,names,grid=None):
        # Baseline tuning is also entirely inside the outer training partition.
        baseline=[]
        for params in candidates("thermal"):
            pred=self.cv("thermal",params,names);rows=self.rows(pred)
            baseline.append((selection_key(rows),params,pred,rows))
        _,base_params,base_pred,base_rows=max(baseline,key=lambda a:a[0])
        best={"model":"thermal","params":base_params,"baseline_params":base_params,"weight":0.,"inner_summary":summary(base_rows)}
        key=selection_key(base_rows,0)
        if name=="thermal": return best
        for params in grid or candidates(name):
            pred=self.cv(name,params,names)
            for weight in [.25,.5,.75,1.]:
                blended={n:(1-weight)*base_pred[n]+weight*pred[n] for n in names}
                rows=self.rows(blended);metrics=summary(rows)
                # Strict benefit over baseline; no worst-case regression.
                if metrics["primary_metric"] <= best_baseline(base_rows) or metrics["worst_rank"]>summary(base_rows)["worst_rank"]: continue
                candidate_key=selection_key(rows,MODEL_NAMES.index(name))
                if candidate_key>key:
                    key=candidate_key;best={"model":name,"params":params,"baseline_params":base_params,"weight":weight,"inner_summary":metrics}
        return best

    def apply(self,selection,train,test):
        base=self.predict("thermal",selection["baseline_params"],train,test)
        if selection["model"]=="thermal": return base
        pred=self.predict(selection["model"],selection["params"],train,test);w=selection["weight"]
        return {n:(1-w)*base[n]+w*pred[n] for n in test}

def best_baseline(rows): return summary(rows)["primary_metric"]

def evaluate(name,cfg,device="auto",max_epochs=100,quick=False):
    data,labels=training_data(cfg);names=sorted(data)
    deadline=time.monotonic()+cfg["neural_max_seconds"] if name=="tcn_mil" else None
    ev=Evaluator(data,labels,cfg,device,deadline)
    grid=candidates(name)
    if name=="tcn_mil": grid=[{"max_epochs":max_epochs,**({"seeds":[42]} if quick else {})}]
    if quick and name!="tcn_mil":grid=grid[:1]
    root=Path(cfg["output_root"])/"evaluation";root.mkdir(parents=True,exist_ok=True)
    report={"model":name,"code_hash":code_hash(),"versions":versions(),"status":"running","quick":quick,
            "candidate_grid":grid,"case_hashes":{n:data[n].case.metadata["source_hash"] for n in names},
            "outer_folds":[],"interpretation":"Development validation on six cases; initial exploratory analysis used all six. Not hidden-test accuracy."}
    path=root/(name+("_smoke" if quick else "")+".json")
    try:
        for test in names:
            train=[n for n in names if n!=test]
            print(f"{name}: outer holdout {test}",flush=True)
            choice=ev.choose(name,train,grid)
            result=ev.rows(ev.apply(choice,train,[test]))[0]
            report["outer_folds"].append({**result,"selection":choice,"training_files":train,
                                          "training_hashes":{n:data[n].case.metadata["source_hash"] for n in train}})
            save_json(path,report)
        report["summary"]=summary(report["outer_folds"])
        report["final_selection"]=ev.choose(name,names,grid)
        # Standalone challengers are reported even when a blend is rejected.
        standalone=[]
        for params in grid:
            rows=ev.rows(ev.cv(name,params,names));standalone.append({"params":params,"summary":summary(rows),"cases":rows})
        report["standalone_development_comparisons"]=standalone
        report["status"]="complete"
    except (ImportError,TimeoutError) as exc:
        report["status"]="unavailable" if isinstance(exc,ImportError) else "time_budget_exhausted"
        report["reason"]=str(exc)
    save_json(path,report)
    print(json.dumps({"model":name,"status":report["status"],"summary":report.get("summary"),"reason":report.get("reason")}),flush=True)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/acv.yaml")
    p.add_argument("--model",choices=MODEL_NAMES,required=True);p.add_argument("--nested",action="store_true")
    p.add_argument("--device",default="auto");p.add_argument("--max-epochs",type=int,default=100);p.add_argument("--quick",action="store_true")
    args=p.parse_args()
    report=evaluate(args.model,config(args.config),args.device,args.max_epochs,args.quick)
    if report["status"]!="complete": raise SystemExit(2)

if __name__=="__main__":main()
