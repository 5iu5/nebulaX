import argparse
from pathlib import Path
import pandas as pd
from .data import HEADER

def validate(input_path,predictions):
    root=Path(input_path);files=sorted(root.glob("*.xlsx")) if root.is_dir() else [root]
    if not files: raise ValueError("No input workbooks")
    frame=pd.read_csv(predictions,dtype=str,keep_default_na=False)
    if frame.columns.tolist()!=["file_id","ranked_cars"]: raise ValueError("Expected exactly file_id,ranked_cars")
    if frame.file_id.duplicated().any(): raise ValueError("Duplicate prediction file IDs")
    if set(frame.file_id)!={p.name for p in files}:raise ValueError("Prediction filenames do not exactly match inputs")
    for path in files:
        book=pd.ExcelFile(path);matches=[]
        for sheet in book.sheet_names:
            cols=pd.read_excel(book,sheet_name=sheet,nrows=0).columns
            cars={HEADER.match(str(c))[1] for c in cols if HEADER.match(str(c))}
            if cars and "Time" in cols:matches.append(cars)
        if len(matches)!=1:raise ValueError("Ambiguous telemetry sheets")
        ranked=frame.loc[frame.file_id.eq(path.name),"ranked_cars"].iloc[0].split("|")
        if len(ranked)!=len(set(ranked)) or set(ranked)!=matches[0]:raise ValueError(f"Incomplete or invalid car IDs in {path.name}")
    return {"valid":True,"files":len(files)}

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--predictions",required=True)
    a=p.parse_args();print(validate(a.input,a.predictions))
if __name__=="__main__":main()
