import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/"outputs/acv"
selection=json.loads((out/"selection.json").read_text())
lines=["\n### Completed local run\n",f"Selected model: **{selection['model']}**; challenger weight {selection['weight']}.\n",
       "Each nested row evaluates the inner-selected baseline/challenger blend, including fallback to thermal when the challenger fails the acceptance rule. It is not the standalone challenger score.\n\n",
       "| Experiment | Status | Nested rank score | Top-1 |\n|---|---|---:|---:|\n"]
notes=[]
for path in sorted((out/"evaluation").glob("*.json")):
    r=json.loads(path.read_text())
    if "model" not in r:continue
    s=r.get("summary",{})
    lines.append(f"| {r['model']} | {r['status']} | {s.get('primary_metric','—')} | {s.get('top1_accuracy','—')} |\n")
    if r.get("reason"):notes.append(f"\n{r['model']}: {r['reason']}\n\n")
    if r['model']=="tcn_mil" and r.get("standalone_development_comparisons"):
        tcn=r['standalone_development_comparisons'][0]['summary']
        notes.append(f"\nStandalone temporal CNN: development LOCO rank score {tcn['primary_metric']:.4f}, top-1 {tcn['top1_accuracy']:.1%}. This CPU experiment did not improve the baseline and was not retained.\n\n")
lines.extend(notes)
lines.append("\nThe model-family selector's nested development summary is:\n\n```json\n"+json.dumps(selection['validation_summary'],indent=2)+"\n```\n")
lines.append("\nThese scores use held-out development cases, not the hidden test labels. Classical comparisons ran on the Mac; CUDA access on the separate PC was not configured. The model artifact records the exact dependencies, training hashes and preprocessing.\n")
robust=json.loads((out/"robustness.json").read_text()) if (out/"robustness.json").exists() else {}
if robust:
    lines.append("\n#### Robustness evidence\n\n")
    lines.append("Group stress tests keep the selected hyperparameters fixed; they are not additional unbiased model-selection estimates.\n\n")
    for name,result in robust["grouped"].items():
        lines.append(f"- Leave-one-{name}-out: rank score {result['summary']['primary_metric']:.4f}, top-1 {result['summary']['top1_accuracy']:.1%}.\n")
    fragility=robust.get("bootstrap",{}).get("acv_case_05.xlsx",{}).get("cars",{}).get("04",{})
    if fragility:
        lines.append(f"\nCase 05 remains fragile: car 04 ranks first in {fragility['top1_frequency']:.1%} of synchronized hourly-evidence bootstrap samples. This diagnostic is not a fault probability or a confidence interval for the full-case model.\n")
    lines.append("\nOperating-state-conditioned circuit and peer pressure comparisons are in `rich_diagnostics.json`. Their source units are unspecified and they do not change the production ranking.\n")
if (out/"tests.xml").exists():
    import xml.etree.ElementTree as ET
    suite=ET.parse(out/"tests.xml").getroot().find("testsuite")
    lines.append(f"\n#### Verification\n\nPytest: {suite.attrib['tests']} tests, {suite.attrib['failures']} failures, {suite.attrib['errors']} errors. The suite includes all seven workbooks, leading-zero IDs, column/ID invariance, invalid and missing data, timing gaps, the score formula, fold separation, artifact integrity, neural car permutation, and app/CLI CSV parity.\n")
lines.append("\nThe user is recording the required live demo manually. Packaging checks that it exists and lasts no more than 180 seconds; staged files are marked incomplete until that video is supplied.\n")
path=ROOT/"docs/ACV_METHOD.md"
text=path.read_text().split("\n### Completed local run")[0]
path.write_text(text+"".join(lines))
print(path)
