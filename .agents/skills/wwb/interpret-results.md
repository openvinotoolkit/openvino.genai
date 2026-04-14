---
name: wwb-interpret-results
description: Parse WWB metrics.csv, compute the similarity score, write commentary with per-sample variance analysis and score-band classification, note which patches/wheels were active, and persist results to agent-results/pipeline_state.json and wwb_result.json.
---

# Skill: Interpret Results

**Trigger:** After `run-benchmark.md`. Consumes `agent-results/wwb/metrics/metrics.csv`.
Produces `agent-results/wwb/wwb_result.json` and updates `agent-results/pipeline_state.json`.

---

## Step 1 — Parse Score

```bash
python scripts/parse_wwb_score.py \
  agent-results/wwb/metrics/metrics.csv \
  "${SIMILARITY_THRESHOLD:-0.9}" \
  2>&1 | tee agent-results/wwb/parse_score.log
```

If `parse_wwb_score.py` is not available, parse manually:

```python
import csv, statistics
from pathlib import Path

metrics_path = Path("agent-results/wwb/metrics/metrics.csv")
rows = list(csv.DictReader(metrics_path.open(newline="", encoding="utf-8")))

# Detect score column: "similarity", "score", "metric" etc.
score_col = None
for col in rows[0].keys():
    if "similar" in col.lower() or col.lower() in ("score", "metric", "sim"):
        score_col = col
        break

if score_col is None:
    print("[WWB-INTERPRET] Cannot detect score column. Columns:", list(rows[0].keys()))
    raise SystemExit(1)

scores = []
for r in rows:
    try:
        scores.append(float(r[score_col]))
    except (ValueError, KeyError):
        pass

mean_score = statistics.mean(scores) if scores else 0.0
stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0

print(f"[WWB-INTERPRET] Score column : {score_col}")
print(f"[WWB-INTERPRET] Samples      : {len(scores)}")
print(f"[WWB-INTERPRET] Mean score   : {mean_score:.4f}")
print(f"[WWB-INTERPRET] Std dev      : {stdev:.4f}")
```

## Step 2 — Score Band Classification

```python
THRESHOLD = float(open("agent-results/wwb/threshold.txt").read().strip()
                  if Path("agent-results/wwb/threshold.txt").exists() else "0.9")

if mean_score >= 0.95:
    band = "strong_pass"
    band_comment = (
        f"Score {mean_score:.4f} is well above threshold {THRESHOLD}. "
        "Accuracy is solid; OV model output closely matches the reference."
    )
elif mean_score >= 0.90:
    band = "pass"
    band_comment = (
        f"Score {mean_score:.4f} meets threshold {THRESHOLD}. Acceptable accuracy."
    )
elif mean_score >= 0.85:
    band = "borderline"
    band_comment = (
        f"Score {mean_score:.4f} is below threshold {THRESHOLD} but within noise range. "
        "Re-run with do_sample=False / greedy decoding to rule out non-determinism. "
        "If score does not improve, investigate logit differences in layers near the end of the model."
    )
elif mean_score >= 0.70:
    band = "low_pass"
    band_comment = (
        f"Score {mean_score:.4f} is notably below threshold {THRESHOLD}. "
        "Likely a systematic accuracy issue, not sampling noise. "
        "Inspect per-sample failures; check quantization configuration and padding mode."
    )
else:
    band = "accuracy_regression"
    band_comment = (
        f"Score {mean_score:.4f} strongly suggests an accuracy regression. "
        "This is not noise — mean this far below threshold indicates structural mismatch. "
        "Check: conversion flags, precision settings, graph transformations applied, "
        "and whether the OV model matches the original architecture exactly."
    )

passed = mean_score >= THRESHOLD
print(f"[WWB-INTERPRET] Band    : {band}")
print(f"[WWB-INTERPRET] Passed  : {passed}")
print(f"[WWB-INTERPRET] Comment : {band_comment}")
```

## Step 3 — Per-Sample Variance Analysis

```python
rows_with_scores = [(i, r, float(r[score_col])) for i, r in enumerate(rows) if r.get(score_col, "").strip()]
rows_with_scores.sort(key=lambda x: x[2])

# Lowest-scoring samples
worst_samples = rows_with_scores[:5]
best_samples = rows_with_scores[-3:]

print("[WWB-INTERPRET] Lowest-scoring samples:")
for idx, row, sc in worst_samples:
    prompt_key = next((k for k in row if "prompt" in k.lower() or "question" in k.lower() or "input" in k.lower()), None)
    prompt_preview = str(row.get(prompt_key, ""))[:80] if prompt_key else "(unknown)"
    print(f"  [{idx}] score={sc:.4f}  prompt={prompt_preview!r}")

# Check for score variance pattern
if stdev > 0.15:
    variance_note = (
        f"High variance (stdev={stdev:.4f}): scores range widely across samples. "
        "Specific input patterns may trigger failures — review worst samples above."
    )
elif stdev < 0.02 and mean_score < THRESHOLD:
    variance_note = (
        f"Low variance (stdev={stdev:.4f}) but below threshold: consistently low scores "
        "indicate a systemic issue, not outlier samples."
    )
else:
    variance_note = f"Normal variance (stdev={stdev:.4f})."

print(f"[WWB-INTERPRET] Variance note: {variance_note}")
```

## Step 4 — Note Active Patches / Wheels

```python
import json
from pathlib import Path

patches_note = ""
installed_pkg_path = Path("agent-results/wwb/installed_packages.json")
if installed_pkg_path.exists():
    installed = json.loads(installed_pkg_path.read_text())
    non_standard = {
        pkg: ver for pkg, ver in installed.items()
        if "+" in str(ver) or "dev" in str(ver).lower() or "git" in str(ver).lower()
    }
    wheel_sources = installed.get("_sources", {})
    if non_standard or wheel_sources:
        lines = []
        if wheel_sources.get("wheels"):
            lines.append(f"Custom wheels from Package Builder: {wheel_sources['wheels']}")
        if wheel_sources.get("install_cmd_patches"):
            lines.append(f"install_cmd patches applied: {wheel_sources['install_cmd_patches']}")
        for pkg, ver in non_standard.items():
            lines.append(f"  {pkg}=={ver}")
        patches_note = "Active non-standard packages: " + "; ".join(lines)
    else:
        patches_note = "All packages at standard PyPI releases."
else:
    patches_note = "installed_packages.json not found — package context unknown."

print(f"[WWB-INTERPRET] Patches: {patches_note}")
```

## Step 5 — Write `wwb_result.json`

```python
import json
from pathlib import Path

wwb_result = {
    "status": "passed" if passed else "failed",
    "score": round(mean_score, 6),
    "threshold": THRESHOLD,
    "passed": passed,
    "band": band,
    "num_samples": len(scores),
    "stdev": round(stdev, 6),
    "commentary": band_comment,
    "variance_note": variance_note,
    "patches_context": patches_note,
    "worst_samples": [
        {"index": idx, "score": round(sc, 4), "prompt_preview": str(row.get(next(iter(row)), ""))[:100]}
        for idx, row, sc in worst_samples
    ],
}

out_path = Path("agent-results/wwb/wwb_result.json")
out_path.write_text(json.dumps(wwb_result, indent=2))
print(f"[WWB-INTERPRET] Written: {out_path}")
```

## Step 6 — Update `agent-results/pipeline_state.json`

```python
import json
from pathlib import Path

state_path = Path("agent-results/pipeline_state.json")
state = json.loads(state_path.read_text()) if state_path.exists() else {}

state.setdefault("signals", {})
state["wwb_result"] = {
    "status": "passed" if passed else "failed",
    "score": round(mean_score, 6),
    "threshold": THRESHOLD,
    "passed": passed,
    "band": band,
    "commentary": band_comment,
}
state["signals"]["wwb_passed"] = passed
state["signals"]["wwb_score"] = round(mean_score, 6)

state_path.write_text(json.dumps(state, indent=2))
print(f"[WWB-INTERPRET] agent-results/pipeline_state.json updated — wwb_passed={passed} score={mean_score:.4f}")
```

## Step 7 — Final Summary

Print a human-readable summary for the agent log:

```python
separator = "=" * 60
print(separator)
print("WWB RESULT SUMMARY")
print(separator)
print(f"  Model      : {MODEL_ID}")
print(f"  Score      : {mean_score:.4f}  (threshold: {THRESHOLD})")
print(f"  Result     : {'PASS' if passed else 'FAIL'}  [{band}]")
print(f"  Samples    : {len(scores)}  (stdev: {stdev:.4f})")
print()
print(f"  Commentary : {band_comment}")
print()
print(f"  Patches    : {patches_note}")
print(separator)
```

---

**On completion:**
- `agent-results/wwb/wwb_result.json` — machine-readable result with commentary
- `agent-results/pipeline_state.json` updated with `wwb_result` and `signals.wwb_passed`

The WWB pipeline is complete. Return `wwb_result.json` content as the final agent output.
