---
name: wwb-generate-gt
description: Generate ground-truth reference outputs from the original HuggingFace model. Reuses existing gt.csv if valid. If WWB cannot generate GT for this model, patches WWB sources to add support and retries, or falls back to Analyze-and-Convert for inference samples.
---

# Skill: Generate Ground Truth

**Trigger:** After `locate-model.md`. Produces `agent-results/wwb/gt.csv`.

---

## Step 1 — Check for Existing GT

```bash
GT_CSV="agent-results/wwb/gt.csv"
GT_VALID=false

# Check working dir
if [ -f "${GT_CSV}" ]; then
  LINE_COUNT=$(wc -l < "${GT_CSV}")
  if [ "${LINE_COUNT}" -gt 1 ]; then
    GT_VALID=true
    echo "[WWB-GT] Reusing existing gt.csv (${LINE_COUNT} lines)"
  else
    echo "[WWB-GT] gt.csv found but empty — regenerating"
  fi
fi

# Check analyze-and-convert output
if [ "${GT_VALID}" = "false" ] && [ -f "agent-results/analyze-and-convert/gt.csv" ]; then
  AC_LINES=$(wc -l < "agent-results/analyze-and-convert/gt.csv")
  if [ "${AC_LINES}" -gt 1 ]; then
    cp "agent-results/analyze-and-convert/gt.csv" "${GT_CSV}"
    GT_VALID=true
    echo "[WWB-GT] Reused gt.csv from Analyze-and-Convert (${AC_LINES} lines)"
  fi
fi
```

## Step 2 — Generate GT with WWB

```bash
if [ "${GT_VALID}" = "false" ]; then
  MODEL_TYPE=$(cat agent-results/wwb/model_type.txt 2>/dev/null || echo "text")
  echo "[WWB-GT] Generating ground truth: model=${MODEL_ID} type=${MODEL_TYPE} samples=${NUM_SAMPLES}"

  wwb \
    --base-model "$MODEL_ID" \
    --gt-data "${GT_CSV}" \
    --model-type "${MODEL_TYPE}" \
    --num-samples "${NUM_SAMPLES:-32}" \
    --hf \
    --trust-remote-code \
    2>&1 | tee agent-results/wwb/wwb_gt.log
  GT_EXIT=$?

  if [ "${GT_EXIT}" -eq 0 ] && [ -f "${GT_CSV}" ] && [ "$(wc -l < "${GT_CSV}")" -gt 1 ]; then
    GT_VALID=true
    echo "[WWB-GT] GT generation successful"
  else
    echo "[WWB-GT] GT generation failed (exit ${GT_EXIT})"
  fi
fi
```

## Step 3 — Self-Patch WWB If GT Failed

If GT generation failed, inspect `wwb_gt.log` to determine why and patch WWB sources.

```python
import re, subprocess, sys
from pathlib import Path

log = Path("agent-results/wwb/wwb_gt.log").read_text(errors="replace") if Path("agent-results/wwb/wwb_gt.log").exists() else ""

# Detect cause
CAUSE = "unknown"
if re.search(r"task.*not.*support|unsupported.*task|pipeline.*not.*found", log, re.I):
    CAUSE = "unsupported_pipeline_tag"
elif re.search(r"chat_template|apply_chat_template|TokenizerError", log, re.I):
    CAUSE = "missing_chat_template"
elif re.search(r"OutOfMemory|OOM|CUDA out of memory|Cannot allocate", log, re.I):
    CAUSE = "oom"
elif re.search(r"ImportError|ModuleNotFoundError", log, re.I):
    CAUSE = "missing_dependency"

print(f"[WWB-GT] Failure cause: {CAUSE}")
```

### Patch strategy by cause

**`unsupported_pipeline_tag`** — WWB `who_what_benchmark` does not know this model type.
Find the WWB source, add a pipeline registration:

```python
import subprocess, sys
from pathlib import Path

# Locate WWB source
result = subprocess.run(["pip", "show", "who-what-benchmark"], capture_output=True, text=True)
for line in result.stdout.splitlines():
    if line.startswith("Location:"):
        wwb_root = Path(line.split(":", 1)[1].strip()) / "who_what_benchmark"
        break
else:
    print("[WWB-GT] Cannot locate WWB source — cannot self-patch")
    sys.exit(1)

# Read the model factory file and add model type support
# WWB registers model types in text_evaluation.py or similar
factory_file = wwb_root / "text_evaluation.py"
if factory_file.exists():
    content = factory_file.read_text()
    print(f"[WWB-GT] Patching WWB source at: {factory_file}")
    # Append model type alias if not already present
    model_type = open("agent-results/wwb/model_type.txt").read().strip()
    # Agent: read the file, understand the registration pattern, add the missing type
    # The patch should map the model's pipeline_tag to the nearest supported WWB model class
    print(f"[WWB-GT] Model type to register: {model_type}")
    print(f"[WWB-GT] Read {factory_file} and extend the class/type registry")
else:
    print(f"[WWB-GT] Expected factory file not found at {factory_file}")
    print("[WWB-GT] List WWB source files:")
    for f in sorted(wwb_root.rglob("*.py")):
        print(f"  {f}")
    print("[WWB-GT] Identify the correct registration point and patch it")
```

After patching: retry Step 2.

**`missing_chat_template`** — model needs an explicit chat template override:

```bash
# Add --tokenizer and --chat-template flags and retry
wwb \
  --base-model "$MODEL_ID" \
  --tokenizer "$MODEL_ID" \
  --gt-data "${GT_CSV}" \
  --model-type "${MODEL_TYPE}" \
  --num-samples "${NUM_SAMPLES:-32}" \
  --hf \
  --trust-remote-code \
  2>&1 | tee agent-results/wwb/wwb_gt_retry.log
```

**`oom`** — reduce sample count and retry:

```bash
echo "[WWB-GT] OOM detected — retrying with 8 samples"
wwb \
  --base-model "$MODEL_ID" \
  --gt-data "${GT_CSV}" \
  --model-type "${MODEL_TYPE}" \
  --num-samples 8 \
  --hf \
  --trust-remote-code \
  2>&1 | tee agent-results/wwb/wwb_gt_retry.log
```

**`unknown` / still failing** — fall back to Analyze and Convert:

```
[WWB-GT] All GT strategies failed. Invoking Analyze and Convert to obtain inference samples.
```

Invoke Analyze and Convert agent with context:
> "Generate inference samples for WhoWhatBench ground truth for model `<MODEL_ID>`.
> Write 32 prompt/response pairs to `agent-results/analyze-and-convert/gt.csv`.
> Use the model's native tokenizer and standard text-generation pipeline."

After it returns, copy the resulting CSV to `agent-results/wwb/gt.csv` and
verify it has > 1 line. If still missing, stop with status `blocked` and write:

```json
{
  "status": "blocked",
  "reason": "cannot_generate_gt",
  "detail": "WWB and Analyze-and-Convert both failed to produce ground truth. Manual inspection required."
}
```

## Step 4 — Validate GT

```python
import csv
from pathlib import Path

gt_path = Path("agent-results/wwb/gt.csv")
if not gt_path.exists():
    print("[WWB-GT] [ERROR] gt.csv missing")
    raise SystemExit(1)

with open(gt_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"[WWB-GT] gt.csv: {len(rows)} samples")
if rows:
    print(f"[WWB-GT] Sample columns: {list(rows[0].keys())}")
    print(f"[WWB-GT] First prompt:   {str(rows[0])[:120]}")
```

---

**On completion:** `agent-results/wwb/gt.csv` is valid (>1 line).
Proceed to `wwb/run-benchmark.md`.
