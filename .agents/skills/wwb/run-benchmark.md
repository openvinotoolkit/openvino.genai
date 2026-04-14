---
name: wwb-run-benchmark
description: Run WhoWhatBench against the OpenVINO IR model using ground-truth CSV. Applies retry strategies for model-type mismatch, OOM, missing tokenizer, and unknown model architectures. Auto-updates WWB from source if the model type is not yet supported by the installed version.
---

# Skill: Run Benchmark

**Trigger:** After `generate-gt.md`. Consumes `agent-results/wwb/gt.csv` and OV IR path.
Produces `agent-results/wwb/metrics/metrics.csv`.

---

## Step 1 — Prepare Run Parameters

```python
import json
from pathlib import Path

# Read model info
model_type = (Path("agent-results/wwb/model_type.txt").read_text().strip()
              if Path("agent-results/wwb/model_type.txt").exists() else "text")

# Find OV IR path
ov_model_path = None
if Path("ov_model").is_dir():
    ov_model_path = "ov_model"
elif Path("agent-results/pipeline_state.json").exists():
    state = json.loads(Path("agent-results/pipeline_state.json").read_text())
    ov_model_path = state.get("artifacts", {}).get("model_ir")

if not ov_model_path:
    candidates = sorted(Path("agent-results/analyze-and-convert").glob("ov_model_*"))
    if candidates:
        ov_model_path = str(candidates[-1])

if not ov_model_path:
    print("[WWB-BENCH] [ERROR] No OV IR found — cannot run benchmark")
    raise SystemExit(1)

num_samples = int(os.environ.get("NUM_SAMPLES", "32"))
gt_csv = "agent-results/wwb/gt.csv"
output_dir = "agent-results/wwb/metrics"

print(f"[WWB-BENCH] ov_model_path = {ov_model_path}")
print(f"[WWB-BENCH] model_type    = {model_type}")
print(f"[WWB-BENCH] num_samples   = {num_samples}")
print(f"[WWB-BENCH] gt_csv        = {gt_csv}")
```

## Step 2 — Run WWB (Attempt 1: Primary)

```bash
MODEL_TYPE=$(cat agent-results/wwb/model_type.txt 2>/dev/null || echo "text")
OV_MODEL_PATH="<from step 1>"

mkdir -p agent-results/wwb/metrics

wwb \
  --target-model "${OV_MODEL_PATH}" \
  --gt-data agent-results/wwb/gt.csv \
  --model-type "${MODEL_TYPE}" \
  --num-samples 32 \
  --output agent-results/wwb/metrics \
  --trust-remote-code \
  2>&1 | tee agent-results/wwb/wwb_bench.log

BENCH_EXIT=$?
```

## Step 3 — Diagnose Failure and Retry

Parse `wwb_bench.log` to select the correct retry strategy.

```python
import re
from pathlib import Path

log = Path("agent-results/wwb/wwb_bench.log").read_text(errors="replace") if Path("agent-results/wwb/wwb_bench.log").exists() else ""
metrics_ok = Path("agent-results/wwb/metrics/metrics.csv").exists() and \
             Path("agent-results/wwb/metrics/metrics.csv").stat().st_size > 0

if metrics_ok:
    print("[WWB-BENCH] Metrics produced on attempt 1")
else:
    # Classify failure
    if re.search(r"task.*not.*support|unsupported.*task|no.*pipeline.*for|KeyError.*pipeline", log, re.I):
        STRATEGY = "wrong_model_type"
    elif re.search(r"OutOfMemory|OOM|CUDA out of memory|Cannot allocate|RuntimeError.*memory", log, re.I):
        STRATEGY = "oom"
    elif re.search(r"tokenizer.*not.*found|TokenizerError|cannot.*load.*tokenizer", log, re.I):
        STRATEGY = "missing_tokenizer"
    elif re.search(r"AttributeError.*forward|NotImplementedError|pipeline.*not.*implemented", log, re.I):
        STRATEGY = "unsupported_architecture"
    else:
        STRATEGY = "generic_failure"
    print(f"[WWB-BENCH] Strategy: {STRATEGY}")
```

### Retry A — Wrong Model Type → Fall Back to `text`

When WWB does not recognize the model's pipeline tag:

```bash
echo "[WWB-BENCH] Retry A: forcing --model-type text"
wwb \
  --target-model "${OV_MODEL_PATH}" \
  --gt-data agent-results/wwb/gt.csv \
  --model-type text \
  --num-samples 32 \
  --output agent-results/wwb/metrics \
  --trust-remote-code \
  2>&1 | tee agent-results/wwb/wwb_bench_retryA.log
```

### Retry B — OOM → Reduce Sample Count

```bash
echo "[WWB-BENCH] Retry B: OOM — reducing to 8 samples"
wwb \
  --target-model "${OV_MODEL_PATH}" \
  --gt-data agent-results/wwb/gt.csv \
  --model-type "${MODEL_TYPE}" \
  --num-samples 8 \
  --output agent-results/wwb/metrics \
  --trust-remote-code \
  2>&1 | tee agent-results/wwb/wwb_bench_retryB.log
```

### Retry C — Missing Tokenizer → Add `--tokenizer`

```bash
echo "[WWB-BENCH] Retry C: adding explicit --tokenizer"
wwb \
  --target-model "${OV_MODEL_PATH}" \
  --tokenizer "${MODEL_ID}" \
  --gt-data agent-results/wwb/gt.csv \
  --model-type "${MODEL_TYPE}" \
  --num-samples 32 \
  --output agent-results/wwb/metrics \
  --trust-remote-code \
  2>&1 | tee agent-results/wwb/wwb_bench_retryC.log
```

## Step 4 — Auto-Update WWB From Source

If all retries fail with `unsupported_architecture`, `unknown_model_type`, or the error references a missing method or pipeline class, it likely means this model architecture is not yet supported in the installed WWB release.

**Strategy: re-install WWB from openvino.genai HEAD.**

```python
import subprocess, sys
from pathlib import Path

# Locate the openvino.genai clone used during bootstrap-env
genai_repo = None
for candidate in [
    "/tmp/ov_genai_repo",
    "build/ov_genai_repo",
    str(Path.home() / "ov_genai_repo"),
]:
    if Path(candidate).is_dir():
        genai_repo = candidate
        break

if not genai_repo:
    print("[WWB-BENCH] Cloning openvino/openvino.genai for fresh WWB install")
    result = subprocess.run(
        ["git", "clone", "--depth=1", "https://github.com/openvinotoolkit/openvino.genai.git", "/tmp/ov_genai_repo"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[WWB-BENCH] Clone failed: {result.stderr}")
        sys.exit(1)
    genai_repo = "/tmp/ov_genai_repo"
else:
    print(f"[WWB-BENCH] Pulling latest changes in {genai_repo}")
    subprocess.run(["git", "-C", genai_repo, "pull", "--rebase"], capture_output=True)

# Re-install WWB from latest source
result = subprocess.run(
    ["pip", "install", f"{genai_repo}/tools/who_what_benchmark", "--force-reinstall", "--quiet"],
    capture_output=True, text=True
)
if result.returncode == 0:
    print("[WWB-BENCH] WWB re-installed from source")
else:
    print(f"[WWB-BENCH] Re-install failed: {result.stderr}")
    sys.exit(1)
```

After re-installing, retry the full benchmark command (Step 2).

```bash
echo "[WWB-BENCH] Retry D: using updated WWB from HEAD"
wwb \
  --target-model "${OV_MODEL_PATH}" \
  --gt-data agent-results/wwb/gt.csv \
  --model-type "${MODEL_TYPE}" \
  --num-samples "${NUM_SAMPLES:-32}" \
  --output agent-results/wwb/metrics \
  --trust-remote-code \
  2>&1 | tee agent-results/wwb/wwb_bench_retryD.log
```

If Retry D also fails with `unsupported_architecture` or a missing class,
read the new WWB source files to understand the registration/factory pattern,
then add support for this model architecture directly in the installed WWB code
(update the file in site-packages) and retry one final time.

## Step 5 — Self-Patch WWB Source for Unsupported Architecture

```python
import subprocess
from pathlib import Path

# Find WWB package location
result = subprocess.run(["pip", "show", "who-what-benchmark"], capture_output=True, text=True)
wwb_root = None
for line in result.stdout.splitlines():
    if line.startswith("Location:"):
        wwb_root = Path(line.split(":", 1)[1].strip()) / "who_what_benchmark"

if wwb_root and wwb_root.exists():
    print("[WWB-BENCH] WWB source files:")
    for f in sorted(wwb_root.rglob("*.py")):
        print(f"  {f.relative_to(wwb_root.parent)}")
    print("[WWB-BENCH] Read the factory/registry file, understand the pattern, then:")
    print(f"[WWB-BENCH] MODEL_TYPE = {open('agent-results/wwb/model_type.txt').read().strip()}")
    print("[WWB-BENCH] Add the model type registration and retry Step 2")
```

## Step 6 — Write Attempt Summary

```python
import json
from pathlib import Path

metrics_ok = (
    Path("agent-results/wwb/metrics/metrics.csv").exists() and
    Path("agent-results/wwb/metrics/metrics.csv").stat().st_size > 0
)

summary = {
    "benchmark_produced_metrics": metrics_ok,
    "attempts": [],  # fill in based on which log files exist
}

for label, log_path in [
    ("primary", "agent-results/wwb/wwb_bench.log"),
    ("retry_A_text_fallback", "agent-results/wwb/wwb_bench_retryA.log"),
    ("retry_B_oom", "agent-results/wwb/wwb_bench_retryB.log"),
    ("retry_C_tokenizer", "agent-results/wwb/wwb_bench_retryC.log"),
    ("retry_D_updated_wwb", "agent-results/wwb/wwb_bench_retryD.log"),
]:
    if Path(log_path).exists():
        summary["attempts"].append(label)

Path("agent-results/wwb/bench_summary.json").write_text(json.dumps(summary, indent=2))
print(f"[WWB-BENCH] metrics_ok={metrics_ok}  attempts={summary['attempts']}")

if not metrics_ok:
    Path("agent-results/wwb/wwb_result.json").write_text(json.dumps({
        "status": "blocked",
        "reason": "benchmark_failed_all_retries",
        "attempts": summary["attempts"],
        "detail": "No metrics.csv produced after all retry strategies"
    }, indent=2))
    raise SystemExit(1)
```

---

**On completion:** `agent-results/wwb/metrics/metrics.csv` exists and is non-empty.
Proceed to `wwb/interpret-results.md`.
