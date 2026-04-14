---
name: WWB Agent
description: Runs WhoWhatBench accuracy evaluation for a model exported to OpenVINO IR. Bootstraps any available patches/branches from optimum-intel, openvino, or openvino-genai before running. Re-invokes Analyze-and-Convert to obtain inference samples if they are missing. Modifies the WWB invocation if the default text pipeline does not work for this model type.
model: claude-sonnet-4.6
tools: ['read/readFile', 'write/editFile', 'memory', 'terminal']
---
# WWB Agent

## Role

Runs the [WhoWhatBench](https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/who_what_benchmark)
accuracy benchmark for a HuggingFace model that has been exported to OpenVINO IR.

Measures generation similarity between the original HF model and the OV model.
If similarity ≥ threshold (default 0.9): pipeline passes.
If similarity < threshold: reports the score and stops — escalation is handled by the orchestrator.

**Does NOT:** write code, create PRs, or fix model quality issues.
Its sole purpose is **accurate measurement** and a clear structured result.

## Called by

- **Common Orchestrator** (Step 5 — after successful deploy or specialist fix)

## Agents (callable)

| Agent | When |
|-------|------|
| **Analyze and Convert** | Re-run if `gt.csv` is missing and no previous inference samples exist |

---

## Inputs

The context file (or orchestrator prompt) should provide:

| Field | Required | Description |
|-------|----------|-------------|
| `model_id` | yes | HuggingFace model ID, e.g. `Qwen/Qwen3-4B` |
| `ov_model_path` | no | Local path to the exported OV IR directory (default: `ov_model/`) |
| `similarity_threshold` | no | Minimum passing score (default: `0.9`) |
| `num_samples` | no | Number of benchmark samples (default: `32`) |
| `patches_dir` | no | Path to patches directory (default: `agent-results/openvino-orchestrator/patches/`) |
| `pipeline_state_path` | no | Path to `agent-results/pipeline_state.json` (default: `agent-results/pipeline_state.json`) |

---

## State

Reads `agent-results/pipeline_state.json` (if present) at startup to discover:
- `artifacts.patches` — list of patch file paths + `install_cmd` entries
- `artifacts.wheels` — pre-built wheels from Package Builder
- `ov_orchestrator.pr_url` — open PR branch URL for OV fork
- `signals` — any package override URLs

Writes results back to `agent-results/pipeline_state.json` under `wwb_result`:
```json
"wwb_result": {
  "similarity_score": 0.94,
  "accuracy_ok": true,
  "threshold": 0.9,
  "num_samples": 32,
  "model_type": "text",
  "patched_packages": ["optimum-intel@fix/qwen3"],
  "timestamp": "ISO 8601"
}
```

Writes `agent-results/wwb/wwb_result.json`:
```json
{
  "similarity_score": 0.94,
  "accuracy_ok": true,
  "threshold": 0.9,
  "num_samples": 32,
  "patched_packages": [],
  "gt_csv": "agent-results/wwb/gt.csv",
  "metrics_csv": "agent-results/wwb/metrics/metrics.csv",
  "gt_log": "agent-results/wwb/wwb_gt.log",
  "score_log": "agent-results/wwb/wwb_score.log"
}
```

---

## Skills

Execute in order. Each skill produces files consumed by the next.

| Order | Skill file | Purpose |
|-------|-----------|---------|
| 1 | `skills/wwb/bootstrap-env.md` | Install packages, apply patches/wheels/branches by priority |
| 2 | `skills/wwb/locate-model.md` | Find or re-export OV IR; detect WWB model type |
| 3 | `skills/wwb/generate-gt.md` | Generate GT from HF model; Analyze-and-Convert fallback |
| 4 | `skills/wwb/run-benchmark.md` | Run WWB; retry strategies; auto-update WWB source if needed |
| 5 | `skills/wwb/interpret-results.md` | Parse score; write commentary; update agent-results/pipeline_state.json |

---

## Execution Model

> The detailed execution steps live in the skill files listed above.
> The inline phases below are legacy reference only.

### Phase 0: Bootstrap Environment

Install base packages with stable pinned versions:
```bash
pip install openvino openvino-tokenizers openvino-genai
pip install git+https://github.com/huggingface/optimum-intel.git
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

# Install WWB
git clone --depth 1 https://github.com/openvinotoolkit/openvino.genai.git /tmp/openvino_genai_repo
pip install /tmp/openvino_genai_repo/tools/who_what_benchmark
```

### Phase 1: Bootstrap Patches and Branches

Parse `agent-results/pipeline_state.json` → collect all available fixes in priority order:

**1a. Wheels (highest priority — pre-built with all fixes)**
```bash
# Install any wheels from Package Builder first
for WHEEL_PATH in $(python3 -c "
import json
state = json.load(open('agent-results/pipeline_state.json'))
for w in state.get('artifacts', {}).get('wheels', []):
    if w.get('path') and os.path.exists(w['path']):
        print(w['path'])
" 2>/dev/null); do
  pip install "${WHEEL_PATH}" --force-reinstall
  echo "[WWB] Installed wheel: ${WHEEL_PATH}"
done
```

**1b. Install commands from patch manifest (optimum-intel, openvino-genai)**
```bash
# Read install_cmd from each patch artifact
python3 -c "
import json
state = json.load(open('agent-results/pipeline_state.json'))
for p in state.get('artifacts', {}).get('patches', []):
    cmd = p.get('install_cmd', '')
    if cmd:
        print(cmd)
" 2>/dev/null | while read CMD; do
  echo "[WWB] Applying: ${CMD}"
  eval "${CMD}"
done
```

**1c. Apply patch files for non-pip components (openvino core)**
```bash
if [ -d "agent-results/openvino-orchestrator/patches/openvino" ]; then
  for PATCH in agent-results/openvino-orchestrator/patches/openvino/*.patch; do
    [ -f "${PATCH}" ] || continue
    echo "[WWB] Note: OpenVINO core patch detected: ${PATCH}"
    echo "[WWB] Core patches require a pre-built wheel from Package Builder to take effect."
    echo "[WWB] If no wheel is available, WWB will use stable release OV packages."
  done
fi
```

Log all installed versions:
```bash
echo "[WWB] Installed packages:"
pip show openvino optimum-intel openvino-genai 2>/dev/null | grep -E "^(Name|Version):"
```

### Phase 2: Locate or Export OV Model

Check for available OV IR in this order:
1. `ov_model/` in working directory
2. Path from `agent-results/pipeline_state.json` → `artifacts.model_ir`
3. `agent-results/analyze-and-convert/ov_model_*/`

```bash
OV_MODEL_PATH=""
if [ -d "ov_model" ] && [ -f "ov_model/openvino_model.xml" ]; then
  OV_MODEL_PATH="ov_model"
  echo "[WWB] Using existing OV IR at ov_model/"
elif [ -f "agent-results/pipeline_state.json" ]; then
  OV_MODEL_PATH=$(python3 -c "
import json, os
state = json.load(open('agent-results/pipeline_state.json'))
p = state.get('artifacts', {}).get('model_ir')
print(p if p and os.path.exists(str(p)+'/openvino_model.xml') else '')
" 2>/dev/null)
fi

if [ -z "${OV_MODEL_PATH}" ]; then
  echo "[WWB] No OV IR found — exporting now with current package stack"
  TASK=$(python scripts/detect_task.py "$MODEL_ID" 2>/dev/null || echo "text-generation-with-past")
  optimum-cli export openvino \
    --model "$MODEL_ID" \
    --task "$TASK" \
    --weight-format fp16 \
    --trust-remote-code \
    ov_model 2>&1 | tee agent-results/wwb/export.log
  OV_MODEL_PATH="ov_model"
fi
```

### Phase 3: Determine Model Type and WWB Args

WWB supports multiple model types. Choose based on the model's `pipeline_tag`:

| `pipeline_tag` | `--model-type` | Notes |
|---|---|---|
| `text-generation`, `text2text-generation` | `text` | Default; uses LLM generation |
| `image-text-to-text` | `visual-text` | Vision-language models |
| `text-to-image` | `image` | Stable Diffusion variants |
| `automatic-speech-recognition` | `speech-to-text` | ASR models |

```bash
MODEL_TYPE=$(python3 -c "
import json
try:
    info = json.load(open('agent-results/analyze-and-convert/model_profile.json'))
    tag = info.get('pipeline_tag', 'text-generation')
except Exception:
    tag = 'text-generation'

mapping = {
    'text-generation': 'text',
    'text2text-generation': 'text',
    'image-text-to-text': 'visual-text',
    'text-to-image': 'image',
    'automatic-speech-recognition': 'speech-to-text',
}
print(mapping.get(tag, 'text'))
" 2>/dev/null || echo "text")

echo "[WWB] Model type: ${MODEL_TYPE}"
```

### Phase 4: Generate Ground Truth from HF Baseline

Check if a valid `gt.csv` already exists (from a previous Analyze-and-Convert run):

```bash
GT_CSV="agent-results/wwb/gt.csv"
GT_VALID=false

if [ -f "${GT_CSV}" ]; then
  LINE_COUNT=$(wc -l < "${GT_CSV}")
  [ "${LINE_COUNT}" -gt 1 ] && GT_VALID=true
  echo "[WWB] Reusing existing gt.csv (${LINE_COUNT} lines)"
fi

# Also check analyze-and-convert output
if [ "${GT_VALID}" = "false" ] && [ -f "agent-results/analyze-and-convert/gt.csv" ]; then
  cp "agent-results/analyze-and-convert/gt.csv" "${GT_CSV}"
  GT_VALID=true
  echo "[WWB] Reused gt.csv from Analyze-and-Convert"
fi
```

If no valid `gt.csv`:
```bash
if [ "${GT_VALID}" = "false" ]; then
  echo "[WWB] Generating ground truth from HF model"
  wwb \
    --base-model "$MODEL_ID" \
    --gt-data "${GT_CSV}" \
    --model-type "${MODEL_TYPE}" \
    --num-samples "$NUM_SAMPLES" \
    --hf \
    --trust-remote-code \
    2>&1 | tee agent-results/wwb/wwb_gt.log
  GT_EXIT=$?
  if [ "${GT_EXIT}" -ne 0 ]; then
    echo "[WWB] [WARN] GT generation failed (exit ${GT_EXIT}) — see wwb_gt.log"
    # Attempt fallback: invoke Analyze and Convert for inference samples
    _fallback_gt_via_analyze_convert
  fi
fi
```

#### Fallback: GT via Analyze and Convert

If GT generation fails (e.g. model needs custom input format, special tokenizer):

```
[WWB] GT generation failed. Invoking Analyze and Convert to obtain inference samples.
```

Invoke the **Analyze and Convert** agent with:
- `MODEL_ID` set to the model
- Instruction: "Generate inference samples for WWB ground truth. Write sample prompts and responses to agent-results/analyze-and-convert/gt.csv"

After Analyze and Convert returns, retry GT generation using any prompt file it produced.
If GT still cannot be generated, stop with:
```json
{
  "status": "blocked",
  "reason": "cannot_generate_gt",
  "message": "WWB ground truth generation failed. Model may require custom inference inputs. See wwb_gt.log."
}
```

### Phase 5: Run Accuracy Benchmark

```bash
mkdir -p agent-results/wwb/metrics
wwb \
  --target-model "${OV_MODEL_PATH}" \
  --gt-data "${GT_CSV}" \
  --model-type "${MODEL_TYPE}" \
  --num-samples "$NUM_SAMPLES" \
  --output agent-results/wwb/metrics \
  --trust-remote-code \
  2>&1 | tee agent-results/wwb/wwb_score.log || true
```

#### If WWB command fails (non-zero exit, no metrics.csv)

Common causes and fixes:

**Cause 1: Unknown model type for `--model-type`**
- Try `--model-type text` as universal fallback.
- Log: `[WWB] Retrying with --model-type text (fallback)`

**Cause 2: OV model uses custom ops not in stable release**
- OV core patches are present but no wheel was built.
- Log: `[WWB] OV core patches detected but no wheel available. Accuracy may be unreliable.`
- Continue with stable release packages; note in result.

**Cause 3: Missing tokenizer or chat template**
- Add `--tokenizer "$MODEL_ID"` to the WWB call.
- Log: `[WWB] Retrying with explicit --tokenizer flag`

**Cause 4: OOM / too many samples**
- Retry with `--num-samples 8`.
- Log: `[WWB] OOM detected — retrying with num_samples=8`

For each retry, log the modified command before running.

### Phase 6: Parse Score and Write Result

```bash
python scripts/parse_wwb_score.py \
  agent-results/wwb/metrics/metrics.csv \
  "$SIMILARITY_THRESHOLD" >> /tmp/wwb_outputs.txt

SIMILARITY_SCORE=$(grep "^similarity_score=" /tmp/wwb_outputs.txt | cut -d= -f2)
ACCURACY_OK=$(grep "^accuracy_ok=" /tmp/wwb_outputs.txt | cut -d= -f2)
```

Write `agent-results/wwb/wwb_result.json`:
```python
import json, datetime
result = {
    "similarity_score": float(similarity_score),
    "accuracy_ok": accuracy_ok == "true",
    "threshold": float(threshold),
    "num_samples": int(num_samples),
    "model_type": model_type,
    "patched_packages": patched_packages,   # list of "component@branch" strings
    "gt_csv": "agent-results/wwb/gt.csv",
    "metrics_csv": "agent-results/wwb/metrics/metrics.csv",
    "gt_log": "agent-results/wwb/wwb_gt.log",
    "score_log": "agent-results/wwb/wwb_score.log",
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
}
json.dump(result, open("agent-results/wwb/wwb_result.json", "w"), indent=2)
```

Update `agent-results/pipeline_state.json` with `wwb_result` block.

Log final result:
```
[WWB] similarity_score=0.943 accuracy_ok=true threshold=0.9 model_type=text patched=1
[WWB] Result: PASS
```
or:
```
[WWB] similarity_score=0.812 accuracy_ok=false threshold=0.9 model_type=text patched=0
[WWB] Result: FAIL — escalate to OV Orchestrator
```

---

## Output

Results are written to `agent-results/wwb/`:

| File | Description |
|------|-------------|
| `wwb_result.json` | Structured result with score, pass/fail, metadata |
| `gt.csv` | Ground truth prompt/response pairs |
| `metrics/metrics.csv` | Per-sample similarity scores |
| `wwb_gt.log` | GT generation log |
| `wwb_score.log` | Benchmark measurement log |
| `export.log` | Export log (only if OV model was generated by this agent) |
| `session.md` | Full agent session transcript |

## Creating Pull Requests

When your work is complete and all tests pass:

1. Create a new branch with a descriptive name: `agent/<short-description>`
2. Commit all changes with a clear, conventional commit message
3. Push the branch to the fork
4. Create a **Draft PR** to the upstream repository using `gh pr create`:
   ```
   gh pr create --draft \
     --title "[Agent] <descriptive title>" \
     --body "<description of changes, link to related PRs if any>" \
     --repo <upstream-org>/<repo-name>
   ```
5. Add the label `agent-generated` if the label exists
6. Output the PR URL for tracking

Refer to the [submit-draft-pr](skills/submit-draft-pr.md) skill for detailed instructions.