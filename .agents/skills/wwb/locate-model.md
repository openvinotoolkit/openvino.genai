---
name: wwb-locate-model
description: Locate the OpenVINO IR model to benchmark — from local filesystem, agent-results/pipeline_state.json, or by re-exporting with current patched packages.
---

# Skill: Locate OV Model

**Trigger:** After `bootstrap-env.md`. Finds or creates the OV IR to benchmark.
Produces `OV_MODEL_PATH` (environment variable / shell variable used by subsequent skills).

---

## Step 1 — Check for Existing IR

Search in priority order:

```bash
OV_MODEL_PATH=""

# 1. Standard local path
if [ -d "ov_model" ] && [ -f "ov_model/openvino_model.xml" ]; then
  OV_MODEL_PATH="ov_model"
  echo "[WWB-LOCATE] Found IR at ov_model/"
fi

# 2. agent-results/pipeline_state.json → artifacts.model_ir
if [ -z "${OV_MODEL_PATH}" ] && [ -f "agent-results/pipeline_state.json" ]; then
  CANDIDATE=$(python3 -c "
import json, os
state = json.load(open('agent-results/pipeline_state.json'))
p = state.get('artifacts', {}).get('model_ir')
if p and isinstance(p, str) and os.path.isfile(os.path.join(p, 'openvino_model.xml')):
    print(p)
" 2>/dev/null)
  if [ -n "${CANDIDATE}" ]; then
    OV_MODEL_PATH="${CANDIDATE}"
    echo "[WWB-LOCATE] Found IR from agent-results/pipeline_state.json at ${OV_MODEL_PATH}"
  fi
fi

# 3. analyze-and-convert output directory
if [ -z "${OV_MODEL_PATH}" ]; then
  for DIR in agent-results/analyze-and-convert/ov_model_*/; do
    if [ -f "${DIR}openvino_model.xml" ]; then
      OV_MODEL_PATH="${DIR}"
      echo "[WWB-LOCATE] Found IR from analyze-and-convert at ${OV_MODEL_PATH}"
      break
    fi
  done
fi
```

## Step 2 — Re-export If No IR Found

If no prior IR exists, export the model now using the fully patched package stack
from `bootstrap-env.md`. This ensures the benchmark tests the actual fixed state.

```bash
if [ -z "${OV_MODEL_PATH}" ]; then
  echo "[WWB-LOCATE] No OV IR found — exporting with current package stack"

  # Detect task
  TASK=$(python scripts/detect_task.py "$MODEL_ID" 2>/dev/null || echo "text-generation-with-past")
  echo "[WWB-LOCATE] Detected task: ${TASK}"

  # Export
  optimum-cli export openvino \
    --model "$MODEL_ID" \
    --task "${TASK}" \
    --weight-format fp16 \
    --trust-remote-code \
    ov_model 2>&1 | tee agent-results/wwb/export.log

  EXPORT_EXIT=$?
  if [ "${EXPORT_EXIT}" -ne 0 ] || [ ! -f "ov_model/openvino_model.xml" ]; then
    echo "[WWB-LOCATE] [ERROR] Export failed (exit ${EXPORT_EXIT})"
    echo "[WWB-LOCATE] See agent-results/wwb/export.log for details"
    echo "[WWB-LOCATE] If export fails with missing ops, ensure the correct patches/wheels are installed"
    exit 1
  fi

  OV_MODEL_PATH="ov_model"
  echo "[WWB-LOCATE] Export succeeded → ${OV_MODEL_PATH}"
fi

export OV_MODEL_PATH
echo "[WWB-LOCATE] OV model path: ${OV_MODEL_PATH}"
```

## Step 3 — Detect Model Type

Determine the `--model-type` argument for WWB based on the model profile
(produced by `probe-model.md` if available, otherwise from HF metadata).

```python
import json, os

MODEL_TYPE = "text"  # default

# Try model_profile.json from analyze-and-convert
for profile_path in [
    "agent-results/analyze-and-convert/model_profile.json",
    "agent-results/wwb/model_profile.json",
]:
    if os.path.exists(profile_path):
        profile = json.load(open(profile_path))
        tag = profile.get("pipeline_tag", "")
        print(f"[WWB-LOCATE] pipeline_tag from profile: {tag}")
        break
else:
    # Fallback: read from agent-results/pipeline_state.json or HF API
    tag = ""
    if os.path.exists("agent-results/pipeline_state.json"):
        state = json.load(open("agent-results/pipeline_state.json"))
        tag = state.get("model_info", {}).get("pipeline_tag", "")

MAPPING = {
    "text-generation":               "text",
    "text2text-generation":          "text",
    "image-text-to-text":            "visual-text",
    "text-to-image":                 "image",
    "automatic-speech-recognition":  "speech-to-text",
    "audio-classification":          "speech-to-text",
}
MODEL_TYPE = MAPPING.get(tag, "text")
print(f"[WWB-LOCATE] WWB model type: {MODEL_TYPE}")

# Write for subsequent skills
with open("agent-results/wwb/model_type.txt", "w") as f:
    f.write(MODEL_TYPE)
```

---

**On completion:** `OV_MODEL_PATH` and `agent-results/wwb/model_type.txt` are set.
Proceed to `wwb/generate-gt.md`.
