---
name: wwb-bootstrap-env
description: Set up the environment for WhoWhatBench: install OpenVINO, optimum-intel, and WWB from source; then apply all available patches, branches, and wheels from previous specialist agents.
---

# Skill: Bootstrap WWB Environment

**Trigger:** Always first. Prepares the full software stack before any WWB step runs.
Produces `agent-results/wwb/bootstrap.log` and `agent-results/wwb/installed_packages.json`.

---

## Step 1 — Base Installation

Install stable release packages. WWB is always built from the openvino.genai repo
to get the latest benchmark runner.

```bash
pip install openvino openvino-tokenizers openvino-genai 2>&1 | tee -a agent-results/wwb/bootstrap.log
pip install git+https://github.com/huggingface/optimum-intel.git 2>&1 | tee -a agent-results/wwb/bootstrap.log
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu 2>&1 | tee -a agent-results/wwb/bootstrap.log

# WWB from source — always latest, ensures support for new optimum-intel pipeline_tags
git clone --depth 1 https://github.com/openvinotoolkit/openvino.genai.git /tmp/ov_genai_repo 2>&1 | tee -a agent-results/wwb/bootstrap.log
pip install /tmp/ov_genai_repo/tools/who_what_benchmark 2>&1 | tee -a agent-results/wwb/bootstrap.log
```

## Step 2 — Apply Patches and Branches from agent-results/pipeline_state.json

Parse `agent-results/pipeline_state.json` for all available fixes. Apply in priority order:

### Priority 1 — Pre-built wheels (Package Builder output)

Wheels contain all OV-core patches compiled in. Highest priority.

```python
import json, os, subprocess

state = {}
if os.path.exists("agent-results/pipeline_state.json"):
    state = json.load(open("agent-results/pipeline_state.json"))

wheels_installed = []
for w in state.get("artifacts", {}).get("wheels", []):
    path = w.get("path", "")
    if path and os.path.exists(path):
        result = subprocess.run(
            ["pip", "install", path, "--force-reinstall"],
            capture_output=True, text=True
        )
        status = "ok" if result.returncode == 0 else "failed"
        print(f"[WWB-BOOTSTRAP] wheel {path}: {status}")
        wheels_installed.append({"path": path, "status": status})
```

### Priority 2 — install_cmd entries from patch manifest

Captures `pip install git+https://...@branch` commands written by optimum-intel,
openvino-genai, and tokenizer agents.

```python
cmds_run = []
for patch in state.get("artifacts", {}).get("patches", []):
    cmd = patch.get("install_cmd", "").strip()
    if not cmd:
        continue
    print(f"[WWB-BOOTSTRAP] Applying: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    status = "ok" if result.returncode == 0 else f"failed ({result.returncode})"
    print(f"[WWB-BOOTSTRAP]   → {status}")
    if result.returncode != 0:
        print(f"[WWB-BOOTSTRAP]   stderr: {result.stderr[:500]}")
    cmds_run.append({"cmd": cmd, "status": status})
```

### Priority 3 — OpenVINO core patch files

Core C++ patches can only take effect via a compiled wheel. If a wheel was
installed in Priority 1, the patches are already included. If no wheel exists,
log a warning — WWB will use stable OV packages.

```python
for patch in state.get("artifacts", {}).get("patches", []):
    if patch.get("component") == "openvino" and not wheels_installed:
        patch_path = patch.get("path", "")
        print(f"[WWB-BOOTSTRAP] [WARN] OV core patch detected: {patch_path}")
        print(f"[WWB-BOOTSTRAP] [WARN] No compiled wheel available. Accuracy may not reflect core fix.")
        print(f"[WWB-BOOTSTRAP] [WARN] To include core fix: run Package Builder first, then re-run WWB.")
```

## Step 3 — Record Installed Versions

```python
import subprocess, json

packages = ["openvino", "optimum-intel", "openvino-genai", "openvino-tokenizers",
            "torch", "transformers", "who-what-benchmark"]
installed = {}
for pkg in packages:
    result = subprocess.run(
        ["pip", "show", pkg], capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if line.startswith("Version:"):
            installed[pkg] = line.split(":", 1)[1].strip()
            break
    else:
        installed[pkg] = "not installed"

json.dump(installed, open("agent-results/wwb/installed_packages.json", "w"), indent=2)
print("[WWB-BOOTSTRAP] Installed packages:")
for k, v in installed.items():
    print(f"  {k}: {v}")
```

---

**On completion:** `agent-results/wwb/installed_packages.json` is ready.
Proceed to `wwb/locate-model.md`.
