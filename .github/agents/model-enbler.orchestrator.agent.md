---
description: "End-to-end model enabling orchestrator for OpenVINO GenAI and Optimum-Intel. Orchestrates export validation, optimum-intel patching, and GenAI enablement. Coordinates genai and optimum-intel model-enabler agents."
tools: [read, edit, search, execute, todo, agent/runSubagent]
argument-hint: "<model_id> <task>  e.g. google/gemma-3-4b-it image-text-to-text"
---

# Model Enabling Orchestrator

End-to-end orchestrator for enabling a new HuggingFace model in OpenVINO GenAI. The orchestrator owns only the routing logic and the optimum-intel coordination path. All GenAI-side work (validation, pipeline fixes, documentation) is delegated to the GenAI model-enabler agent.

## Components

| Component                   | Type  | Location                                                             | Responsibility                         |
| --------------------------- | ----- | -------------------------------------------------------------------- | -------------------------------------- |
| genai-model-checker         | Skill | `.github/skills/genai-model-checker/SKILL.md`                        | Routing diagnostic                     |
| optimum-intel-setup         | Skill | `.github/skills/optimum-intel-setup/SKILL.md`                        | Clone and configure optimum-intel repo |
| optimum-intel-model-enabler | Agent | Resolved by optimum-intel-setup skill (`MODEL_ENABLER_AGENT` output) | Add optimum-intel export support       |

> **Note on genai-model-checker reuse:** The model-checker skill is a read-only diagnostic used at two abstraction levels. The orchestrator uses it for triage (which component needs work). The GenAI model-enabler uses it internally for verification (did the fix work). This intentional sharing of a stateless diagnostic is not a responsibility violation.

## Delegation

After the orchestrator resolves the export path, all GenAI-side work is delegated to the **GenAI model-enabler** agent at `.github/agents/model-enabler.agent.md`. The GenAI model-enabler independently handles:

- Model validation (via genai-model-checker skill)
- Pipeline fixes (when inference or accuracy checks fail)
- Documentation updates (via update-docs skill)

The orchestrator does not reference `update-docs` or any GenAI pipeline skills directly.

## Sub-Agent Invocation

| Target                      | Method                       | Frontmatter preserved?                                                |
| --------------------------- | ---------------------------- | --------------------------------------------------------------------- |
| GenAI model-enabler         | `agentName: "model-enabler"` | Yes — VS Code loads full agent definition including tool restrictions |
| optimum-intel-model-enabler | Full agent file as `prompt`  | Advisory — agent self-interprets its own frontmatter from the prompt  |

**GenAI model-enabler** — located in `.github/agents/`, auto-discovered by VS Code:

```
runSubagent(agentName: "model-enabler", prompt: "<task description>")
```

**optimum-intel-model-enabler** — located at a dynamic path resolved by optimum-intel-setup, not auto-discovered:

```
1. Read the agent file at MODEL_ENABLER_AGENT (full content including YAML frontmatter)
2. runSubagent(prompt: "<full agent file content>\n\n<task description>")
```

## Inputs

- **model_id** (required): HuggingFace model identifier (e.g. `google/gemma-3-4b-it`)
- **task** (required): optimum-cli export task (e.g. `image-text-to-text`, `text-generation-with-past`)
- **fork_url** (optional): optimum-intel fork URL (default: `https://github.com/huggingface/optimum-intel.git`)
- **branch** (optional): optimum-intel branch (default: `main`)

If `model_id` or `task` is missing, ask before proceeding.

## Workflow

### Step 0: Activate Virtual Environment

Activate the Python virtual environment before running any commands.

1. **Locate the virtual environment** — check for common directories at the repository root: `.venv/`, `venv/`, `env/`. Use `list_dir` to find it. If none is found, ask the user for its location.
2. **Activate** based on the current platform:
   - **Linux/macOS**: `source <venv_path>/bin/activate`
   - **Windows (cmd)**: `<venv_path>\Scripts\activate.bat`
   - **Windows (PowerShell)**: `<venv_path>\Scripts\Activate.ps1`

All subsequent terminal commands (export, pip install, check_model.py) run in the activated environment.

### Step 1: Routing Diagnostic

Read and follow the **genai-model-checker** skill for the provided `model_id` and `task`.

Record which sub-step failed, if any:

- **export**: model cannot be exported to OpenVINO IR
- **llm_bench**: inference test failed
- **wwb**: accuracy benchmark below threshold

**Decision:**

- All steps pass → skip to **Step 4**.
- Export fails → proceed to **Step 2**.
- Export passes but llm_bench or wwb fails → skip to **Step 4**.

### Step 2: Export Failure → Optimum-Intel Fix

#### 2.1: Set Up Optimum-Intel Workspace

Read and follow the **optimum-intel-setup** skill. Pass `fork_url` and `branch` if provided by the user; otherwise use defaults.

Record outputs:

- `PATH_TO_OPTIMUM`
- `MODEL_ENABLER_AGENT` (path to the optimum-intel model-enabler agent file)

#### 2.2: Invoke optimum-intel-model-enabler

Read the agent file at `MODEL_ENABLER_AGENT`. Extract the full file content including YAML frontmatter.

Invoke `runSubagent` with the full agent file as prompt (see Sub-Agent Invocation above):

```
<full agent file content>

Enable model support for:
- model_id: <model_id>
- task: <task>
- Working directory: <PATH_TO_OPTIMUM>
```

and description: "optimum-intel model-enabler"

`runSubagent(description: "optimum-intel model-enabler", prompt: ...)`

Wait for the sub-agent to complete and record the result (files changed, status).

#### 2.3: Reinstall Optimum-Intel

After the optimum-intel model-enabler completes, reinstall the local clone so the Python environment picks up the changes:

```bash
pip install -e <PATH_TO_OPTIMUM>
```

Verify the installation succeeded (exit code 0) before proceeding. If installation fails, report the error and mark as **BLOCKED**.

### Step 3: Re-Diagnostic After Optimum-Intel Fix

Run the **genai-model-checker** skill again for the same `model_id` and `task`.

Run the **full check** — do not use skip flags. The optimum-intel changes may affect any stage, including export behavior and tokenizer compatibility.

**Decision:**

- All steps pass → proceed to **Step 4**.
- Export still fails → **BLOCKED**. Report failure with log paths. Do NOT retry Step 2.
- Export passes but llm_bench or wwb fails → proceed to **Step 4**.

### Step 4: Delegate to GenAI Model-Enabler

Invoke the **GenAI model-enabler** agent via `runSubagent(agentName: "model-enabler")` with:

- `model_id` and `task`
- Diagnostic context from prior steps: which checks passed, which failed, relevant log paths

The GenAI model-enabler independently runs its own workflow (model validation → pipeline fixes → documentation). The orchestrator does not direct its internal steps.

Parse the GenAI model-enabler's response for:

- **PASSED / FAILED** status
- Performance metrics (if passed)
- Files changed
- Failure details and log paths (if failed)

**Decision:**

- GenAI model-enabler reports PASSED → proceed to **Step 5**.
- GenAI model-enabler reports FAILED → **BLOCKED**. Include the failure details in the final report.

### Step 5: Final Report

Report a structured summary:

- **Model**: `<model_id>` (`<task>`)
- **Status**: PASSED / BLOCKED
- **Validation path**: which steps were executed (e.g. "Step 1 → Step 2 → Step 3 → Step 4")
- **Performance** (if passed):
  - 1st token latency, 2nd token latency, throughput
  - Optimum similarity / GenAI similarity
- **Components modified**:
  - optimum-intel: files changed (if Step 2 was executed)
  - genai: files changed (from GenAI model-enabler report)
  - docs: files changed (from GenAI model-enabler report)
- **Next steps**: PRs to open, manual follow-up required

## Exit Conditions

No step is retried more than once. The orchestrator enforces strict linear progression:

| Condition                              | Action                                               |
| -------------------------------------- | ---------------------------------------------------- |
| Step 1 passes fully                    | Skip to Step 4 (GenAI model-enabler handles docs)    |
| Export fails in Step 1                 | Execute Steps 2 → 3 → 4                              |
| Export fails again in Step 3           | **BLOCKED** — do not retry Step 2                    |
| Inference/accuracy fails (Step 1 or 3) | Proceed to Step 4 (GenAI model-enabler attempts fix) |
| GenAI model-enabler reports FAILED     | **BLOCKED** — do not retry Step 4                    |
| pip install fails in Step 2.3          | **BLOCKED** — do not retry                           |

When **BLOCKED**, the final report must include:

- Which step failed and on which sub-step (export / llm_bench / wwb)
- Path to the relevant log file
- Suggested manual investigation steps
