---
name: optimum-intel-setup
description: "Set up the optimum-intel workspace for model enablement. Use when: starting model-enablement work in optimum-intel, a downstream agent or skill needs PATH_TO_OPTIMUM or the model-enabler.agent.md location, switching to a different fork or branch of optimum-intel."
argument-hint: "[fork_url] [branch] defaults: https://github.com/huggingface/optimum-intel.git main"
---

# Optimum-Intel Setup

Clones the optimum-intel repository into the workspace and resolves the path to the `model-enabler.agent.md` agent file so that downstream agents and skills can operate on it.

## Inputs

- **fork_url** (optional): GitHub repository URL to clone from (default: `https://github.com/huggingface/optimum-intel.git`)
- **branch** (optional): git branch, tag, or commit SHA to check out (default: `main`)

Use defaults unless the caller explicitly provides different values.

## Procedure

### Step 1: Prepare the Local Clone

#### 1.1: Check whether the clone exists

```bash
git -C .model_enabler/optimum-intel rev-parse --is-inside-work-tree 2>/dev/null
```

**If it does not exist** — clone and skip to Step 2:

```bash
git clone --depth=1 -b <branch> <fork_url> .model_enabler/optimum-intel
```

**If it exists** — continue with the steps below.

#### 1.2: Check for uncommitted changes

```bash
git -C .model_enabler/optimum-intel status --porcelain
```

If there are uncommitted changes, stop and report:

> Uncommitted changes found in `.model_enabler/optimum-intel`. Commit or stash them before switching branches.

#### 1.3: Verify the origin remote points to the requested fork

```bash
git -C .model_enabler/optimum-intel remote get-url origin
```

If it does not match `<fork_url>`, stop and report:

> Origin remote URL does not match the requested fork_url. Current: `<current_url>`, requested: `<fork_url>`. Update the remote URL or provide the correct fork_url.

#### 1.4: Check if the requested branch is already checked out

```bash
git -C .model_enabler/optimum-intel rev-parse --abbrev-ref HEAD
```

If it matches `<branch>`, the requested branch is already active. Skip to Step 2.

#### 1.5: Fetch and switch to the requested branch

```bash
git -C .model_enabler/optimum-intel fetch origin <branch>
git -C .model_enabler/optimum-intel switch <branch>
```

### Step 2: Resolve the Agent Path

The model-enabler agent is at a fixed location inside the cloned repository:

```
.model_enabler/optimum-intel/.github/agents/model-enabler.agent.md
```

Verify the file exists:

```bash
test -f .model_enabler/optimum-intel/.github/agents/model-enabler.agent.md && echo "found" || echo "missing"
```

If the file is **missing**, stop and report:

> `model-enabler.agent.md` not found at `.model_enabler/optimum-intel/.github/agents/model-enabler.agent.md` on branch `<branch>`. The repository may not contain the agent yet.

## Output

Return the following values for use by the calling agent or skill:

- **PATH_TO_OPTIMUM**: `.model_enabler/optimum-intel`
- **MODEL_ENABLER_AGENT**: `.model_enabler/optimum-intel/.github/agents/model-enabler.agent.md`
- **branch**: the active branch checked out
- **fork_url**: the URL that was cloned
