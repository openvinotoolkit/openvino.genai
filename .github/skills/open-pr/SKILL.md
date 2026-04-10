---
name: open-pr
description: "Open a pull request to openvinotoolkit/openvino.genai. Use when: submitting changes, creating PR, opening pull request, pushing work for review, finalizing a task with a PR."
argument-hint: "<branch_name> <pr_title> — e.g. fix/kv-cache-leak Fix KV-cache memory leak in continuous batching"
---

# Open Pull Request

Creates a branch, commits only task-related changes, and opens a pull request against `openvinotoolkit/openvino.genai` with a description aligned to the repository PR template.

## When to Use

- After completing a coding task that needs to be submitted as a PR
- When the user asks to open, create, or submit a pull request

## Inputs

The user provides (or the agent infers from prior conversation):

- **branch_name**: Name for the new branch (e.g. `fix/kv-cache-leak`)
- **pr_title**: Concise PR title for release notes (e.g. `Fix KV-cache memory leak in continuous batching`)
- **ticket** _(optional)_: Jira ticket number (e.g. `CVS-12345`) or GitHub issue number
- **base_branch** _(optional)_: Target branch to merge into. Defaults to `master`.

If the user does not provide these, infer sensible values from the conversation context and confirm with the user before proceeding.

## Procedure

### Step 1: Identify Current State

Determine the current branch and its relationship to `master`:

```bash
git branch --show-current
git log --oneline master..HEAD
git status
```

This reveals:

- **Current branch name** — may already be a feature branch, not `master`
- **Commits ahead of master** — prior commits that may or may not be in scope
- **Uncommitted changes** — staged and unstaged modifications

### Step 2: Analyze All Changes Against Master

Compare the **full delta** from `master` to the current state (both committed and uncommitted):

```bash
git diff master --stat
git diff master
```

Examine every changed file in this diff. Determine which changes are **in scope** (directly related to the task) and which are **out of scope** (unrelated formatting, debug leftovers, scratch files, unrelated refactors, changes from prior unrelated commits).

If out-of-scope changes are detected, inform the user and ask whether to:

- Revert or exclude the out-of-scope changes before committing
- Include them anyway

**Do not silently include unrelated changes.**

### Step 3: Prepare the Branch

Based on the analysis from Steps 1–2, determine the scenario:

**Scenario A — Already on a feature branch with only in-scope changes (committed and/or uncommitted):**
Reuse the current branch as `<branch_name>`. No branch switching needed. Proceed to Step 4 to commit any remaining uncommitted changes.

**Scenario B — On `master` with uncommitted in-scope changes only:**
Create a new branch from `master`:

```bash
git checkout -b <branch_name>
```

**Scenario C — Out-of-scope changes detected (any branch):**
Create a new branch from `master` with only the in-scope changes:

```bash
git stash  # if there are uncommitted in-scope changes
git checkout master
git checkout -b <branch_name>
# apply only in-scope changes (e.g. git stash pop, git add only in-scope files)
```

If the branch name already exists, ask the user whether to reuse it or pick a different name.

The goal is a branch where the **full diff against `master`** contains only in-scope changes.

### Step 4: Stage and Commit

Stage only the in-scope files identified in Step 1:

```bash
git add <file1> <file2> ...
```

Do **not** use `git add .` or `git add -A` unless every change is confirmed in-scope.

Write a clear, conventional commit message. Use a single-line summary (≤72 chars):

```bash
git commit -m "<summary>"
```

### Step 5: Determine the Fork Remote

The branch must be pushed to the user's **fork**, never directly to `openvinotoolkit/openvino.genai`.

Identify the correct remote by inspecting remote URLs:

```bash
git remote -v
```

Look for a remote whose URL contains the **user's GitHub username** (not `openvinotoolkit`). Common setups:

| Remote name | URL contains                     | Use for push?             |
| ----------- | -------------------------------- | ------------------------- |
| `origin`    | `<username>/openvino.genai`      | Yes — this is the fork    |
| `upstream`  | `openvinotoolkit/openvino.genai` | No — never push here      |
| `origin`    | `openvinotoolkit/openvino.genai` | No — find the fork remote |

If no fork remote is found, ask the user which remote to use. **Do not guess.**

### Step 6: Push the Branch

```bash
git push <fork_remote> <branch_name>
```

If the push is rejected (e.g. branch exists on remote), ask the user how to proceed.

### Step 7: Read the PR Template

Read the repository pull request template to get the latest structure:

```
.github/pull_request_template.md
```

Parse the template to identify all sections and checklist items. The PR description must follow this template exactly.

### Step 8: Compose the PR Description

Fill in every section of the PR template:

- **Description**: Summarize the change based on the conversation context and the diff. Include motivation and context.
- **Ticket/Issue**: Fill in the ticket number if provided, otherwise remove the placeholder line.
- **Checklist**: Evaluate each checklist item honestly:
  - Mark `[x]` only for items that are genuinely satisfied.
  - Mark `[ ]` for items not yet addressed, and add a brief note explaining why.

Do **not** fabricate information. If unsure whether tests cover the change, leave the test checkbox unchecked and note it.

### Step 9: Create the Pull Request

Use the GitHub MCP tool to create the pull request as a **draft**:

- **Repository**: `openvinotoolkit/openvino.genai`
- **Head branch**: `<fork_owner>:<branch_name>` (cross-fork format, e.g. `myuser:fix/kv-cache-leak`)
- **Base branch**: `<base_branch>` (default: `master`)
- **Title**: `<pr_title>`
- **Body**: The composed PR description from Step 8
- **Draft**: `true`

### Step 10: Report

After PR creation, report:

- PR URL
- Branch name
- Files included in the commit
- Any out-of-scope changes that were excluded

## Security

- **NEVER** force-push (`--force`, `--force-with-lease`).
- **NEVER** commit secrets, tokens, API keys, or credentials.
- **NEVER** mark a checklist item as done if it is not verified.
- **NEVER** use `git add .` or `git add -A` without confirming all changes are in scope.
- **NEVER** push to `openvinotoolkit/openvino.genai` directly. Always push to the user's fork.
