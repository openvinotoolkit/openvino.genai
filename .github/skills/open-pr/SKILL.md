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

### Step 1: Analyze Changes

Review the current working tree to understand what was modified:

```bash
git status
git diff --stat
```

Examine each changed file. Determine which changes are **in scope** (directly related to the task) and which are **out of scope** (unrelated formatting, debug leftovers, scratch files, unrelated refactors).

If out-of-scope changes are detected, inform the user and ask whether to:

- Stash or revert the out-of-scope changes before committing
- Include them anyway

**Do not silently include unrelated changes.**

### Step 2: Create a Branch

Create and switch to a new branch from the current HEAD:

```bash
git checkout -b <branch_name>
```

If the branch already exists, ask the user whether to reuse it or pick a different name.

### Step 3: Stage and Commit

Stage only the in-scope files identified in Step 1:

```bash
git add <file1> <file2> ...
```

Do **not** use `git add .` or `git add -A` unless every change is confirmed in-scope.

Write a clear, conventional commit message. Use a single-line summary (≤72 chars):

```bash
git commit -m "<summary>"
```

### Step 4: Determine the Fork Remote

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

### Step 5: Push the Branch

```bash
git push <fork_remote> <branch_name>
```

If the push is rejected (e.g. branch exists on remote), ask the user how to proceed.

### Step 6: Read the PR Template

Read the repository pull request template to get the latest structure:

```
.github/pull_request_template.md
```

Parse the template to identify all sections and checklist items. The PR description must follow this template exactly.

### Step 7: Compose the PR Description

Fill in every section of the PR template:

- **Description**: Summarize the change based on the conversation context and the diff. Include motivation and context.
- **Ticket/Issue**: Fill in the ticket number if provided, otherwise remove the placeholder line.
- **Checklist**: Evaluate each checklist item honestly:
  - Mark `[x]` only for items that are genuinely satisfied.
  - Mark `[ ]` for items not yet addressed, and add a brief note explaining why.

Do **not** fabricate information. If unsure whether tests cover the change, leave the test checkbox unchecked and note it.

### Step 8: Create the Pull Request

Use the GitHub MCP tool to create the pull request as a **draft**:

- **Repository**: `openvinotoolkit/openvino.genai`
- **Head branch**: `<fork_owner>:<branch_name>` (cross-fork format, e.g. `myuser:fix/kv-cache-leak`)
- **Base branch**: `<base_branch>` (default: `master`)
- **Title**: `<pr_title>`
- **Body**: The composed PR description from Step 7
- **Draft**: `true`

### Step 9: Report

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
