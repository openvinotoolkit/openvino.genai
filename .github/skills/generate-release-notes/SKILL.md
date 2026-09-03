---
name: generate-release-notes
description: "Generate OpenVINO GenAI release notes from commits since the latest release. Use when: preparing release notes, summarizing a release branch, or comparing the latest release tag with the current release branch."
---

# Generate Release Notes

Generates release notes for `openvinotoolkit/openvino.genai` from commits added after the latest release tag and writes them to a Markdown file.

## Procedure

### Step 1: List Release Commits

Run the bundled script from the repository root:

```bash
bash .github/skills/generate-release-notes/list_release_commits.sh
```

### Step 2: Check Out the Release Branch

Read the `Current release branch` reported by the script and remove the `upstream/` prefix to get the local release branch name. For example, use `releases/2026/4` for `upstream/releases/2026/4`.

Check out the local release branch:

```bash
git checkout <release-branch>
```

If checkout fails for any reason, stop immediately without creating or modifying the release notes. Ask the user to check out the release branch cleanly, then rerun this skill.

### Step 3: Generate Release Notes

If a commit subject does not provide enough information, inspect the currently checked-out release branch's source code, public APIs, tests, bindings, samples, or documentation to prepare an accurate description.

Group commit into categories:

- Major core features
- LLM Bench, WWB Tools
- JS Bindings
- Docs updates
- Discontinued and deprecated features
- Minor features, bug fixes, and other changes

Create a Markdown file <release_notes_YYYY.N.md> with the following template:

```markdown
# OpenVINO™ GenAI Release Notes YYYY.N

## Features

## LLM Bench, WWB Tools

## JS Bindings

## Other Changes

## Docs Updates

## Discontinued

## Deprecated
```
