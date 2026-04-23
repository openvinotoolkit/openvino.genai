---
name: update-docs
description: "Update OpenVINO GenAI site documentation for API or feature changes. Use when: new pipelines, models, or use-cases are introduced; site docs need to reflect new capabilities."
argument-hint: "Description of what changed (e.g. 'added SpeculativeDecodingPipeline' or 'changed GenerationConfig fields')"
---

# Update Docs

Updates Docusaurus documentation pages under `/docs/` after a code change. Content in this repository is consumed by a centralized Docusaurus hub (multi-instance), so no Docusaurus app lives here.

## When to Use

- New pipeline, model type, or use-case was introduced
- The supported-models table needs a new entry
- A new public API, config option, guide, or concept needs to be documented on the site

## Inputs

The user must provide (or the agent infers from the diff):

- **change_description**: what was added or changed (e.g. `"added VisualLanguageModelPipeline"`, `"changed GenerationConfig.max_new_tokens default"`)

If the description is not provided, derive it from the git diff against `master` (see Step 1).

## Procedure

### Step 1: Identify Changed Files

Use the `get_changed_files` tool to list files changed relative to `master`. Focus on paths under `src/cpp/include/`, `src/python/`, `src/js/lib/`.

To understand what changed in each relevant file, prefer using appropriate tool calls over custom bash commands.

### Step 2: Update Site Documentation

Decide which site sections need updating based on what changed:

| Change type                     | Section to update                     |
| ------------------------------- | ------------------------------------- |
| New pipeline / use-case         | `docs/use-cases/<category>/index.mdx` |
| New model type supported        | `docs/supported-models/_components/`  |
| New public API or config option | Relevant guide in `docs/guides/`      |
| New concept or algorithm        | `docs/concepts/`                      |

**Rules:**

- MDX format; match surrounding file structure.
- Code snippets must have both C++ and Python tabs (see existing `index.mdx` in use-cases for the tab component pattern).
- Code snippets must have JavaScript tab if NodeJS API changed.
- Do not invent model names, benchmark numbers, or unverified capabilities.
- Cross-link to related pages using relative links (e.g., `[Supported Models](../supported-models/)`).

**Rules for `models.ts` entries:**

- **Architecture** (optional): verify the `architecture` value against the model's `config.json` (`"architectures"` field) or HuggingFace model card. Do not guess from the model name.
- **Name**: use the existing `models.ts` as the source of truth for naming style. The `name` is usually the marketing / family name grouping related versions under one entry (e.g. `name: 'Phi3'` covers Phi-3, Phi-3.5, Phi-4). Strip organisation prefixes and per-size suffixes (`-7B`, `-Instruct`), but preserve hyphens and version numbers when the existing table uses them (e.g. `Qwen2.5`, `Phi-3.5-MoE`).
- **Order**: insert new entries in alphabetical order by `architecture`.

### Step 3: Validate

This repository no longer hosts a Docusaurus app; validation happens in the centralized hub build. Before committing, verify locally:

- MDX frontmatter is well-formed and matches surrounding files.
- All relative imports (`./...` or `../...`) resolve to files that exist in this repository.
- All image references use per-section relative paths (e.g. `![alt](./img/foo.png)`); images live under the nearest `docs/<section>/img/` folder.
- No `@site/...` imports (those only work inside a full Docusaurus app).

If a hub checkout is available, running the hub's build against this branch is the authoritative check.

### Step 4: Verify Completeness

Run the following checklist before declaring the documentation update done:

- [ ] Docs cover the new capability (use-case page, guide, or model table entry).
- [ ] No `@site/...` imports introduced; relative imports resolve in this repository.
- [ ] Images live under the nearest `docs/<section>/img/` folder and are referenced with relative paths.

### Step 5: Report

Summarize to the user:

- Files changed and what was added/updated in each.
- Any gaps where documentation could not be written because implementation details are unclear — list those explicitly and ask the user to clarify.
