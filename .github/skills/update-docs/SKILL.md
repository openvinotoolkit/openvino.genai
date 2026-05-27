---
name: update-docs
description: "Update OpenVINO GenAI site documentation for API or feature changes. Use when: new pipelines, models, or use-cases are introduced; site docs need to reflect new capabilities."
argument-hint: "Description of what changed (e.g. 'added SpeculativeDecodingPipeline' or 'changed GenerationConfig fields')"
---

# Update Docs

Updates Docusaurus site pages under `/site/docs/` after a code change.

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

| Change type                     | Section to update                          |
| ------------------------------- | ------------------------------------------ |
| New pipeline / use-case         | `site/docs/use-cases/<category>/index.mdx` |
| New model type supported        | `site/docs/supported-models/_components/`  |
| New public API or config option | Relevant guide in `site/docs/guides/`      |
| New concept or algorithm        | `site/docs/concepts/`                      |

**Rules:**

- MDX format; match surrounding file structure.
- Code snippets must have both C++ and Python tabs (see existing `index.mdx` in use-cases for the tab component pattern).
- Code snippets must have JavaScript tab if NodeJS API changed.
- Do not invent model names, benchmark numbers, or unverified capabilities.
- Cross-link to related pages using relative links (e.g., `[Supported Models](/docs/supported-models/)`).

**Rules for `models.ts` entries:**

- **Architecture** (optional): verify the `architecture` value against the model's `config.json` (`"architectures"` field) or HuggingFace model card. Do not guess from the model name.
- **Name**: use the existing `models.ts` as the source of truth for naming style. The `name` is usually the marketing / family name grouping related versions under one entry (e.g. `name: 'Phi3'` covers Phi-3, Phi-3.5, Phi-4). Strip organisation prefixes and per-size suffixes (`-7B`, `-Instruct`), but preserve hyphens and version numbers when the existing table uses them (e.g. `Qwen2.5`, `Phi-3.5-MoE`).
- **Order**: insert new entries in alphabetical order by `architecture`.

### Step 3: Lint and Build

From the `site/` directory, run lint and then a production build to catch any errors:

```bash
cd site
npm run lint:fix
npm run build
```

Fix any errors reported before proceeding. Do not skip this step — a passing build confirms MDX syntax, broken imports, and broken internal links are all resolved.

### Step 4: Verify Completeness

Run the following checklist before declaring the documentation update done:

- [ ] Site docs cover the new capability (use-case page, guide, or model table entry).
- [ ] `npm run lint:fix` passes with no errors.
- [ ] `npm run build` completes successfully.

### Step 5: Report

Summarize to the user:

- Files changed and what was added/updated in each.
- Any gaps where documentation could not be written because implementation details are unclear — list those explicitly and ask the user to clarify.
