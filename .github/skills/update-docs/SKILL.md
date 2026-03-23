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

Use the `get_changed_files` tool to list files changed relative to `master`. Focus on paths under `src/cpp/include/`, `src/python/`, `site/docs/`, and `samples/`.

To understand what changed in each relevant files prefer using appropriate tools calls over custom bash commands.

### Step 2: Update Site Documentation

Decide which site sections need updating based on what changed:

| Change type                     | Section to update                          |
| ------------------------------- | ------------------------------------------ |
| New pipeline / use-case         | `site/docs/use-cases/<category>/index.mdx` |
| New model type supported        | `site/docs/supported-models/_components/`  |
| New public API or config option | Relevant guide in `site/docs/guides/`      |
| New concept or algorithm        | `site/docs/concepts/`                      |
| New sample                      | `site/docs/samples/`                       |

**Rules:**

- MDX format; match surrounding file structure.
- Code snippets must have both C++ and Python tabs (see existing `index.mdx` in use-cases for the tab component pattern).
- Do not invent model names, benchmark numbers, or unverified capabilities.
- Cross-link to related pages using relative links (e.g., `[Supported Models](/docs/supported-models/)`).

**Model name convention for `models.ts` entries:**

If the model ID contains a version, that version must be reflected in the `name` field (e.g. `Qwen2`, `Phi3.5`, `HY-MT1.5`). Strip the organisation prefix and per-size suffixes (e.g. `-7B`, `-Instruct`). When a new version differs from an existing entry, add a new entry instead of appending to the existing one.

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
