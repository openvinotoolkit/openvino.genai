---
description: |
  Shared pre-agent step for the CI Doctor workflows. Pre-downloads failed CI logs
  and pre-locates error hints into /tmp/gh-aw/agent/ci-doctor/ so the agent can
  start from a compact summary instead of re-downloading logs itself.

  The step auto-detects its mode from the environment (no parameters required):
    - run mode  (RUN_ID set):    analyse a single workflow run (CI Doctor — Merge Queue).
    - pr mode   (PR_NUMBER set):  analyse every failed run on a pull request head commit.

  Output layout (identical in both modes):
    - /tmp/gh-aw/agent/ci-doctor/logs/job-<job-id>.log
    - /tmp/gh-aw/agent/ci-doctor/filtered/job-<job-id>-hints.txt
    - /tmp/gh-aw/agent/ci-doctor/summary.txt
  Run mode additionally writes logs/failed-jobs.json; PR mode additionally writes
  logs/failed-runs.json and logs/run-<run-id>-failed-jobs.json.
steps:
  - name: Checkout CI Doctor scripts
    uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1
    with:
      sparse-checkout: .github/scripts/ci-doctor
      sparse-checkout-cone-mode: false
      persist-credentials: false
  - name: Set up Python
    uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405  # v6.2.0
    with:
      python-version: '3.11'
  - name: Install dependencies
    run: python -m pip install --quiet -r .github/scripts/ci-doctor/requirements.txt
  - name: Download CI failure logs
    env:
      GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      REPO: ${{ github.repository }}
      PR_NUMBER: ${{ github.event.issue.number }}
      RUN_ID: ${{ github.event.workflow_run.id || github.event.inputs.run_id }}
    run: python .github/scripts/ci-doctor/ci_doctor_prepare.py
---

<!--
Shared CI Doctor pre-analysis step. This file has no `on:` trigger, so it is a
shared workflow component: it is imported (never compiled standalone) via

    imports:
      - shared/agentic-workflows/download-failure-logs.md

Imported `steps:` are prepended to the importing workflow's own steps at compile
time. See https://github.github.com/gh-aw/reference/imports/#importing-steps

The pre-analysis logic lives in .github/scripts/ci-doctor/ci_doctor_prepare.py,
which is checked out (sparsely) and executed by the steps above.
-->
