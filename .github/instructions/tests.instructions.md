---
applyTo: "**/tests/**,.github/workflows/**,**/*requirements*.txt,**/pyproject.toml"
---

# Tests and CI rules

1. Flag new tests that never run in CI. GitHub Actions runs tests only on CPU, and on NPU through NPUW with CPU fallback. So tests that need GPU or real NPU hardware never run, and neither do tests that depend on env vars or models that `.github/workflows` never enables. Tests marked `real_models` or `nightly` are excluded from CI on purpose; don't flag them.
2. Repeated runs and greedy decoding on CPU are reproducible. Compare their results with exact equality and an exact `max_new_tokens`. Use a tolerance only for real floats and for sampled outputs compared with a reference.
3. Test each new option to show its effect: the output changes, or for a performance option, the output stays the same and the option is applied.
4. Use `pytest.mark.parametrize` instead of near-duplicate tests.
5. Link a ticket (CVS-### or a GitHub issue) in every skip and xfail.
6. Keep tests cheap: a small `max_new_tokens` and pipelines in module-scoped fixtures. Put a new test suite in its own file instead of growing large files such as `test_vlm_pipeline.py`.
7. Add a new test file to every matching file in `.github/workflows/test_matrices`: `wheel_tests.yml` or `samples_tests.yml` in each OS folder that has one.
8. Make workflow `-m` filters match the pytest markers of the tests they should run.
9. Add a new test or export dependency to `tests/python_tests/requirements.txt` or `samples/export-requirements.txt`, not only to a workflow step.
10. Pin new dependencies in `tests/python_tests/requirements.txt` to an exact version or commit, never `*`.
11. When a dependency release breaks something, exclude that version with `!=` instead of adding a new upper cap. Dependabot updates the existing `<=` caps in `tools/`, `samples/` and the repository root; don't flag them. A new dependency in `tools/` may use the same `>=x,<=y` form.
