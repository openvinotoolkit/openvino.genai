# OpenVINO GenAI Copilot Instructions

OpenVINO GenAI is a C++17 library (`ov::genai`, public headers in `src/cpp/include`) with Python (pybind11, `src/python`), C (`src/c`) and JS (`src/js`) bindings. When reviewing, comment only on issues a maintainer would ask to fix. Skip formatting that linters already check.

## General Coding Guidelines

Follow these rules when writing, modifying, or reviewing code in this repository. Rules are grouped by topic.

### Correctness

1. Don't adjust config values silently or with a warning. Throw an exception instead. A default that the library picks when the user set nothing may log `GENAI_WARN`.
2. Use `OPENVINO_ASSERT(condition, ...)` for checks instead of `if` + `OPENVINO_THROW(...)` or `throw`. Use `OPENVINO_THROW(...)` only when the exception is unconditional.
3. In checks on user input, the error message names the wrong value and the expected one. Don't add a comment that repeats the message.
4. Don't remove an existing exception or error path, and don't turn it into a warning or a no-op. The only exception is a PR that supports a case the check used to reject and tests that case.
5. Don't add try/catch blocks just to be safe. Catch only an exception that is expected and handled.
6. An `ov::Tensor` from `InferRequest::get_tensor()` is a view into the request's buffer, and the next `infer()` on that request overwrites it. Flag such a tensor that is kept (in a member, a cache or chat history) and read after the next `infer()` without a copy.
7. `tensor.data<T>()` checks the element type internally, so don't ask for an assert on `get_element_type()` before it. It doesn't check bounds: check indices that come from user input or model output (for example, token ids) against `get_size()` or the shape.
8. When you move code into a new function, keep its behavior the same.

### Public API and bindings

9. When the public C++ API in `src/cpp/include` changes, make the same change in the Python bindings in `src/python`: overloads, constructors (including `std::filesystem::path`), argument names, defaults, getters and setters. Regenerate the `.pyi` stubs in `src/python/openvino_genai`. Check `src/c` and `src/js` if they wrap the changed API.
10. Give new public APIs docstrings in the C++ header and the Python binding, and documentation in `site/docs`.
11. Use model-neutral public names: `Talker`, not `Qwen3OmniTalker`.
12. Use the same base name for related public entities across layers, for example enum/class `VisionPreprocess` and env var `VISION_PREPROCESS`. Flag two public names for the same thing.
13. Keep the public API small. Flag new public getters, setters, flags or env vars that the change doesn't need.
14. Don't use deprecated APIs in new code, tests or samples. For example, use `generate()` with `ChatHistory` instead of `start_chat()` and `finish_chat()`.
15. Don't flag ABI breaks: ABI stability isn't required.

### C++

16. Don't deep-copy tensor data or large containers without need (rule 6 is the exception). `ov::Tensor` is a reference-counted handle, so passing it by value is cheap. Pass other non-trivial objects by `const&`. Take an argument that is moved from as `&&`.
17. Avoid `dynamic_cast` in hot paths (inference loops). Use `static_cast` or redesign if the type is known.
18. Don't write `using namespace std;`. Don't use `auto` for primitive types where it hides the type. Use `const` and `constexpr` wherever possible.
19. Initialize members in the constructor's member initializer list, not in its body. When initial container values are known, use brace-initialization instead of inserting into an empty container.
20. Add a comment to every `const_cast` that says why the write is safe.
21. Don't add unused functions or constructors, except in `debug_utils.hpp`. Don't add a parameter that no caller or override uses.
22. Flag a new non-`void` function whose return value no caller uses.
23. Never include `debug_utils.hpp`.
24. Before adding a helper, search the repository for an existing one. Extract code that the change repeats.
25. Don't limit the code to a specific device unless the feature needs it.
26. Report an algorithmic branch choice (log or warning) once, not in every place that sees it.

### Comments and files

27. Write comments about the current code. When a comment refers to other code, name the function. Avoid vague references such as "upstream" or "the two".
28. Point permalinks to a commit, not a branch.
29. Start new source files with the copyright header and the current year, for example `Copyright (C) 2026 Intel Corporation` and `SPDX-License-Identifier: Apache-2.0`.
30. Don't commit one-off scripts (repro, debug, local conversion) that CI, tests and docs don't use.
31. Make removals complete: no stale docs, empty config sections or leftover initializers. The PR description says why the code was removed.

### Tests and CI

32. Add tests for new features and bug fixes: Python tests in `tests/python_tests`, JS tests in `src/js/tests`. C++ tests are optional. Documentation-only and CI-only changes don't need tests.
33. Flag new tests that never run in CI. GitHub Actions runs tests only on CPU, and on NPU through NPUW with CPU fallback. So tests that need GPU or real NPU hardware never run, and neither do tests that depend on env vars, markers or models that `.github/workflows` never enables.
34. For a new model architecture, add a `tiny-random` model to the functional and sample tests.
35. CPU results are reproducible. Compare them with exact equality and an exact `max_new_tokens`. Use `pytest.approx` only for real floats.
36. Test each new option to show its effect: the output changes, or for a performance option, the output stays the same and the option is applied.
37. Use `pytest.mark.parametrize` instead of near-duplicate tests.
38. Link a ticket (CVS-### or a GitHub issue) in every skip and xfail.
39. Keep tests cheap: a small `max_new_tokens` and pipelines in module-scoped fixtures. Put a new test suite in its own file instead of growing large files such as `test_vlm_pipeline.py`.
40. Add a new test file to every matching file in `.github/workflows/test_matrices`: `wheel_tests.yml` or `samples_tests.yml` in each OS folder that has one.
41. Make workflow `-m` filters match the pytest markers of the tests they should run.
42. Pin new dependencies in `tests/python_tests/requirements.txt` to an exact version or commit, never `*`.
43. When a dependency release breaks something, exclude that version with `!=` instead of adding a new upper cap. Dependabot updates the existing `<=` caps in `tools/`; don't flag them.
44. Add a new component to `.github/labeler.yml`, `.github/components.yml` and `.github/CODEOWNERS`.

### Samples and docs

45. Avoid adding new samples unless there is a strong, clearly justified reason. Add tests for new samples.
46. Keep command-line arguments minimal: required inputs are positional, and other values are hardcoded with a comment. An optional positional `device` is allowed.
47. Update the C++ and Python versions of a sample, its README and the docs examples together.

### Build

48. Use all available cores for building code from source.

## Code Review Instructions for PRs

When performing a code review on a Pull Request, additionally follow this protocol:

49. A PR has one purpose. Flag hunks unrelated to it: reformatting, edits to unrelated comments, submodule bumps, extra features.
50. The description says what changed and why, and matches the diff. Its checklist covers these items:
    - The PR follows the contributing guidelines.
    - Tests were added or updated, and the description names them.
    - The PR fully addresses the ticket, or says what is left.
    - The documentation was updated.

    The last three items may be marked N/A. If the description is missing or doesn't match the diff, say so once in the review summary and continue the code review.
51. If documentation under `site/` changes, the description links to the docs deployed from the author's fork with `deploy_gh_pages.yml`.
52. A PR that improves accuracy includes accuracy metrics. A PR that improves performance includes both accuracy and performance metrics.
