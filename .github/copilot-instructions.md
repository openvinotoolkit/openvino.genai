# OpenVINO GenAI Copilot Instructions

OpenVINO GenAI is a C++17 library (`ov::genai`, public headers in `src/cpp/include`) with Python (pybind11, `src/python`), C (`src/c`) and JS (`src/js`) bindings. When reviewing, comment only on issues a maintainer would ask to fix. Skip formatting that linters already check.

This file holds rules for the whole repository. Rules for C++ sources, Python bindings, tests, samples and builds are in `.github/instructions/`.

## General Coding Guidelines

Follow these rules when writing, modifying, or reviewing code in this repository. Rules are grouped by topic.

### Correctness

1. Don't adjust config values silently or with a warning. Throw an exception instead.
2. Don't remove an existing exception or error path, and don't turn it into a warning or a no-op. This doesn't apply to a PR that supports a case the check used to reject and tests that case.
3. Don't add try/catch blocks just to be safe. Catch only an exception that is expected and handled. Catch-all blocks at the C and JS binding boundaries, which turn exceptions into status codes or JS errors, are expected.
4. When you move code into a new function, keep its behavior the same.

### General

5. Before adding a helper, search the repository for an existing one. Extract code that the change repeats.
6. Don't limit the code to a specific device unless the feature needs it.

### Public API and bindings

7. When the public C++ API in `src/cpp/include` changes, make the same change in the Python bindings in `src/python`: overloads, constructors (including `std::filesystem::path`), argument names, defaults, getters and setters. `.pyi` stubs are enforced by tests, so no review comments needed. Check `src/c` and `src/js` if they wrap the changed API.
8. Give new public APIs docstrings in the C++ header and the Python binding, and documentation in `site/docs`.
9. Use model-neutral public names: `Talker`, not `Qwen3OmniTalker`.
10. Use the same base name for related public entities across layers, for example enum/class `VisionPreprocess` and env var `VISION_PREPROCESS`. Flag two public names for the same thing.
11. Keep the public API small. Flag new public getters, setters, flags or env vars that the change doesn't need.
12. Give a new pipeline, config or overload the same shape as its siblings: an optional config argument is `std::optional<...Config>` with the default `std::nullopt`, properties map to Python kwargs, and a `ChatHistory` argument is named `history`.
13. Don't use deprecated APIs in new code, tests or samples. For example, use `generate()` with `ChatHistory` instead of `start_chat()` and `finish_chat()`. Tests of the deprecation itself may call the deprecated API.
14. Don't remove or rename public API in `src/cpp/include` or the bindings in one step. Deprecate it first (`OPENVINO_DEPRECATED` in C++, `PyErr_WarnEx(PyExc_DeprecationWarning, ...)` in the Python bindings) and link a ticket for its removal.
15. Don't flag ABI breaks: ABI stability isn't required. Source compatibility is: flag public API in `src/cpp/include` or the bindings that is removed, renamed or loses a default argument without the deprecation step above.

### Comments and files

16. Write comments about the current code. When a comment refers to other code, name the function. Avoid vague references such as "upstream" or "the two".
17. Avoid ambiguous words like "some" or "capable".
18. Point permalinks to a commit, not a branch.
19. Start new source files with the copyright header and the current year, for example `Copyright (C) 2026 Intel Corporation` and `SPDX-License-Identifier: Apache-2.0`.
20. Don't commit one-off scripts (repro, debug, local conversion) that CI, tests and docs don't use.
21. Make removals complete: no stale docs, empty config sections or leftover initializers. The PR description says why the code was removed.

### Tests and components

22. Add tests for new features and bug fixes: Python tests in `tests/python_tests`, JS tests in `src/js/tests`. C++ tests are optional. Documentation, CI, sample-only and xfail-only changes don't need new tests. Don't ask for tests of test utilities or of tokenizer behavior.
23. For a new model architecture, add a `tiny-random` model to the functional and sample tests. If no such model exists yet, a follow-up with a linked ticket is fine.
24. Add a new component to `.github/labeler.yml`, `.github/components.yml` and `.github/CODEOWNERS` when a new use case is introduced.

## Code Review Instructions for PRs

When performing a code review on a Pull Request, additionally follow this protocol:

25. PR description must be aligned with [./pull_request_template.md](./pull_request_template.md) and its checklist must be filled out. If not, request the author to update the description and checklist before proceeding with the review. Individual items can be marked as N/A in the checklist.
26. If the documentation is updated, PR description MUST include a link to the corresponding documentation deployed on the fork.
27. PR description must be up to date and include all information about the changes.
28. If a PR aims to improve accuracy or performance, the PR description must include corresponding metrics. Performance-oriented PRs require both accuracy and performance metrics.
29. A PR has one purpose. Flag hunks unrelated to it: reformatting, edits to unrelated comments, submodule bumps, extra features.
