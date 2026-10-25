# OpenVINO GenAI Copilot Instructions

OpenVINO GenAI is a C++17 library (`ov::genai`, public headers in `src/cpp/include`) with Python (pybind11, `src/python`), C (`src/c`) and JS (`src/js`) bindings. When reviewing, comment only on issues a maintainer would ask to fix. Skip formatting that linters already check.

## General Coding Guidelines

Follow these rules when writing, modifying, or reviewing code in this repository. They are ordered by importance.

### Correctness

1. Don't adjust config values silently or with a warning. Throw an exception instead. A default that the library picks when the user set nothing may log `GENAI_WARN`.
2. Use `OPENVINO_ASSERT(condition, ...)` for checks instead of `if` + `OPENVINO_THROW(...)` or `throw`. Use `OPENVINO_THROW(...)` only when the exception is unconditional. The message must name the wrong value and what was expected. Don't add a comment next to it.
3. Don't remove an existing exception or error path, and don't turn it into a warning or a no-op.
4. Don't add try/catch blocks just to be safe. Catch only an exception that is expected and handled.
5. An `ov::Tensor` from `InferRequest::get_tensor()` is a view into the request's buffer, and the next `infer()` overwrites it. Flag such a tensor stored past the request scope (in a member, a cache, or chat history) without a copy.
6. `tensor.data<T>()` checks the element type internally, so don't ask for an assert on `get_element_type()` before it. It doesn't check bounds: indices must be checked against `get_size()` or the shape.
7. When code is moved into a new function, its behavior must not change.

### API and bindings

8. A C++ API change needs the same change in the Python bindings in `src/python`: overloads, constructors (including `std::filesystem::path`), argument names, defaults, getters and setters. The `.pyi` stubs in `src/python/openvino_genai` must be regenerated. Check `src/c` and `src/js` if they wrap the changed API.
9. New public APIs have docstrings in the C++ header and the Python binding, and documentation in `site/docs`.
10. Public names are model-neutral: `Talker`, not `Qwen3OmniTalker`.
11. Related entities use the same base name across layers, e.g. enum/class `VisionPreprocess` and env var `VISION_PREPROCESS`. Flag two names for the same thing.
12. Keep the public surface small. Flag getters, setters, flags or env vars that the change doesn't need.
13. New code, tests and samples don't use deprecated APIs such as `start_chat()`.
14. ABI stability isn't required.

### C++

15. Don't copy tensors or other large objects. Pass them by `const&`. An argument that is moved from is `&&`.
16. Avoid `dynamic_cast` in hot paths (inference loops). Use `static_cast` or redesign if the type is known.
17. No `using namespace std;`. No `auto` for primitive types where it hides the type. Use `const` and `constexpr` wherever possible.
18. Initialize members in the constructor's member initializer list, not in its body. When initial container values are known, use brace-initialization instead of inserting into an empty container.
19. Every `const_cast` has a comment that says why the write is safe.
20. No unused functions, constructors or parameters, except in `debug_utils.hpp`. The result of every new non-`void` function is used. `debug_utils.hpp` must never be included.
21. Before accepting a new helper, search the repository for an existing one. Extract code that the change repeats.
22. Don't limit the code to a specific device unless the feature needs it.
23. Report an algorithmic branch choice (log or warning) once, not in every place that sees it.
24. Comments describe the current code and name concrete functions. Avoid vague references such as "upstream" or "the two". Permalinks point to a commit, not a branch.

### Tests and CI

25. New features and fixes need Python tests in `tests/python_tests`. C++ tests are optional.
26. GitHub Actions runs tests only on CPU, and on NPU through NPUW with CPU fallback. Flag new tests that need GPU or other hardware: they never run in CI.
27. A new model architecture needs a `tiny-random` model in the functional and sample tests.
28. CPU results are reproducible. Compare them with exact equality and an exact `max_new_tokens`. Use `pytest.approx` only for real floats.
29. Each new option has a test that shows it changes the output.
30. Use `pytest.mark.parametrize` instead of near-duplicate tests.
31. Every skip and xfail links a ticket and states the error.
32. Flag tests that depend on env vars, markers or models that the workflows in `.github/workflows` never enable.
33. New tests are wired into every OS in `.github/workflows/test_matrices` (linux, mac, manylinux, windows), and workflow `-m` filters match the pytest markers. Pin dependency versions exactly, without `*`. Exclude a broken version (`!=`) instead of adding an upper cap.
34. A new component gets entries in `.github/labeler.yml`, `.github/components.yml` and `.github/CODEOWNERS`.
35. Helper scripts shouldn't be committed.

### Samples

36. Avoid adding new samples unless there is a strong, clearly justified reason. New samples have tests.
37. Keep command-line arguments minimal: required inputs are positional, other values are hardcoded with a comment. The C++ and Python versions of a sample match.

### Build

38. Use all available cores for building code from source.

## Code Review Instructions for PRs

When performing a code review on a Pull Request, additionally follow this protocol:

1. A PR has one purpose. Flag hunks unrelated to it: reformatting, edits to unrelated comments, submodule bumps, extra features.
2. The description says what changed and why, and matches the diff. Its checklist covers these items, and each one may be marked N/A:
   - The PR follows the contributing guidelines.
   - Tests were added or updated, and the description names them.
   - The PR fully addresses the ticket, or says what is left.
   - The documentation was updated.

   If the description is missing or doesn't match the diff, say so once in the review summary and continue the code review.
3. If documentation under `site/` changes, the description links to the docs deployed from the author's fork with `deploy_gh_pages.yml`.
4. A PR that improves accuracy includes accuracy metrics. A PR that improves performance includes both accuracy and performance metrics.
