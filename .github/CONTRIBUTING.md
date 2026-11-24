# Contributing
1. See [pull_request_template.md](/.github/pull_request_template.md) for pull request (PR) requirements.
2. See [BUILD.md](/src/docs/BUILD.md) for instructions on how to build `OpenVINO™ GenAI`.
3. Code style is determined by the file the change is made in. If ambiguous, look into the neighboring files of the same type. In case of contradiction, pick any of the options but stay consistent in your choice.
4. Don't push branches directly to the upstream repository. Once a branch is pushed to upstream, non-admins lose push access to it, preventing you from updating your changes. Instead, push to your fork and open PRs from there.
5. Your PR will be tested after one of the developers approves the tests run.
6. Branching policy is aligned with [OpenVINO's policy](https://github.com/openvinotoolkit/openvino/blob/71ee9cc42ec63b3affb2801dbbc4a77e6d8003f6/CONTRIBUTING_PR.md#branching-policy).
7. Contributions with use of AI must comply with [OpenVINO's AI Usage Policy](https://github.com/openvinotoolkit/openvino/blob/c4f4325c57977c684184e758449d1f8825ebbfd7/AI_USAGE_POLICY.md).

# Code Quality
This repository uses [pre-commit](https://pre-commit.com/) hooks to catch common issues early (JSON/YAML syntax errors, merge conflict markers, line endings, etc.).

To set up locally:

> **Note for Windows users:** Run `git config core.autocrlf input` first to avoid line-ending conflicts with the hooks.

```bash
pip install pre-commit
pre-commit install
```

After that, checks run automatically on every `git commit`. To run manually on changed files:
```bash
pre-commit run --from-ref origin/master --to-ref HEAD
```

# GGUF CI selection

The platform workflows use Smart CI labels to select test matrices. The standalone
`tests/python_tests/gguf_reference/CMakeLists.txt` builds only the GGUF test generator
and oracle, so it belongs to `category: GGUF`, not the product-wide CMake/build
category. Changes to this reference project retain the existing GGUF Reader jobs
on all platforms and the GGUF Frontend tiny-model jobs on Linux and Manylinux.

Changes to `src/cpp/src/gguf_utils/**` and `tests/cpp/test_gguf_*.cpp` are labeled
`category: GGUF` only. They run the GGUF matrix jobs and the GGUF gtests
(`--gtest_filter="*GGUF*"`) in the Linux and Windows tools jobs. Keep new GGUF
gtest suite names containing `GGUF`. The Linux `ENABLE_XGRAMMAR=OFF` job is skipped
when GGUF is the only affected component.

Production build files, shared code, unmatched files, and other component labels
keep their existing validation rules. The broad Python-test GGUF label is unchanged;
do not narrow it as part of GGUF CI optimizations. No test commands or platform
coverage are removed. Post-commit runs retain the Smart CI action's full scope.

Validate CI selection changes with:

```bash
python -m pytest -v .github/scripts/tests
```

# New feature contribution
In order to get accepted PR with new features, the following list of items MUST be completed. Otherwise, PR will be rejected.
1. Proof of Concept (PoC) pipeline including model preparation step using `optimum-intel` and `GenAI` inference implementation.
2. Pass architectural review with
    1. API proposal for `optimum-intel` and `GenAI`
    2. Working PoC
    3. Command line arguments for model conversion with `optimum-cli export openvino`
    4. `GenAI` sample
