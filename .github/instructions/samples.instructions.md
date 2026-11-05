---
applyTo: "samples/**"
---

# Samples rules

1. Avoid adding new samples unless there is a strong, clearly justified reason.
2. Add tests for new samples.
3. Keep command-line arguments minimal: required inputs are positional, and other values are hardcoded with a comment. An optional `device` argument and the optional flags that sibling samples already use are allowed.
4. Update the C++ and Python versions of a sample, its README and the docs examples together.
5. Samples are minimal demos. Don't ask them to handle edge cases such as malformed input, integer overflow in argument parsing or platform-specific file APIs.
