# Optional per-schema whitespace bound (2026.4 draft)

This companion to [OVMS #4563](https://github.com/openvinotoolkit/model_server/pull/4563) transfers the bounded-whitespace schema API from the frozen downstream 2026.4 runtime.

JSONSchema(schema) retains its existing unbounded serialization. JSONSchema(schema, 2) emits max_whitespace_cnt=2 at the json_schema format level. A bound of zero remains explicit. Equality includes the optional policy so schemas with different whitespace constraints are not treated as equal. This changes the C++ struct layout and requires a coordinated rebuild of consumers; it is not an ABI-compatible DLL replacement.

The XGrammar revision is pinned to 9aa840b6d16abf094f3e8e2ac9c10465b77656c9, matching the downstream runtime used for the tool-calling repair. No cache diagnostics/cache-off experiments are included. The OpenVINO/GenAI release line remains 2026.4. Python bindings are not extended in this C++-only transfer.

## Validation

A standalone executable compiled against the real generation_config.hpp with MSVC checked legacy serialization, bounded serialization and formatting, zero-bound preservation, and equality. The unchanged releases/2026/4 header failed compilation because the field/two-argument constructor is absent (RED). The transferred header compiled and all assertions passed (GREEN). This test loads no model or GenAI DLL.

The equivalent four GTest cases are added under tests/cpp and discovered by its existing CMake glob. Those GTests, the full GenAI build, Python bindings, ABI/compatibility review and Linux CI on this companion head are NOT RUN. The downstream RC was previously built with this dependency patch and exercised on Windows GPU; that evidence does not validate this new PR head.

Before merge, review the dependency update, accepted whitespace-bound values and nested structural-tag matcher behavior, run the native test suite and paired OVMS build, and verify both release consumers together.
