---
applyTo: "**/*.cpp,**/*.hpp"
---

# C++ rules

## Correctness

1. Use `OPENVINO_ASSERT(condition, ...)` for checks instead of `if` + `OPENVINO_THROW(...)` or `throw`. Use `OPENVINO_THROW(...)` only when the exception is unconditional.

   ```cpp
   // Bad
   if (num_frames == 0) {
       OPENVINO_THROW("num_frames must be positive");
   }
   // Good
   OPENVINO_ASSERT(num_frames > 0, "num_frames must be positive, got ", num_frames);
   ```

2. In checks on user input, the error message names the wrong value and the expected one. Don't add a comment that repeats the message.
3. An output `ov::Tensor` from `InferRequest::get_tensor()` is a view into the request's buffer, and the next `infer()` on that request overwrites it. Flag such a tensor that is kept (in a member, a cache or chat history) and read after the next `infer()` without a deep copy of its data. Copying the `ov::Tensor` handle still shares the buffer. Reusing an input tensor from `get_tensor()` is fine.

   ```cpp
   // Bad: the next infer() overwrites m_last_logits
   m_last_logits = m_request.get_tensor("logits");
   // Good
   ov::Tensor logits = m_request.get_tensor("logits");
   m_last_logits = ov::Tensor{logits.get_element_type(), logits.get_shape()};
   logits.copy_to(m_last_logits);
   ```

4. `tensor.data<T>()` checks the element type internally, so don't ask for an assert on `get_element_type()` before it.
5. Treat values read from model files (config JSON, tokenizer data, model shapes) as untrusted. Flag a new read of such a value that is used without a check as an allocation size, a divisor or an index.
6. Flag state that a streamer callback or another `std::thread` writes and the generation loop reads without synchronization, for example in `ThreadedStreamerWrapper`.

## Design

7. A public base class has only virtual methods (overrides are marked `override`), no data members and no templates. Keep state in the `Impl`.
8. Put model- or format-specific code in its own directory or class, for example `src/cpp/src/gguf_utils`, not in shared files such as `utils.cpp` or `llm/pipeline.cpp`.

## Performance and style

9. Don't deep-copy tensor data or large containers without need (the `get_tensor()` copy above is the exception). Pass `ov::Tensor` and other non-trivial objects by `const&`. Take an argument that is moved from as `&&`. A by-value `ov::Tensor` shares its data, so don't call it a deep copy.
10. Avoid `dynamic_cast` in hot paths (inference loops). Use `static_cast` or redesign if the type is known.
11. Don't write `using namespace std;`. Don't use `auto` for primitive types where it hides the type. Use `const` and `constexpr` wherever possible.
12. Initialize members in the constructor's member initializer list, not in its body. When initial container values are known, use brace-initialization instead of inserting into an empty container.

    ```cpp
    // Bad
    Cache::Cache(size_t capacity) {
        m_capacity = capacity;
    }
    std::vector<int64_t> ids;
    ids.push_back(bos_token_id);
    ids.push_back(eos_token_id);
    // Good
    Cache::Cache(size_t capacity) : m_capacity{capacity} {}
    std::vector<int64_t> ids{bos_token_id, eos_token_id};
    ```

13. Add a comment to every `const_cast` that says why the write is safe.
14. Don't add unused non-public functions or constructors, except in `debug_utils.hpp`. Don't add a parameter that no caller or override uses.
15. Flag a new non-public, non-`void` function whose return value no caller uses. Public API in `src/cpp/include` and the bindings has callers outside the repository.
16. Never include `debug_utils.hpp`.
17. Report an algorithmic branch choice (log or warning) once, not in every place that sees it.
