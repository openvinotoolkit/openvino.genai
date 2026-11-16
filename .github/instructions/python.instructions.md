---
applyTo: "src/python/**"
---

# Python bindings rules

1. Release the GIL with `py::gil_scoped_release` around C++ calls that can block, such as generation, model compilation and inference. Keep it released only for the C++ call: converting the result to Python needs the GIL.
2. Acquire the GIL with `py::gil_scoped_acquire` before C++ code calls into Python. For streamers, use `pyutils::pystreamer_to_streamer()`, which already does this.

   ```cpp
   StreamerVariant streamer = pyutils::pystreamer_to_streamer(py_streamer);
   ov::genai::EncodedResults encoded_results;
   {
       py::gil_scoped_release release;
       encoded_results = pipe.generate(input_ids, config, streamer);
   }
   return py::cast(encoded_results);
   ```
