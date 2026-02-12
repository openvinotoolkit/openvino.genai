# Blueprint: Implementation of Text2Video Node.js Bindings

## 🎯 Objective
Enable video generation in Node.js by wrapping the C++ `Text2VideoPipeline` class using N-API.

## 🏗️ Current State (Pre-Scaffolded)
1. **Registered**: `Text2VideoPipeline` is already added to `AddonData` and `addon.cpp`.
2. **Infrastructure**: Initial placeholders created in `src/js/include/text2video_pipeline/` and `src/js/src/text2video_pipeline/`.
3. **TS Interface**: Initial draft created in `src/js/lib/pipelines/text2VideoPipeline.ts`.

---

## 🛠️ Implementation Steps for Claude Code

### 1. C++ Wrapper Logic (`src/js/src/text2video_pipeline/`)
The wrapper must handle two main asynchronous operations:
- **`init`**: Compiles the model onto a device.
- **`generate`**: Runs the inference loop.

#### Required Pattern:
Use the `LLMPipeline` worker pattern. Since video generation is time-consuming, the `generate` method should support a progress callback if the C++ API allows it.

**Key Mapping**:
- Input: `std::string prompt`, `ov::AnyMap config`.
- Output: `ov::Tensor` (which is converted to a JS `Tensor` object by our helper).

### 2. Header Inclusion
**Crucial**: The developer must locate the final `text2video_pipeline.hpp`. Based on `text2image_pipeline.hpp`, the likely namespace is `ov::genai::Text2VideoPipeline`.
*Update `text2video_pipeline_wrapper.hpp` with the correct `#include` once the C++ header is merged/identified.*

### 3. TypeScript Layer (`src/js/lib/pipelines/`)
Refine `text2VideoPipeline.ts` to ensure it exports a clean `generate()` method that returns a `Promise`.

**Proposed API**:
```typescript
const pipeline = await Text2VideoPipeline(modelPath, 'CPU');
const result = await pipeline.generate("A cat playing with a ball", {
  width: 512,
  height: 512,
  num_frames: 25 // Example video parameter
});
```

### 4. Build Configuration
Ensure `src/js/CMakeLists.txt` includes the new source files.
*(Note: Current GLOB_RECURSE in `src/js/CMakeLists.txt` should pick them up automatically, but verify during build).*

---

## 🧪 Verification Strategy
1. **Mock Test**: Create `tests/text2video.test.js`. If a real video model is too heavy for CI, use the `setOpenvinoAddon` helper to test the binding structure.
2. **Sample**: Create `samples/js/text2video.js` demonstrating the end-to-end flow.

## 🚩 Risks & Doubts
1. **C++ API Stability**: The exact names of parameters in `ov::genai::ImageGenerationConfig` for video (like `num_frames` or `fps`) need to be verified against the C++ implementation.
2. **Memory Management**: Videos are large tensors; ensure buffer sharing between C++ and JS is efficient to avoid OOM.
