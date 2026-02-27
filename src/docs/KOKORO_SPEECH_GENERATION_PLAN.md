# Kokoro Speech Generation Integration Plan (OpenVINO GenAI)

Date: 2026-02-27

This document proposes a concrete, incremental implementation plan to add Kokoro support to OpenVINO GenAI `speech_generation` APIs while preserving existing SpeechT5 behavior.

## 1) Current Baseline (as-is)

### Public API
- `src/cpp/include/openvino/genai/speech_generation/text2speech_pipeline.hpp`
- `src/cpp/include/openvino/genai/speech_generation/speech_generation_config.hpp`

### Current backend wiring
- `src/cpp/src/speech_generation/text2speech_pipeline.cpp`
  - Reads `config.json` and checks `architectures[0]`.
  - Instantiates only `SpeechT5TTSImpl` when architecture is `SpeechT5ForTextToSpeech`.

### Current backend implementation
- `src/cpp/src/speech_generation/speecht5_tts_model.hpp`
- `src/cpp/src/speech_generation/speecht5_tts_model.cpp`
- `src/cpp/src/speech_generation/speecht5_tts_decoder.*`

### Config constraints today
- `SpeechGenerationConfig` is SpeechT5-oriented (`minlenratio`, `maxlenratio`, `threshold`).
- `Text2SpeechPipeline::generate(...)` currently ignores runtime `properties` argument and passes only stored config.

## 2) Target Architecture (SpeechT5 + Kokoro)

## 2.1 Backend routing strategy

Introduce an explicit backend discriminator for speech generation:
- `speecht5_tts`
- `kokoro`

Resolution order:
1. Optional explicit property (recommended) in `ov::AnyMap` (e.g., `speech_model_type`).
2. Fallback to `config.json` auto-detection via `architectures[0]`.
3. Final fallback to deterministic model-file signatures.

Rationale:
- Keeps compatibility with existing SpeechT5 export layouts.
- Avoids brittle architecture-name assumptions for Kokoro-converted IR folders.

## 2.2 Keep top-level API stable

Do not fork the pipeline class. Keep:
- `Text2SpeechPipeline`
- `Text2SpeechPipeline::generate(...)`

Extend config to support backend-specific options cleanly.

## 3) Proposed File-level Changes

## 3.1 Modify existing files

1. `src/cpp/include/openvino/genai/speech_generation/speech_generation_config.hpp`
2. `src/cpp/src/speech_generation/speech_generation_config.cpp`
   - Add generic/shared fields and backend-specific Kokoro fields (see section 4).
   - Keep SpeechT5 fields for backward compatibility.

3. `src/cpp/src/speech_generation/text2speech_pipeline.cpp`
   - Add backend resolution helper.
   - Route to `SpeechT5TTSImpl` or new `KokoroTTSImpl`.
   - Fix runtime-property handling in `generate(...)` by merging temporary config from `properties`.

4. `src/python/py_speech_generation.cpp`
   - Expose new config fields for Kokoro.
   - Keep old SpeechT5 fields unchanged.

5. `src/cpp/CMakeLists.txt`
   - Link `misaki-cpp` target conditionally for Kokoro support.
   - Add Kokoro source files.

## 3.2 Add new C++ backend files

Create under `src/cpp/src/speech_generation/`:

1. `speech_generation_backend_type.hpp`
   - enum class `SpeechGenerationBackendType { SpeechT5, Kokoro }`

2. `speech_generation_backend_resolver.hpp/.cpp`
   - backend auto-detection and override resolution.

3. `kokoro_tts_model.hpp/.cpp`
   - `KokoroTTSImpl : public Text2SpeechPipelineImpl`
   - OpenVINO model load/compile and inference orchestration.

4. `kokoro_preprocess.hpp/.cpp`
   - Text splitting/chunking logic (English-first scope).
   - Converts Misaki tokens to Kokoro phoneme sequences with max length constraints.

5. `kokoro_g2p.hpp/.cpp`
   - Thin wrapper over `misaki::G2P` API for stable integration boundary.

6. `kokoro_voice_pack.hpp/.cpp` (optional in phase 1)
   - Voice embedding/pack load and validation.
   - Keep optional if phase 1 passes voice embedding tensor directly.

## 4) Config Proposal

Add these fields to `SpeechGenerationConfig`.

### 4.1 Shared fields (both backends)
- `std::string model_type` (empty/default = auto)
- `float speed = 1.0f`
- `uint32_t sample_rate = 24000` (Kokoro default; SpeechT5 can ignore/override)

### 4.2 SpeechT5 fields (existing)
- `float minlenratio`
- `float maxlenratio`
- `float threshold`

### 4.3 Kokoro fields (new)
- `std::string language = "en-us"` (or `"en-gb"`)
- `std::string voice = ""` (required for Kokoro synthesis path)
- `size_t max_phoneme_length = 510`
- `std::string split_pattern = "\\n+"`
- `bool return_token_timestamps = false` (future-facing)

Validation rules:
- `speed > 0`
- `sample_rate > 0`
- `model_type` must be empty/known
- if backend resolves to Kokoro, require non-empty `voice`
- `max_phoneme_length` in sensible range (e.g. `[32, 4096]`, initially default 510)

## 5) Misaki C++ Integration Contract

`misaki-cpp` public API used:
- `misaki::make_engine("en", "en-us"|"en-gb")`
- `engine->phonemize_with_tokens(text)`

Phase-1 English scope:
- `language=en-us|en-gb`
- use token stream from Misaki result
- chunk by Kokoro phoneme length limit (~510)

Kokoro preprocessing parity targets (from Python `kokoro/pipeline.py`):
- preserve whitespace-sensitive phoneme concatenation
- chunking with punctuation-aware split strategy
- cap and truncate over-limit chunks safely

## 6) Vertical Slice Milestones

## Milestone A: compile-time plumbing
- Add backend resolver and Kokoro class stubs.
- Build with `misaki-cpp` linked.
- No behavioral change for SpeechT5 path.

## Milestone B: Kokoro English preprocess + single inference
- Implement `kokoro_g2p` and `kokoro_preprocess`.
- Implement one-shot Kokoro inference for single text input.

## Milestone C: full `Text2SpeechPipeline` parity path
- Support vector-of-text input.
- Merge per-call properties into temporary config.
- Return multiple audio tensors in order.

## Milestone D: tests
- Unit tests for chunking and tokenizer/phoneme normalization.
- Integration test for deterministic smoke synthesis.
- Regression tests ensuring SpeechT5 remains unaffected.

## 7) Immediate First Patch Set (recommended)

1. Implement backend resolver (`SpeechT5` default).
2. Add `model_type`, `speed`, `language`, `voice`, `max_phoneme_length` fields to config.
3. Fix `Text2SpeechPipeline::generate(..., properties)` to apply runtime overrides.
4. Add Kokoro backend class skeleton that throws `Not implemented` at runtime.
5. Add pybind exposure for new config fields.

This patch set is low-risk and unblocks rapid incremental Kokoro implementation.

## 8) Open Questions To Freeze Before Coding Milestone B

1. Voice-pack source and format in C++ path:
   - provided as tensor by caller, or loaded from files/HF cache by pipeline?

2. Model file naming convention for Kokoro OpenVINO export in GenAI package:
   - canonical expected names for validation at pipeline init.

3. Required output sample rate behavior:
   - fixed 24 kHz, or allow optional post-resample in API.

4. Scope of phase-1 language support:
   - English only (`en-us`, `en-gb`) using `misaki-cpp`.
