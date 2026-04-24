# GenAI VLM Pipeline Architecture Reference

Quick reference for the GenAI Visual Language Model pipeline: key interfaces, data flow, and extension points.

## Pipeline Data Flow

The pipeline has two execution paths selected at construction time (see **Attention Backend** below).
The front-end (vision encoding, prompt normalization, embedding merge) is shared; only the LLM inference backend differs.

```
User Input (prompt + images/videos)
       │
       ▼
 VLMPipeline::generate()
       │
       ▼
 VLMChatContext::process()
  ├── Register images/videos in VisionRegistry
  ├── Normalize prompt (model-specific placeholder tokens)
  └── Determine vision sequences (which images go where)
       │
       ▼
 InputsEmbedder::get_inputs_embeds()
  ├── VisionEncoder::encode()          ← model-specific image encoding
  │     └── Image preprocessing (resize, normalize, tile)
  │     └── Vision model inference → EncodedImage
  ├── Tokenize prompt text
  ├── EmbeddingsModel::infer()         ← text token → embedding lookup
  └── Merge image + text embeddings    ← model-specific merge logic
       │
       ├─────────────────────────────────────────────┐
       ▼ SDPA path (VLMPipelineImpl)                 ▼ PA/CB path (VLMContinuousBatchingAdapter)
  LLM with SDPA attention                     ContinuousBatchingPipeline
  ├── Manual KV-cache (CacheState)             ├── Paged KV-cache (scheduler)
  ├── Token-by-token sampler loop              ├── Batch-oriented generation
  └── slice_before_matmul transform            └── SDPAToPagedAttention transform
       │                                             │
       └──────────────┬──────────────────────────────┘
                      ▼
        Sampling → Detokenization → VLMDecodedResults
```

## Source Layout

All VLM sources are under `src/cpp/src/visual_language/`:

| File | Purpose |
|------|---------|
| `pipeline.cpp`, `pipeline_base.hpp` | Public `VLMPipeline` and internal `VLMPipelineBase` / `VLMPipelineImpl` |
| `inputs_embedder.hpp/.cpp` | `InputsEmbedder` (public) and `IInputsEmbedder` (model-specific interface) |
| `vision_encoder.hpp/.cpp` | `VisionEncoder` base class and factory |
| `vision_registry.hpp` | `VisionRegistry` — caches encoded images/videos by ID |
| `vlm_config.hpp` | `VLMConfig` — loaded from `config.json`, includes `VLMModelType` enum |
| `processor_config.hpp` | `ProcessorConfig` — image preprocessing params from `preprocessor_config.json` |
| `video_processor_config.hpp` | `VideoProcessorConfig` — video preprocessing params |
| `embedding_model.hpp` | `EmbeddingsModel` — text token → embedding lookup |
| `vlm_chat_context.hpp` | `VLMChatContext` — chat history and vision data management |
| `continuous_batching_adapter.hpp` | `VLMContinuousBatchingAdapter` — PA/CB backend wrapper |
| `vl_sdpa_transformations.cpp` | SDPA-specific model transformations |

Model-specific implementations live in subdirectories, each containing `classes.hpp` and `classes.cpp`.
To discover the current list, run: `ls src/cpp/src/visual_language/*/classes.hpp`

## Core Interfaces

### VLMModelType (vlm_config.hpp)

Enum that identifies the model type. Parsed from `config.json` `"model_type"` field.
To see the current list of supported types, read `vlm_config.hpp` and look for the `VLMModelType` enum.

### VisionEncoder (vision_encoder.hpp)

Abstract base for image/video encoding. One subclass per model type.

```cpp
class VisionEncoder {
public:
    using Ptr = std::shared_ptr<VisionEncoder>;

    // Factory — dispatches by VLMModelType
    static Ptr create(const std::filesystem::path& model_dir,
                      VLMModelType model_type,
                      const std::string& device,
                      const ov::AnyMap properties = {});

    // Encode a single image → EncodedImage
    virtual EncodedImage encode(const ov::Tensor& image,
                                const ov::AnyMap& config_map = {}) = 0;

    // Encode video frames (optional, throws by default)
    virtual EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames,
                                       const ov::AnyMap& config_map = {});

    ProcessorConfig get_processor_config() const;

protected:
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_encoder;
    ProcessorConfig m_processor_config;
};
```

### EncodedImage (vision_encoder.hpp)

Output of `VisionEncoder::encode()`. Not all fields are used by every model.

```cpp
struct EncodedImage {
    ov::Tensor resized_source;               // [N, num_patches, hidden_size]
    ImageSize resized_source_size;            // H/patch, W/patch
    ov::Shape slices_shape;                   // MiniCPM only
    std::pair<int, int> patches_grid;         // LLaVA-Next tiling grid
    ImageSize original_image_size;            // Pre-resize dimensions
    ov::Tensor images_features_projection;    // Phi3, Phi4, VideoChatFlash
    ResampledImage resampled_image;           // MiniCPM only
    size_t num_image_tokens = 0;             // Token count for prompt expansion
};
```

### IInputsEmbedder (inputs_embedder.hpp)

Model-specific interface for prompt normalization and embedding merge. One subclass per model type.

```cpp
class InputsEmbedder::IInputsEmbedder {
public:
    // Merge text + vision embeddings into a single tensor for the LLM
    virtual ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<EncodedImage>& images,
        VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {}) = 0;

    // Encode raw images via the VisionEncoder
    virtual std::vector<EncodedImage> encode_images(
        const std::vector<ov::Tensor>& images) = 0;

    // Normalize prompt: insert model-specific image placeholder tokens
    virtual NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images) const = 0;

    // Chat lifecycle
    virtual void start_chat(const std::string& system_message);
    virtual void finish_chat();
    virtual void update_chat_history(const std::string& decoded_results,
                                     GenerationStatus status);

    // Position IDs (custom for models like Qwen2-VL with 3D positions)
    std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(
        size_t inputs_embeds_size, size_t history_size);
};
```

### VLMConfig (vlm_config.hpp)

Model configuration loaded from `config.json`. Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `model_type` | `VLMModelType` | Model family identifier |
| `hidden_size` | `size_t` | Embedding dimension |
| `scale_emb` | `float` | Embedding scale factor |
| `query_num` | `size_t` | Tokens per image slice (MiniCPM) |
| `im_start` / `im_end` | `std::string` | Image boundary tokens |
| `vision_start_token` / `vision_end_token` | `std::string` | Qwen2VL vision tokens |
| `image_context_token` | `std::string` | InternVL2 placeholder |
| `image_newline` | `std::vector<float>` | LLaVA-Next newline embedding |
| `sub_GN` / `glb_GN` | `std::vector<float>` | Phi3/Phi4 separator embeddings |

### ProcessorConfig (processor_config.hpp)

Image preprocessing parameters loaded from `preprocessor_config.json`:

| Field | Type | Description |
|-------|------|-------------|
| `image_size` | `size_t` | Target image size |
| `patch_size` | `size_t` | Vision transformer patch size |
| `norm_mean` / `norm_std` | `std::array<float,3>` | Normalization constants |
| `crop_size_height` / `crop_size_width` | `size_t` | LLaVA crop dimensions |
| `min_pixels` / `max_pixels` | `size_t` | Qwen2VL dynamic resolution bounds |
| `scale_resolution` | `size_t` | MiniCPM scale target |
| `max_slice_nums` | `size_t` | Max image tiles |

## Model Directory Structure

Expected layout of an exported model directory:

```
model_dir/
├── config.json                              # VLMConfig (model_type, tokens, hidden_size)
├── generation_config.json                   # GenerationConfig (temperature, top_k, etc.)
├── preprocessor_config.json                 # ProcessorConfig (image sizes, normalization)
├── openvino_language_model.xml/.bin         # Language model (required)
├── openvino_vision_embeddings_model.xml/.bin # Vision encoder (required)
├── openvino_text_embeddings_model.xml/.bin  # Text embedding lookup (required)
├── openvino_*.xml/.bin                      # Additional model-specific sub-models
└── tokenizer files                          # (tokenizer.model, vocab, merges, etc.)
```

## Attention Backend: SDPA vs PA/CB

The VLM pipeline supports two LLM inference backends. The front-end (vision encoding, embedding merge) is identical for both.

### Class Hierarchy

```
VLMPipelineBase (abstract)
├── VLMPipelineImpl                    ← SDPA path (default)
└── VLMContinuousBatchingAdapter       ← PA/CB path
    └── wraps ContinuousBatchingPipeline
```

### SDPA Path (`VLMPipelineImpl`)

- **KV-cache**: Manual management via `CacheState` — explicit trim/reset per chat turn
- **Generation**: Token-by-token loop with sampler
- **Transform**: `slice_before_matmul` — computes only last-token logits
- **Devices**: CPU, GPU, NPU

### PA/CB Path (`VLMContinuousBatchingAdapter`)

- **KV-cache**: Automatic via PagedAttention scheduler — paged block allocation/eviction
- **Generation**: Batch-oriented via wrapped `ContinuousBatchingPipeline`
- **Transform**: `SDPAToPagedAttention` + `gather_before_matmul`
- **Devices**: CPU, GPU (x86_64 / ARM64 only, no NPU)
- **Adapter pattern**: Wraps single VLM request into batch vectors for CB pipeline

### Backend Selection Logic (in `VLMPipeline` constructor)

1. **NPU** → always SDPA
2. **Linear attention models** (SSM, convolution states) → forced SDPA (PA does not support these)
3. **Explicit PA request** (user sets `scheduler_config`, `ATTENTION_BACKEND="PA"`, draft model, or prompt lookup) → PA/CB
4. **Auto-detect** → PA/CB if architecture supports it and model does not require SDPA
5. **Fallback** → SDPA if PA/CB construction fails

### Implications for New Model Enablement

- Model-specific code (`VisionEncoder`, `IInputsEmbedder`) works with **both** backends — no backend-specific logic needed there.
- If the new model uses non-standard attention (e.g., linear attention, SSM), it will be forced to SDPA. Check `apply_linear_attention_backend_constraints()` in `pipeline.cpp`.
- If the new model is incompatible with PA for other reasons, add a `requires_sdpa()` check similar to the existing Gemma3 check.

## Adding a New Model — Checklist

### 1. Add enum value

In `vlm_config.hpp`, add a new entry to `VLMModelType`:

```cpp
enum class VLMModelType {
    // ... existing
    NEW_MODEL,
};
```

Also add the string-to-enum mapping in `vlm_config.cpp`.

### 2. Create model directory

```
src/cpp/src/visual_language/new_model/
├── classes.hpp
└── classes.cpp
```

### 3. Implement VisionEncoder subclass

```cpp
class VisionEncoderNewModel : public VisionEncoder {
public:
    VisionEncoderNewModel(const std::filesystem::path& model_dir,
                          const std::string& device,
                          const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image,
                        const ov::AnyMap& config_map) override;
};
```

Responsibilities:
- Image preprocessing (resize, normalize, tile/slice)
- Run vision encoder IR model
- Pack results into `EncodedImage`

### 4. Implement IInputsEmbedder subclass

```cpp
class InputsEmbedderNewModel : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderNewModel(const VLMConfig& vlm_config,
                           const std::filesystem::path& model_dir,
                           const std::string& device,
                           const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<EncodedImage>& images,
        VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings,
        const std::vector<size_t>& image_sequence) override;

    std::vector<EncodedImage> encode_images(
        const std::vector<ov::Tensor>& images) override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images) const override;
};
```

Responsibilities:
- Normalize prompt: insert model-specific image placeholder tokens
- Tokenize text, get text embeddings via `EmbeddingsModel`
- Merge vision embeddings into the text embedding sequence at placeholder positions
- Return combined embedding tensor for the LLM

### 5. Register in factories

In `vision_encoder.cpp` — add to `VisionEncoder::create()`:
```cpp
else if (model_type == VLMModelType::NEW_MODEL) {
    return std::make_shared<VisionEncoderNewModel>(model_dir, device, properties);
}
```

In `inputs_embedder.cpp` — add to the `InputsEmbedder` constructor:
```cpp
else if (vlm_config.model_type == VLMModelType::NEW_MODEL) {
    m_impl = std::make_shared<InputsEmbedderNewModel>(vlm_config, model_dir, device, device_config);
}
```

Include the new header in both files:
```cpp
#include "visual_language/new_model/classes.hpp"
```

### 6. Update VLMConfig/ProcessorConfig (if needed)

Add model-specific fields to `VLMConfig` or `ProcessorConfig` and their JSON deserialization.

## Supported Models

The list of supported models changes as new model types are added.
To discover the current set, inspect:
- `VLMModelType` enum in `vlm_config.hpp`
- Model subdirectories: `ls src/cpp/src/visual_language/*/classes.hpp`
- Factory dispatch in `vision_encoder.cpp` and `inputs_embedder.cpp`

## Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Factory** | `VisionEncoder::create()`, `InputsEmbedder` ctor | Dispatch to model-specific implementations by `VLMModelType` |
| **Pimpl** | `VLMPipeline` → `VLMPipelineBase` → `VLMPipelineImpl` | Hide implementation from public API |
| **Strategy** | `IInputsEmbedder` subclasses | Each model implements its own embedding merge strategy |
| **Adapter** | `VLMContinuousBatchingAdapter` | Adapts VLM single-request API to batch `ContinuousBatchingPipeline` |
| **Registry** | `VisionRegistry` | Cache encoded images/videos, avoid re-encoding |

## Maintaining This Document

This document describes the stable architecture and interfaces of the GenAI VLM pipeline.
It intentionally avoids hard-coded lists of model types or model-specific details, since those change frequently.

When working on model enablement and discovering that this document is outdated or missing information:
1. Read the current source files referenced in **Source Layout** to verify interface signatures.
2. Update any changed interfaces, new parameters, or new factory patterns in this document.
3. If a new architectural pattern is introduced (e.g., a new base class, a new pipeline path), add a section describing the pattern and its purpose.
