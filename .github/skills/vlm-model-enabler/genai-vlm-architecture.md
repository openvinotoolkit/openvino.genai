# GenAI VLM Pipeline Architecture Reference

Quick reference for the GenAI Visual Language Model pipeline: key interfaces, data flow, and extension points.

## Pipeline Data Flow

The diagram below follows the `ChatHistory` generation path, which is the most complete VLM use case.
The pipeline has two execution paths selected at construction time (see **Attention Backend** below).
Both paths use the same front-end sequence: vision encoding, history normalization, chat templating, tokenization, and embedding merge.
For SDPA this sequence runs in `VLMPipelineImpl`; for PA/CB it runs after `VLMContinuousBatchingAdapter` forwards the request to `ContinuousBatchingPipeline`.

```
User Input (ChatHistory + images/videos + optional video metadata)
     │
     ▼
 VLMPipeline::generate(history, ...)
     │
     ├──────────────────────────────────────────────┐
     ▼ SDPA path (VLMPipelineImpl)                  ▼ PA/CB path (VLMContinuousBatchingAdapter)
  Setup GenerationConfig and VLM options       Forward request to ContinuousBatchingPipeline
     │
     ▼
 VLMChatContext::process()
    ├── Synchronize ChatHistoryInternalState with the provided history
    ├── Register new images/videos in VisionRegistry and assign vision IDs
    ├── InputsEmbedder::encode_images()/encode_videos() for uncached vision inputs
    │     └── VisionEncoder::encode()/encode_frames() → cache EncodedImage/EncodedVideo embeddings in VisionRegistry
    ├── Normalize user messages with InputsEmbedder::normalize_prompt()
    │     └── Convert multipart content and model-specific vision placeholders
    ├── Build normalized_history
    ├── Resolve full-history and new-message vision sequences
    └── Report whether the SDPA KV cache must be reset
     │
     ▼
 Tokenizer::apply_chat_template(normalized_history, add_generation_prompt=true)
     │
     ▼
 Select embeddings input set
  ├── SDPA: choose full-history or new-message vision inputs
  └── PA/CB: use full-history vision inputs for the batch request
     │
     ▼
 InputsEmbedder::get_inputs_embeds() or get_inputs_embeds_with_token_type_ids()
  ├── Tokenize templated text
  ├── EmbeddingsModel::infer()         ← text token → embedding lookup
  └── Merge vision + text embeddings   ← model-specific merge logic
     │
     ├─────────────────────────────────────────────┐
     ▼ SDPA path (VLMPipelineImpl)                 ▼ PA/CB path (VLMContinuousBatchingAdapter)
  LLM with SDPA attention                     ContinuousBatchingPipeline
  ├── prepare_inputs_and_generate()            ├── Batched input_embeds request (and other model-specific LM inputs if any)
  ├── Manual KV-cache (CacheState)             ├── Paged KV-cache (scheduler)
  ├── Token-by-token sampler loop              ├── Batch-oriented generation
  └── slice_before_matmul transform            └── SDPAToPagedAttention transform
     │                                             │
     └──────────────┬──────────────────────────────┘
                ▼
      Sampling → Detokenization → VLMDecodedResults
                │
                ▼
      Update embedder chat history; rollback VLMChatContext on cancellation
```

For `ChatHistory` input, the chat template is applied after `VLMChatContext::process()` has produced a normalized history and before `InputsEmbedder::get_inputs_embeds()` tokenizes text and merges embeddings. This ordering is used in both SDPA and PA/CB paths. It matters because the template must see model-normalized message content, while embedding merge still needs the resolved vision sequences from the same processing step.

`VisionRegistry` assigns vision IDs to input images/videos, keeps copied original tensors for lookup and lifetime management (`register_vision()` copies with `tensor.copy_to()`), and caches the encoded `EncodedImage` / `EncodedVideo` embeddings after vision encoding. For the `ChatHistory` API, image/video encoding happens in `VLMChatContext::process()` through `InputsEmbedder::encode_images()` / `encode_videos()`. `InputsEmbedder::get_inputs_embeds()` receives already encoded image/video embeddings, not raw image/video tensors.

## Source Layout

All VLM sources are under `src/cpp/src/visual_language/`:

| File | Purpose |
|------|---------|
| `pipeline.cpp`, `pipeline_base.hpp` | Public `VLMPipeline` and internal `VLMPipelineBase` / `VLMPipelineImpl` |
| `inputs_embedder.hpp/.cpp` | `InputsEmbedder` (public) and `IInputsEmbedder` (model-specific interface) |
| `vision_encoder.hpp/.cpp` | `VisionEncoder` base class and factory |
| `vision_registry.hpp` | `VisionRegistry` — keeps copied original image/video tensors and caches encoded vision embeddings by ID |
| `vlm_config.hpp` | `VLMConfig` — loaded from `config.json`, includes `VLMModelType` enum |
| `processor_config.hpp` | `ProcessorConfig` — image preprocessing params |
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

Abstract base for image/video embedding extraction. One subclass per model type.

```cpp
class VisionEncoder {
public:
    using Ptr = std::shared_ptr<VisionEncoder>;

    // Factory — dispatches by VLMModelType
    static Ptr create(const std::filesystem::path& model_dir,
                      VLMModelType model_type,
                      const std::string& device,
                      const ov::AnyMap properties = {});

    // Encode a single image → EncodedImage embeddings
    virtual EncodedImage encode(const ov::Tensor& image,
                                const ov::AnyMap& config_map = {}) = 0;

    // Encode video frames → EncodedVideo embeddings (optional, throws by default)
    virtual EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames);

    ProcessorConfig get_processor_config() const;
    VideoProcessorConfig get_video_processor_config() const;

protected:
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_encoder;
    ProcessorConfig m_processor_config;
    VideoProcessorConfig m_video_processor_config;
};
```

### EncodedImage (vision_encoder.hpp)

Image embedding output of `VisionEncoder::encode()`. Not all fields are used by every model.

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

### EncodedVideo (vision_encoder.hpp)

Video embedding output of `VisionEncoder::encode_frames()`. Used by models with video input support.

```cpp
struct EncodedVideo {
    ov::Tensor video_features;          // Video embeddings after preprocessing and projection/resampling
    size_t num_video_tokens;            // Token count for prompt expansion
    ImageSize resized_source_size;      // H/patch, W/patch
    size_t frame_num;                   // Number of encoded frames
    VideoMetadata metadata;             // Metadata used for processing and prompt normalization
};
```

### IInputsEmbedder (inputs_embedder.hpp)

Model-specific interface for prompt normalization and embedding merge. One subclass per model type. Public `InputsEmbedder::encode_images()` / `encode_videos()` prepare encoded vision embeddings before `get_inputs_embeds()`: the default image path calls `VisionEncoder::encode()`, and video-capable embedders call `VisionEncoder::encode_frames()` after any sampling or metadata preparation.

```cpp
class InputsEmbedder::IInputsEmbedder {
public:
    // Merge text + encoded image embeddings into a single tensor for the LLM
    virtual ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<EncodedImage>& images,
        VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {}) = 0;

    // Complete image/video overload; default implementation falls back to the image-only method
    virtual ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<EncodedImage>& images,
        const std::vector<EncodedVideo>& videos,
        VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {},
        const std::vector<size_t>& video_sequence = {},
        const std::vector<std::pair<size_t, size_t>>& history_vision_count = {});

    virtual std::pair<ov::Tensor, ov::Tensor> get_inputs_embeds_with_token_type_ids(
        const std::string& prompt,
        const std::vector<EncodedImage>& images,
        const std::vector<EncodedVideo>& videos,
        VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {},
        const std::vector<size_t>& video_sequence = {},
        const std::vector<std::pair<size_t, size_t>>& history_vision_count = {});

    // Encode raw images/videos before get_inputs_embeds()
    virtual std::vector<EncodedImage> encode_images(
        const std::vector<ov::Tensor>& images);

    virtual std::vector<EncodedVideo> encode_videos(
        const std::vector<ov::Tensor>& videos,
        const std::vector<VideoMetadata>& videos_metadata = {});

    // Normalize prompt: insert model-specific image placeholder tokens
    virtual NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images) const = 0;

    // Complete image/video overload; default implementation falls back to the image-only method
    virtual NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_image_id,
        size_t base_video_id,
        const std::vector<EncodedImage>& images,
        const std::vector<EncodedVideo>& videos) const;

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

This table is not exhaustive. `VLMConfig` also carries model-specific fields needed by individual embedders.

### ProcessorConfig (processor_config.hpp)

Image preprocessing parameters. `VideoProcessorConfig` extends this for video-specific settings.

Processor config loading priority in `VisionEncoder::resolve_processor_configs()`:

1. Combined `processor_config.json` with `image_processor` and `video_processor` sections.
2. Fallback to `preprocessor_config.json` and `video_preprocessor_config.json`.
3. If `video_preprocessor_config.json` is absent, build `VideoProcessorConfig` from `preprocessor_config.json`.

| Field | Type | Description |
|-------|------|-------------|
| `image_size` | `size_t` | Target image size |
| `patch_size` | `size_t` | Vision transformer patch size |
| `norm_mean` / `norm_std` | `std::array<float,3>` | Normalization constants |
| `crop_size_height` / `crop_size_width` | `size_t` | LLaVA crop dimensions |
| `min_pixels` / `max_pixels` | `size_t` | Qwen2VL dynamic resolution bounds |
| `scale_resolution` | `size_t` | MiniCPM scale target |
| `max_slice_nums` | `size_t` | Max image tiles |

This table lists common fields only. Image and video configs can include additional model-specific preprocessing parameters.

### Preprocessing Utilities (clip.hpp / clip.cpp)

Reusable C++ image preprocessing primitives. Use these from a `VisionEncoder` subclass before composing model-specific pipelines.

| Utility | Purpose |
|---------|---------|
| `bicubic_resize()` / `bilinear_resize()` | Pillow-style image resizing |
| `center_crop()` | Center crop to target dimensions |
| `resize_and_pad_image()` | Resize with center padding |
| `get_image_patches()` | Extract grid patches for multi-resolution |
| `select_best_resolution()` | Pick optimal resolution from candidates |
| `clip_image_preprocess()` | Normalize and convert to CHW |
| `normalize_and_convert_to_chw()` | Double-precision normalization |
| `qwen2_vl_utils::smart_resize()` | Dynamic resolution with min/max pixel bounds (Qwen2VL-style); lives outside `clip.*` |

Note: GenAI's Pillow-style bicubic and bilinear resize implementations may differ from `transformers` at the pixel level by small amounts. Exact token-level match against `transformers` is not expected for image-text mode.

## Model Directory Structure

Expected layout of an exported model directory. Processor config files depend on the exporter version; newer exports use combined `processor_config.json`, while older exports use the fallback files.

```
model_dir/
├── config.json                              # VLMConfig (model_type, tokens, hidden_size)
├── generation_config.json                   # GenerationConfig (temperature, top_k, etc.)
├── processor_config.json                    # Combined image/video processor config (new exports)
├── preprocessor_config.json                 # Fallback ProcessorConfig for image preprocessing
├── video_preprocessor_config.json           # Optional fallback VideoProcessorConfig
├── openvino_language_model.xml/.bin         # Language model (required)
├── openvino_vision_embeddings_model.xml/.bin # Vision encoder (required)
├── openvino_text_embeddings_model.xml/.bin  # Text embedding lookup (required)
├── openvino_*.xml/.bin                      # Additional model-specific sub-models
└── tokenizer files                          # (tokenizer.model, vocab, merges, etc.)
```

## Attention Backend: SDPA vs PA/CB

The VLM pipeline supports two LLM inference backends. The front-end ordering (vision encoding, history/prompt normalization, chat templating where applicable, tokenization, and embedding merge) is shared; backend-specific code starts at LLM generation.

### Class Hierarchy

```
VLMPipelineBase (abstract)
├── VLMPipelineImpl                    ← SDPA path / fallback
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
2. **Explicit PA request** (user sets `scheduler_config`, `ATTENTION_BACKEND="PA"`, draft model, or prompt lookup) → PA/CB construction is attempted and errors are re-thrown
3. **Default PA attempt** (`extract_attention_backend()` defaults to `PA`) → PA/CB if architecture supports it and `requires_sdpa()` is false
4. **Fallback** → SDPA if PA/CB construction fails or is not selected

### Implications for New Model Enablement

- Model-specific code (`VisionEncoder`, `IInputsEmbedder`) works with **both** backends — no backend-specific logic needed there.
- If the new model is incompatible with PA, add a `requires_sdpa()` check similar to the existing Gemma3/Gemma4 checks.

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

    EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames) override;  // if video is supported
};
```

Responsibilities:
- Image/video preprocessing (resize, normalize, tile/slice, sample frames when applicable)
- Run vision encoder IR model and any model-specific projection/resampler models
- Pack resulting vision embeddings into `EncodedImage` / `EncodedVideo`

Implement `encode_frames()` only for models with video input support.

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

    ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<EncodedImage>& images,
        const std::vector<EncodedVideo>& videos,
        VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings,
        const std::vector<size_t>& image_sequence,
        const std::vector<size_t>& video_sequence,
        const std::vector<std::pair<size_t, size_t>>& history_vision_count) override;

    std::vector<EncodedImage> encode_images(  // override only if default image encoding is insufficient
        const std::vector<ov::Tensor>& images) override;

    std::vector<EncodedVideo> encode_videos(  // if video is supported
        const std::vector<ov::Tensor>& videos,
        const std::vector<VideoMetadata>& videos_metadata) override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images) const override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_image_id,
        size_t base_video_id,
        const std::vector<EncodedImage>& images,
        const std::vector<EncodedVideo>& videos) const override;
};
```

Responsibilities:
- Normalize prompt: insert model-specific image/video placeholder tokens
- Tokenize text, get text embeddings via `EmbeddingsModel`
- Merge encoded image/video embeddings into the text embedding sequence at placeholder positions
- Return combined embedding tensor for the LLM

`InputsEmbedder::IInputsEmbedder` can be previous model version class, usefull for logic sharing in case of similar models.
`InputsEmbedder::get_inputs_embeds()` receives encoded image/video embeddings. Raw image/video tensors are converted earlier by `InputsEmbedder::encode_images()` / `encode_videos()`. The image-only `get_inputs_embeds()` and `normalize_prompt()` overloads are required. Implement the image/video overloads only when the model supports video or needs custom video-aware behavior. The base `encode_images()` implementation calls `VisionEncoder::encode()`; override it only for model-specific image preparation. Video-capable embedders override `encode_videos()` and call `VisionEncoder::encode_frames()` after sampling and metadata preparation.

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

### 6. Update VLMConfig/ProcessorConfig/VideoProcessorConfig (if needed)

Add model-specific fields to `VLMConfig`, `ProcessorConfig`, or `VideoProcessorConfig` and their JSON deserialization.

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
| **Registry** | `VisionRegistry` | Keep copied original vision tensors and cache encoded vision embeddings, avoid re-encoding |

## Maintaining This Document

This document describes the stable architecture and interfaces of the GenAI VLM pipeline.
It intentionally avoids hard-coded lists of model types or model-specific details, since those change frequently.

When working on model enablement and discovering that this document is outdated or missing information:
1. Read the current source files referenced in **Source Layout** to verify interface signatures.
2. Update any changed interfaces, new parameters, or new factory patterns in this document.
3. If a new architectural pattern is introduced (e.g., a new base class, a new pipeline path), add a section describing the pattern and its purpose.
