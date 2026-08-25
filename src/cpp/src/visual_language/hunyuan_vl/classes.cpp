// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/hunyuan_vl/classes.hpp"

#include <algorithm>
#include <numeric>

#include "visual_language/clip.hpp"
#include "visual_language/qwen2vl/classes.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

// The HunYuanVL chat template hardcodes these special tokens around an image, so they are hardcoded here as well:
//   <image_start><image_token><image_end>
// image_start / image_end are ordinary text tokens; image_token is the placeholder that is expanded to one token per
// merged image embedding row and later replaced by the vision features.
const std::string IMAGE_START_TOKEN = "<\xef\xbd\x9chy_place\xe2\x96\x81holder\xe2\x96\x81no\xe2\x96\x81" "100\xef\xbd\x9c>";
const std::string IMAGE_TOKEN = "<\xef\xbd\x9chy_place\xe2\x96\x81holder\xe2\x96\x81no\xe2\x96\x81" "102\xef\xbd\x9c>";
const std::string IMAGE_END_TOKEN = "<\xef\xbd\x9chy_place\xe2\x96\x81holder\xe2\x96\x81no\xe2\x96\x81" "101\xef\xbd\x9c>";

// NATIVE_TAG is the placeholder emitted by the chat template for a single image (the bare image token). It is expanded
// to <image_start> + num_image_tokens * <image_token> + <image_end> in normalize_prompt().
const std::string NATIVE_TAG = IMAGE_TOKEN;

/**
 * @brief Preprocess a single image into the flattened patch tensor consumed by the HunYuanVL vision graph.
 *
 * The layout matches transformers' HunYuanVLImageProcessor: smart-resize to a multiple of patch_size * merge_size,
 * rescale + CLIP-normalize, split into patch_size x patch_size patches and flatten each patch to
 * channels * temporal_patch_size * patch_size * patch_size. HunYuanVL uses temporal_patch_size == 1.
 */
ov::Tensor get_pixel_values_hunyuan(const ov::Tensor& image, const ProcessorConfig& config, size_t& out_grid_h, size_t& out_grid_w) {
    const size_t patch_size = config.patch_size;
    const size_t merge_size = config.merge_size;
    const size_t temporal_patch_size = config.temporal_patch_size;

    ov::Shape orig_shape = image.get_shape();
    ImageSize target_image_size = qwen2_vl_utils::smart_resize(
        orig_shape.at(1),
        orig_shape.at(2),
        patch_size * merge_size,
        config.min_pixels,
        config.max_pixels);

    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_u8 resized_image;
    bicubic_resize(input_image, resized_image, target_image_size.width, target_image_size.height);

    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
    clip_image_f32 normalized_image = clip_image_preprocess(ctx, resized_image);

    // [1, 3, H, W]
    ov::Tensor patches = clip_image_f32_to_tensor(normalized_image);

    const size_t channel = patches.get_shape().at(1);
    const size_t grid_t = 1;
    const size_t grid_h = target_image_size.height / patch_size;
    const size_t grid_w = target_image_size.width / patch_size;

    ov::Tensor reshaped_patches = qwen2_vl_utils::reshape_image_patches(
        patches, grid_t, grid_h, grid_w, channel, temporal_patch_size, patch_size, merge_size);
    ov::Tensor transposed_patches = qwen2_vl_utils::transpose_image_patches(reshaped_patches);

    ov::Shape flattened_shape{
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size
    };
    ov::Tensor flattened_patches(transposed_patches.get_element_type(), flattened_shape);
    std::memcpy(flattened_patches.data(), transposed_patches.data(), transposed_patches.get_byte_size());

    out_grid_h = grid_h;
    out_grid_w = grid_w;
    return flattened_patches;
}

/**
 * @brief Replace image placeholder tokens in the text embeddings with the merged vision embeddings.
 *
 * Each image contributes a contiguous run of image_token_id placeholders whose length equals the number of rows of the
 * corresponding vision output. Placeholders are filled from the images in the order given by image_embeds.
 */
ov::Tensor merge_text_and_image_embeddings_hunyuan(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds,
    const std::vector<ov::Tensor>& image_embeds,
    int64_t image_token_id) {
    ov::Shape text_shape = text_embeds.get_shape();
    size_t batch_size = text_shape.at(0);
    size_t seq_len = text_shape.at(1);
    size_t embed_dim = text_shape.at(2);
    size_t flattened_size = batch_size * seq_len;

    ov::Tensor merged_embeds(text_embeds.get_element_type(), text_shape);
    const float* text_data = text_embeds.data<float>();
    const int64_t* ids = input_ids.data<int64_t>();
    float* out_data = merged_embeds.data<float>();
    std::memcpy(out_data, text_data, text_embeds.get_byte_size());

    size_t image_idx = 0;
    size_t row_in_image = 0;
    for (size_t i = 0; i < flattened_size; ++i) {
        if (ids[i] != image_token_id) {
            continue;
        }
        OPENVINO_ASSERT(image_idx < image_embeds.size(),
            "Number of image placeholder tokens exceeds the number of provided image embeddings.");
        const ov::Tensor& single = image_embeds[image_idx];
        ov::Shape single_shape = single.get_shape();
        // Vision output shape is [1, num_rows, embed_dim].
        size_t num_rows = single_shape.at(single_shape.size() - 2);
        OPENVINO_ASSERT(single_shape.at(single_shape.size() - 1) == embed_dim,
            "Vision embedding dimension does not match text embedding dimension.");
        const float* img_data = single.data<float>();
        std::copy_n(img_data + row_in_image * embed_dim, embed_dim, out_data + i * embed_dim);
        if (++row_in_image == num_rows) {
            row_in_image = 0;
            ++image_idx;
        }
    }
    OPENVINO_ASSERT(image_idx == image_embeds.size() && row_in_image == 0,
        "Number of image placeholder tokens does not match the number of vision embedding rows.");

    return merged_embeds;
}

} // namespace

EncodedImage VisionEncoderHunyuanVL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    size_t grid_h = 0;
    size_t grid_w = 0;
    ov::Tensor pixel_values = get_pixel_values_hunyuan(image, config, grid_h, grid_w);
    size_t num_patches = pixel_values.get_shape().at(0);

    // Full (block) attention over a single image: an all-zero additive SDPA mask.
    ov::Tensor attention_mask(ov::element::f32, ov::Shape{1, 1, num_patches, num_patches});
    std::fill_n(attention_mask.data<float>(), attention_mask.get_size(), 0.0f);

    // Only the shape of grid_hw matters: it carries the patch grid (h, w) to the graph.
    ov::Tensor grid_hw(ov::element::i64, ov::Shape{grid_h, grid_w});
    std::fill_n(grid_hw.data<int64_t>(), grid_hw.get_size(), int64_t{0});

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    encoder.set_tensor("pixel_values", pixel_values);
    encoder.set_tensor("attention_mask", attention_mask);
    encoder.set_tensor("grid_hw", grid_hw);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    EncodedImage encoded_image;
    encoded_image.resized_source = std::move(image_features);
    // Store the patch grid (before spatial merge) so that the inputs embedder can rebuild the multimodal RoPE indices.
    encoded_image.resized_source_size = ImageSize{grid_h, grid_w};
    // One LLM token per merged vision row (includes begin, per-row newline and end tokens).
    encoded_image.num_image_tokens = encoded_image.resized_source.get_shape().at(encoded_image.resized_source.get_shape().size() - 2);
    return encoded_image;
}

InputsEmbedderHunyuanVL::InputsEmbedderHunyuanVL(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {
    init_token_ids();
}

InputsEmbedderHunyuanVL::InputsEmbedderHunyuanVL(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    init_token_ids();
}

void InputsEmbedderHunyuanVL::init_token_ids() {
    auto last_token_id = [this](const std::string& token) -> int64_t {
        ov::Tensor ids = m_tokenizer.encode(token, ov::genai::add_special_tokens(false)).input_ids;
        OPENVINO_ASSERT(ids.get_size() > 0, "Failed to tokenize HunYuanVL special token.");
        return ids.data<int64_t>()[ids.get_size() - 1];
    };
    m_image_token_id = last_token_id(IMAGE_TOKEN);
    m_image_start_token_id = last_token_id(IMAGE_START_TOKEN);
    m_image_end_token_id = last_token_id(IMAGE_END_TOKEN);
}

NormalizedPrompt InputsEmbedderHunyuanVL::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG, base_id, images.size());

    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const size_t num_image_tokens = images.at(new_image_id - base_id).num_image_tokens;

        std::string expanded_tag;
        expanded_tag.reserve(IMAGE_START_TOKEN.length() + IMAGE_TOKEN.length() * num_image_tokens + IMAGE_END_TOKEN.length());
        expanded_tag.append(IMAGE_START_TOKEN);
        for (size_t idx = 0; idx < num_image_tokens; ++idx) {
            expanded_tag.append(IMAGE_TOKEN);
        }
        expanded_tag.append(IMAGE_END_TOKEN);

        searched_pos = unified_prompt.find(NATIVE_TAG, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos,
            "Failed to locate HunYuanVL image placeholder while expanding the prompt.");
        unified_prompt.replace(searched_pos, NATIVE_TAG.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

std::pair<ov::Tensor, int64_t> InputsEmbedderHunyuanVL::create_position_ids(
    const ov::Tensor& input_ids_tensor,
    const std::vector<std::array<size_t, 2>>& images_grid_hw,
    const std::vector<size_t>& images_sequence
) const {
    const size_t merge_size = m_vision_encoder->get_processor_config().merge_size;
    const int64_t* input_ids = input_ids_tensor.data<int64_t>();
    const size_t batch_size = input_ids_tensor.get_shape().at(0);
    const size_t seq_len = input_ids_tensor.get_shape().at(1);
    OPENVINO_ASSERT(batch_size == 1, "HunYuanVL supports batch size 1 only.");
    OPENVINO_ASSERT(m_num_mrope_axes >= 3, "HunYuanVL expects at least 3 multimodal RoPE axes.");

    ov::Tensor position_ids{ov::element::i64, {m_num_mrope_axes, batch_size, seq_len}};
    int64_t* pos_data = position_ids.data<int64_t>();
    // Default: every axis follows the plain 1D sequence position.
    for (size_t axis = 0; axis < m_num_mrope_axes; ++axis) {
        std::iota(pos_data + axis * seq_len, pos_data + (axis + 1) * seq_len, int64_t{0});
    }

    // The last three axes encode (width, height, image_index); leading axes keep the 1D position.
    const size_t width_axis = m_num_mrope_axes - 3;
    const size_t height_axis = m_num_mrope_axes - 2;
    const size_t index_axis = m_num_mrope_axes - 1;

    size_t grid_idx = 0;
    int64_t image_index = 0;
    size_t pos = 0;
    while (pos < seq_len) {
        if (input_ids[pos] != m_image_token_id) {
            ++pos;
            continue;
        }
        size_t span_start = pos;
        while (pos < seq_len && input_ids[pos] == m_image_token_id) {
            ++pos;
        }
        size_t span_end = pos;
        size_t span_length = span_end - span_start;

        OPENVINO_ASSERT(grid_idx < images_sequence.size(),
            "Found more image placeholder spans than provided images.");
        const auto& grid = images_grid_hw.at(images_sequence.at(grid_idx));
        const size_t llm_grid_h = grid.at(0) / merge_size;
        const size_t llm_grid_w = grid.at(1) / merge_size;
        const size_t grid_tokens = llm_grid_h * (llm_grid_w + 1);

        size_t grid_start = span_start;
        if (span_length == grid_tokens + 2) {
            // Skip the begin token; the trailing token is the end token.
            grid_start = span_start + 1;
        } else {
            OPENVINO_ASSERT(span_length == grid_tokens,
                "HunYuanVL image placeholder span length ", span_length,
                " does not match grid tokens ", grid_tokens, " (+2).");
        }

        for (size_t k = 0; k < grid_tokens; ++k) {
            size_t token_pos = grid_start + k;
            int64_t col = static_cast<int64_t>(k % (llm_grid_w + 1));  // width index (0..llm_grid_w)
            int64_t row = static_cast<int64_t>(k / (llm_grid_w + 1));  // height index (0..llm_grid_h-1)
            pos_data[width_axis * seq_len + token_pos] = col;
            pos_data[height_axis * seq_len + token_pos] = row;
            pos_data[index_axis * seq_len + token_pos] = image_index;
        }
        ++image_index;
        ++grid_idx;
    }

    OPENVINO_ASSERT(grid_idx == images_sequence.size(),
        "Found fewer image placeholder spans than provided images.");

    const int64_t max_pos = *std::max_element(pos_data, pos_data + position_ids.get_size());
    const int64_t rope_delta = max_pos + 1 - static_cast<int64_t>(seq_len);
    return {position_ids, rope_delta};
}

ov::Tensor InputsEmbedderHunyuanVL::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    std::vector<std::array<size_t, 2>> images_grid_hw;
    images_grid_hw.reserve(images.size());
    for (const auto& encoded_image : images) {
        images_grid_hw.push_back({encoded_image.resized_source_size.height, encoded_image.resized_source_size.width});
    }

    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    std::tie(m_position_ids, m_rope_delta) = create_position_ids(input_ids, images_grid_hw, images_sequence);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
    }

    return merge_text_and_image_embeddings_hunyuan(input_ids, text_embeds, image_embeds, m_image_token_id);
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderHunyuanVL::get_position_ids(const size_t inputs_embeds_size, const size_t history_size) {
    if (history_size != 0) {
        return get_generation_phase_position_ids(inputs_embeds_size, history_size, m_rope_delta);
    }
    return {m_position_ids, m_rope_delta};
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderHunyuanVL::get_generation_phase_position_ids(const size_t inputs_embeds_size, const size_t history_size, int64_t rope_delta) {
    OPENVINO_ASSERT(history_size != 0, "get_generation_phase_position_ids() should only be called during the generation phase.");
    ov::Tensor position_ids{ov::element::i64, {m_num_mrope_axes, 1, inputs_embeds_size}};
    int64_t new_pos_id = static_cast<int64_t>(history_size) + rope_delta;
    for (size_t axis = 0; axis < m_num_mrope_axes; ++axis) {
        int64_t* pos_data = position_ids.data<int64_t>() + axis * inputs_embeds_size;
        std::iota(pos_data, pos_data + inputs_embeds_size, new_pos_id);
    }
    return {position_ids, rope_delta};
}

void InputsEmbedderHunyuanVL::start_chat(const std::string& system_message) {
    IInputsEmbedder::start_chat(system_message);
    m_position_ids = ov::Tensor();
    m_rope_delta = 0;
}

void InputsEmbedderHunyuanVL::finish_chat() {
    IInputsEmbedder::finish_chat();
    m_position_ids = ov::Tensor();
    m_rope_delta = 0;
}

} // namespace ov::genai
