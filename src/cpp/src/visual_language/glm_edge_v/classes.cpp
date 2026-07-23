// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/glm_edge_v/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

// Native tag used by GenAI to mark an image position in the prompt before the
// chat template is applied. It is expanded into repeated boi tokens below.
const std::string NATIVE_TAG = "<image>";

// Mllama-style resize (max_image_tiles == 1): fit the image inside a single
// canvas_size x canvas_size tile preserving aspect ratio, then pad the
// bottom/right with `pad_value`. Matches HF MllamaImageProcessor when
// max_image_tiles == 1 (canvas is always one tile).
clip_image_u8 resize_and_pad_glm(const clip_image_u8& image, int canvas_size, uint8_t pad_value = 0) {
    // target dimensions are clipped between tile_size and canvas_size; with a
    // single tile canvas both equal canvas_size.
    const int target = canvas_size;

    const float scale_h = static_cast<float>(target) / image.ny;
    const float scale_w = static_cast<float>(target) / image.nx;

    int new_width;
    int new_height;
    if (scale_w < scale_h) {
        new_width = target;
        new_height = std::min(std::max(static_cast<int>(std::floor(image.ny * scale_w)), 1), target);
    } else {
        new_height = target;
        new_width = std::min(std::max(static_cast<int>(std::floor(image.nx * scale_h)), 1), target);
    }

    clip_image_u8 resized_image;
    bicubic_resize(image, resized_image, new_width, new_height);

    clip_image_u8 padded_image;
    padded_image.nx = target;
    padded_image.ny = target;
    padded_image.buf.assign(static_cast<size_t>(3) * target * target, pad_value);

    // Top-left aligned copy; remaining bottom/right stays padded.
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                padded_image.buf[3 * (y * target + x) + c] =
                    resized_image.buf[3 * (y * new_width + x) + c];
            }
        }
    }
    return padded_image;
}

ov::Tensor get_pixel_values_glm(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    // GLM-Edge-V uses a square canvas (size.height == size.width).
    const int canvas_size = static_cast<int>(config.size_height);
    clip_image_u8 resized_image = resize_and_pad_glm(input_image, canvas_size);

    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    // clip_image_preprocess rescales by 1/255 and normalizes with mean/std.
    clip_image_f32 normalized_image = clip_image_preprocess(ctx, resized_image);
    return clip_image_f32_to_tensor(normalized_image);
}

// Scatter image embeddings into text embeddings at boi_token positions.
// GLM-Edge-V uses a single repeated placeholder token (boi_token_id), like
// InternVL's image context token.
ov::Tensor merge_text_and_image_embeddings_glm(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds,
    const std::vector<ov::Tensor>& image_embeds,
    int64_t image_pad_token_id) {
    auto text_embeds_shape = text_embeds.get_shape();
    size_t batch_size = text_embeds_shape.at(0);
    size_t seq_len = text_embeds_shape.at(1);
    size_t embed_dim = text_embeds_shape.at(2);

    ov::Tensor merged_embeds(text_embeds.get_element_type(), text_embeds_shape);

    const float* text_embeds_data = text_embeds.data<float>();
    const int64_t* input_ids_data = input_ids.data<int64_t>();
    float* merged_embeds_data = merged_embeds.data<float>();

    size_t flattened_size = batch_size * seq_len;
    std::vector<bool> image_tokens_mask(flattened_size, false);
    size_t image_tokens_count = 0;
    for (size_t i = 0; i < flattened_size; ++i) {
        if (input_ids_data[i] == image_pad_token_id) {
            image_tokens_mask[i] = true;
            ++image_tokens_count;
        }
    }

    OPENVINO_ASSERT(image_tokens_count > 0, "input_ids does not contain GLM-Edge-V image placeholder token ids");

    size_t image_idx = 0;
    size_t image_token_idx = 0;
    for (size_t flat_idx = 0; flat_idx < flattened_size; ++flat_idx) {
        size_t offset = flat_idx * embed_dim;
        if (image_tokens_mask[flat_idx]) {
            OPENVINO_ASSERT(image_idx < image_embeds.size(), "Not enough image embeddings for image placeholder tokens");
            const ov::Tensor& single_image_embeds = image_embeds[image_idx];
            const auto single_shape = single_image_embeds.get_shape();
            // vision output shape is [num_images(==1), num_tokens, embed_dim]
            const size_t num_image_tokens = single_shape.at(single_shape.size() - 2);
            const float* image_embeds_data = single_image_embeds.data<float>();
            std::copy_n(image_embeds_data + image_token_idx * embed_dim, embed_dim, merged_embeds_data + offset);
            ++image_token_idx;
            if (image_token_idx == num_image_tokens) {
                ++image_idx;
                image_token_idx = 0;
            }
        } else {
            std::copy_n(text_embeds_data + offset, embed_dim, merged_embeds_data + offset);
        }
    }

    return merged_embeds;
}

} // namespace

EncodedImage VisionEncoderGLMEdgeV::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values = get_pixel_values_glm(image, config);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    return {std::move(image_features)};
}

InputsEmbedderGLMEdgeV::InputsEmbedderGLMEdgeV(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) { }

InputsEmbedderGLMEdgeV::InputsEmbedderGLMEdgeV(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

std::vector<ov::genai::EncodedImage> InputsEmbedderGLMEdgeV::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
    }
    return embeds;
}

NormalizedPrompt InputsEmbedderGLMEdgeV::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    const std::string& image_pad_token = m_vlm_config.image_pad_token;
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG, base_id, images.size());

    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const ov::Tensor& image_embeds = images.at(new_image_id - base_id).resized_source;
        const auto shape = image_embeds.get_shape();
        // vision output shape is [num_images(==1), num_tokens, embed_dim]
        const size_t num_image_tokens = shape.at(shape.size() - 2);

        std::string expanded_tag;
        expanded_tag.reserve(num_image_tokens * image_pad_token.size());
        for (size_t idx = 0; idx < num_image_tokens; ++idx) {
            expanded_tag += image_pad_token;
        }
        OPENVINO_ASSERT(searched_pos < unified_prompt.length());
        searched_pos = unified_prompt.find(NATIVE_TAG, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos);
        unified_prompt.replace(searched_pos, NATIVE_TAG.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderGLMEdgeV::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
    }

    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor encoded_image_pad_token = m_tokenizer.encode(m_vlm_config.image_pad_token, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_pad_token_id = encoded_image_pad_token.data<int64_t>()[encoded_image_pad_token.get_size() - 1];
    return merge_text_and_image_embeddings_glm(input_ids, text_embeds, image_embeds, image_pad_token_id);
}

} // namespace ov::genai
