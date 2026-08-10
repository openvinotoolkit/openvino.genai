// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/glm_edge_v/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

// Mllama-style preprocessing for GLM-Edge-V with a single image tile:
//  * aspect-preserving bicubic resize to fit a fixed square canvas (size_height x size_width),
//  * rescale (1/255) + normalize with mean/std,
//  * bottom-right zero padding to the full canvas (in normalized space),
//  * return a CHW float32 tensor of shape [1, 3, canvas_h, canvas_w].
ov::Tensor get_pixel_values_glm_edge_v(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    const int canvas_h = static_cast<int>(config.size_height);
    const int canvas_w = static_cast<int>(config.size_width);

    // Aspect-preserving resize so the image fits inside the canvas (top-left aligned).
    const float scale_h = static_cast<float>(canvas_h) / input_image.ny;
    const float scale_w = static_cast<float>(canvas_w) / input_image.nx;
    const float scale = std::min(scale_h, scale_w);
    int new_height = std::min(static_cast<int>(std::floor(input_image.ny * scale)), canvas_h);
    int new_width = std::min(static_cast<int>(std::floor(input_image.nx * scale)), canvas_w);
    new_height = std::max(new_height, 1);
    new_width = std::max(new_width, 1);

    clip_image_u8 resized_image;
    bicubic_resize(input_image, resized_image, new_width, new_height);

    // Normalize into a padded CHW float32 buffer. The reference MllamaImageProcessor pads the
    // raw image with zeros before normalization, so padded pixels equal (0 - mean) / std per
    // channel in normalized space (bottom-right padding, image aligned top-left).
    ov::Tensor pixel_values(ov::element::f32, {1, 3, static_cast<size_t>(canvas_h), static_cast<size_t>(canvas_w)});
    float* out = pixel_values.data<float>();

    const size_t plane = static_cast<size_t>(canvas_h) * canvas_w;
    for (int c = 0; c < 3; ++c) {
        const float pad_value = (0.0f - config.image_mean[c]) / config.image_std[c];
        std::fill_n(out + c * plane, plane, pad_value);
    }
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                float v = static_cast<float>(resized_image.buf[3 * (y * new_width + x) + c]) / 255.0f;
                v = (v - config.image_mean[c]) / config.image_std[c];
                out[c * plane + static_cast<size_t>(y) * canvas_w + x] = v;
            }
        }
    }
    return pixel_values;
}

} // namespace

EncodedImage VisionEncoderGLMEdgeV::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values = get_pixel_values_glm_edge_v(image, config);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    // The exported GLM-Edge-V vision model outputs float16 embeddings already in the LM hidden
    // dimension. Convert to float32 to match text embeddings for the placeholder merge.
    ov::Tensor image_features(ov::element::f32, infer_output.get_shape());
    if (infer_output.get_element_type() == ov::element::f32) {
        std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());
    } else if (infer_output.get_element_type() == ov::element::f16) {
        const ov::float16* src = infer_output.data<const ov::float16>();
        float* dst = image_features.data<float>();
        const size_t n = infer_output.get_size();
        for (size_t i = 0; i < n; ++i) {
            dst[i] = static_cast<float>(src[i]);
        }
    } else {
        infer_output.copy_to(image_features);
    }

    ImageSize resized_source_size{config.size_height / config.patch_size, config.size_width / config.patch_size};

    return {std::move(image_features), resized_source_size};
}

InputsEmbedderGLMEdgeV::InputsEmbedderGLMEdgeV(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) { }

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
    // GLM-Edge-V uses the boi token `<|begin_of_image|>` as the image placeholder. Each image is
    // expanded into one boi token per vision embedding (matching the reference chat template).
    std::string image_token = m_vlm_config.im_start;
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());

    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const ov::Tensor& image_embed = images.at(new_image_id - base_id).resized_source;
        std::string expanded_tag;
        for (size_t idx = 0; idx < image_embed.get_shape().at(1); ++idx) {
            expanded_tag += image_token;
        }
        OPENVINO_ASSERT(searched_pos < unified_prompt.length());
        searched_pos = unified_prompt.find(image_token, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos);
        unified_prompt.replace(searched_pos, image_token.length(), expanded_tag);
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
    ov::Tensor encoded_image_token = m_tokenizer.encode(m_vlm_config.im_start, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];
    return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
}

} // namespace ov::genai
