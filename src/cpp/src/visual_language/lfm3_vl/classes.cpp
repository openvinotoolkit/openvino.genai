// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/lfm3_vl/classes.hpp"

#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "visual_language/clip.hpp"
#include "visual_language/qwen2vl/classes.hpp"

namespace ov::genai {

EncodedImage VisionEncoderLFM3VL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    const ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);
    OPENVINO_ASSERT(config.patch_size > 0 && config.merge_size > 0 && config.temporal_patch_size > 0,
                    "LFM3-VL patch sizes and spatial merge size must be positive.");
    const ov::Shape& shape = image.get_shape();
    OPENVINO_ASSERT(shape.size() == 4 && shape[0] == 1 && shape[1] > 0 && shape[2] > 0 && shape[3] == 3,
                    "LFM3-VL expects a non-empty RGB image with shape [1, height, width, 3].");
    const ImageSize target = qwen2_vl_utils::smart_resize(shape[1], shape[2],
                                                        config.patch_size * config.merge_size,
                                                        config.min_pixels, config.max_pixels);
    clip_image_u8 resized;
    bicubic_resize(tensor_to_clip_image_u8(image), resized, target.width, target.height);
    clip_ctx context;
    std::copy(config.image_mean.begin(), config.image_mean.end(), context.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), context.image_std);
    const clip_image_f32 normalized = clip_image_preprocess(context, resized);

    const size_t grid_h = target.height / config.patch_size;
    const size_t grid_w = target.width / config.patch_size;
    const size_t patch_dim = 3 * config.temporal_patch_size * config.patch_size * config.patch_size;
    ov::Tensor pixel_values(ov::element::f32, {1, grid_h, grid_w, patch_dim});
    float* dst = pixel_values.data<float>();
    // Optimum undoes Qwen's merge-block ordering before invoking the exported vision model.
    for (size_t row = 0; row < grid_h; ++row) {
        for (size_t col = 0; col < grid_w; ++col) {
            for (size_t channel = 0; channel < 3; ++channel) {
                // Match Torchvision's fused rescaling and normalization.
                const float mean = config.image_mean[channel] * 255.0f;
                const float stddev = config.image_std[channel] * 255.0f;
                for (size_t temporal = 0; temporal < config.temporal_patch_size; ++temporal) {
                    for (size_t y = 0; y < config.patch_size; ++y) {
                        const size_t src_offset = ((row * config.patch_size + y) * target.width +
                                                   col * config.patch_size) * 3 + channel;
                        for (size_t x = 0; x < config.patch_size; ++x) {
                            *dst++ = (static_cast<float>(resized.buf[src_offset + x * 3]) - mean) / stddev;
                        }
                    }
                }
            }
        }
    }

    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = guard.get();
    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();
    const ov::Tensor output = encoder.get_output_tensor();
    const size_t num_tokens = grid_h * grid_w / (config.merge_size * config.merge_size);
    OPENVINO_ASSERT(output.get_shape().size() == 3 && output.get_shape()[0] == 1 &&
                        output.get_shape()[1] == num_tokens,
                    "LFM3-VL vision output shape does not match the spatial patch grid.");
    ov::Tensor features(output.get_element_type(), output.get_shape());
    output.copy_to(features);
    EncodedImage result{std::move(features), {grid_h, grid_w}};
    result.num_image_tokens = num_tokens;
    return result;
}

InputsEmbedderLFM3VL::InputsEmbedderLFM3VL(const VLMConfig& vlm_config,
                                         const std::filesystem::path& model_dir,
                                         const Tokenizer& tokenizer,
                                         const std::string& device,
                                         const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {}

InputsEmbedderLFM3VL::InputsEmbedderLFM3VL(const VLMConfig& vlm_config,
                                         const ModelsMap& models_map,
                                         const Tokenizer& tokenizer,
                                         const std::filesystem::path& config_dir_path,
                                         const std::string& device,
                                         const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

std::vector<EncodedImage> InputsEmbedderLFM3VL::encode_images(const std::vector<ov::Tensor>& images) {
    const auto single_images = to_single_image_tensors(images);
    std::vector<EncodedImage> encoded_images;
    encoded_images.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        encoded_images.emplace_back(m_vision_encoder->encode(image));
    }
    return encoded_images;
}

NormalizedPrompt InputsEmbedderLFM3VL::normalize_prompt(const std::string& prompt,
                                                       size_t base_id,
                                                       const std::vector<EncodedImage>& images) const {
    const std::string native_tag =
        m_vlm_config.vision_start_token + m_vlm_config.image_token + m_vlm_config.vision_end_token;
    auto [unified_prompt, images_sequence] = normalize(prompt, native_tag, native_tag, base_id, images.size());
    size_t offset = 0;
    for (size_t image_id : images_sequence) {
        const auto& image = images.at(image_id - base_id);
        std::string expanded_tag = m_vlm_config.vision_start_token;
        for (size_t idx = 0; idx < image.num_image_tokens; ++idx) {
            expanded_tag += m_vlm_config.image_token;
        }
        expanded_tag += m_vlm_config.vision_end_token;
        const size_t pos = unified_prompt.find(native_tag, offset);
        OPENVINO_ASSERT(pos != std::string::npos, "LFM3-VL image tag is missing during prompt normalization.");
        unified_prompt.replace(pos, native_tag.size(), expanded_tag);
        offset = pos + expanded_tag.size();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderLFM3VL::get_inputs_embeds(const std::string& prompt,
                                                 const std::vector<EncodedImage>& images,
                                                 VLMPerfMetrics& metrics,
                                                 bool recalculate_merged_embeddings,
                                                 const std::vector<size_t>& image_sequence) {
    const ov::Tensor input_ids = get_encoded_input_ids(prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> guard(m_embedding->get_request_queue().get());
    ov::Tensor text_embeds = get_text_embedding(guard.get(), input_ids, metrics);

    std::vector<ov::Tensor> image_embeds;
    size_t num_image_tokens = 0;
    for (size_t image_id : image_sequence) {
        const auto& image = images.at(image_id);
        image_embeds.push_back(image.resized_source);
        num_image_tokens += image.num_image_tokens;
    }
    const int64_t* ids = input_ids.data<const int64_t>();
    OPENVINO_ASSERT(static_cast<size_t>(std::count(ids, ids + input_ids.get_size(), m_vlm_config.image_token_id)) ==
                        num_image_tokens,
                    "LFM3-VL image token count does not match the number of vision embeddings.");
    if (!image_embeds.empty()) {
        return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds,
                                                           m_vlm_config.image_token_id);
    }
    ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    text_embeds.copy_to(inputs_embeds);
    return inputs_embeds;
}

}  // namespace ov::genai
