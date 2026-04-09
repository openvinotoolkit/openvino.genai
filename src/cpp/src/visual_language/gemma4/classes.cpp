// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/gemma4/classes.hpp"

#include <cmath>

#include "utils.hpp"
#include "visual_language/clip.hpp"

namespace ov::genai {
namespace {

/// @brief Compute target dimensions for aspect-ratio-preserving resize.
/// Matches HF Gemma4ImageProcessor.get_aspect_ratio_preserving_size().
std::pair<size_t, size_t> get_aspect_ratio_preserving_size(size_t height,
                                                           size_t width,
                                                           size_t patch_size,
                                                           size_t max_patches,
                                                           size_t pooling_kernel_size) {
    const double total_px = static_cast<double>(height * width);
    const double target_px = static_cast<double>(max_patches) * static_cast<double>(patch_size * patch_size);
    const double factor = std::sqrt(target_px / total_px);
    const double ideal_height = factor * static_cast<double>(height);
    const double ideal_width = factor * static_cast<double>(width);
    const size_t side_mult = pooling_kernel_size * patch_size;

    size_t target_height = static_cast<size_t>(std::floor(ideal_height / static_cast<double>(side_mult))) * side_mult;
    size_t target_width = static_cast<size_t>(std::floor(ideal_width / static_cast<double>(side_mult))) * side_mult;

    OPENVINO_ASSERT(target_height != 0 || target_width != 0,
                    "Cannot resize image to 0x0. Dimensions must be divisible by ",
                    side_mult);

    const size_t max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if (target_height == 0) {
        target_height = side_mult;
        target_width = std::min(
            static_cast<size_t>(std::floor(static_cast<double>(width) / static_cast<double>(height))) * side_mult,
            max_side_length);
    } else if (target_width == 0) {
        target_width = side_mult;
        target_height = std::min(
            static_cast<size_t>(std::floor(static_cast<double>(height) / static_cast<double>(width))) * side_mult,
            max_side_length);
    }

    return {target_height, target_width};
}

/// @brief Extract patches from a CHW float image into a flat [num_patches, patch_dim] tensor.
/// Each patch stores pixels in HWC order within the patch, matching HF convert_image_to_patches().
/// @param float_image CHW float image (clip_image_f32)
/// @param patch_size Size of each square patch
/// @param output Pointer to output buffer, must have space for num_patches * patch_dim floats
/// @param num_patches_h Number of patches along height
/// @param num_patches_w Number of patches along width
void extract_patches(const clip_image_f32& float_image,
                     size_t patch_size,
                     float* output,
                     size_t num_patches_h,
                     size_t num_patches_w) {
    const size_t patch_dim = patch_size * patch_size * 3;
    const size_t nx = static_cast<size_t>(float_image.nx);
    const size_t ny = static_cast<size_t>(float_image.ny);

    for (size_t py = 0; py < num_patches_h; py++) {
        for (size_t px = 0; px < num_patches_w; px++) {
            const size_t patch_idx = py * num_patches_w + px;
            float* dst = output + patch_idx * patch_dim;
            for (size_t y = 0; y < patch_size; y++) {
                for (size_t x = 0; x < patch_size; x++) {
                    const size_t src_y = py * patch_size + y;
                    const size_t src_x = px * patch_size + x;
                    for (size_t c = 0; c < 3; c++) {
                        // CHW source: buf[c * ny * nx + src_y * nx + src_x]
                        // HWC within patch: dst[y * patch_size * 3 + x * 3 + c]
                        dst[y * patch_size * 3 + x * 3 + c] = float_image.buf[c * ny * nx + src_y * nx + src_x];
                    }
                }
            }
        }
    }
}

}  // namespace

EncodedImage VisionEncoderGemma4::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    // 1. Convert input tensor (NHWC uint8) to clip_image_u8 (HWC uint8)
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    // 2. Compute aspect-ratio-preserving target size
    const size_t max_patches = config.max_soft_tokens * config.pooling_kernel_size * config.pooling_kernel_size;
    const auto [target_height, target_width] = get_aspect_ratio_preserving_size(static_cast<size_t>(input_image.ny),
                                                                                static_cast<size_t>(input_image.nx),
                                                                                config.patch_size,
                                                                                max_patches,
                                                                                config.pooling_kernel_size);

    // 3. Bicubic resize
    clip_image_u8 resized_image;
    bicubic_resize(input_image, resized_image, static_cast<int>(target_width), static_cast<int>(target_height));

    // 4. Rescale to [0,1] and convert to CHW float
    // With mean=[0,0,0] and std=[1,1,1], clip_image_preprocess produces pixel/255.0
    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
    clip_image_f32 float_image = clip_image_preprocess(ctx, resized_image);

    // 5. Extract patches: (num_patches, patch_size*patch_size*3)
    const size_t num_patches_h = target_height / config.patch_size;
    const size_t num_patches_w = target_width / config.patch_size;
    const size_t num_patches = num_patches_h * num_patches_w;
    const size_t patch_dim = config.patch_size * config.patch_size * 3;

    // Create padded pixel_values tensor [1, max_patches, patch_dim]
    ov::Tensor pixel_values(ov::element::f32, {1, max_patches, patch_dim});
    float* pv_data = pixel_values.data<float>();
    std::fill(pv_data, pv_data + max_patches * patch_dim, 0.0f);

    extract_patches(float_image, config.patch_size, pv_data, num_patches_h, num_patches_w);

    // 6. Compute 2D position IDs: meshgrid(arange(patch_w), arange(patch_h), indexing="xy")
    // Then pad to max_patches with -1
    ov::Tensor image_position_ids(ov::element::i64, {1, max_patches, 2});
    int64_t* pos_data = image_position_ids.data<int64_t>();
    std::fill(pos_data, pos_data + max_patches * 2, int64_t{-1});

    for (size_t py = 0; py < num_patches_h; py++) {
        for (size_t px = 0; px < num_patches_w; px++) {
            const size_t patch_idx = py * num_patches_w + px;
            pos_data[patch_idx * 2 + 0] = static_cast<int64_t>(px);  // x coordinate
            pos_data[patch_idx * 2 + 1] = static_cast<int64_t>(py);  // y coordinate
        }
    }

    // 7. Run vision encoder
    encoder.set_tensor("pixel_values", pixel_values);
    encoder.set_tensor("image_position_ids", image_position_ids);
    encoder.infer();

    // 8. Output shape is [num_soft_tokens, hidden_size] → reshape to [1, num_soft_tokens, hidden_size]
    const ov::Tensor& infer_output = encoder.get_output_tensor();
    const size_t num_soft_tokens = infer_output.get_shape()[0];
    const size_t hidden_size = infer_output.get_shape()[1];

    ov::Tensor image_features(infer_output.get_element_type(), {1, num_soft_tokens, hidden_size});
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    return {std::move(image_features)};
}

InputsEmbedderGemma4::InputsEmbedderGemma4(const VLMConfig& vlm_config,
                                           const std::filesystem::path& model_dir,
                                           const std::string& device,
                                           const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, device, device_config) {
    auto per_layer_model_path = model_dir / "openvino_text_embeddings_per_layer_model.xml";

    auto compiled = utils::singleton_core().compile_model(per_layer_model_path, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled, "VLM per-layer text embeddings model");
    m_per_layer_embeddings_requests = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled.get_property(ov::optimal_number_of_infer_requests),
        [&compiled]() -> ov::InferRequest {
            return compiled.create_infer_request();
        });
}

InputsEmbedderGemma4::InputsEmbedderGemma4(const VLMConfig& vlm_config,
                                           const ModelsMap& models_map,
                                           const Tokenizer& tokenizer,
                                           const std::filesystem::path& config_dir_path,
                                           const std::string& device,
                                           const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    // Per-layer embeddings model may be in models_map
    auto it = models_map.find("text_embeddings_per_layer");
    OPENVINO_ASSERT(it != models_map.end(), "Per-layer text embeddings model not found in models map");

    const auto& [model_str, weights] = it->second;
    auto compiled = utils::singleton_core().compile_model(model_str, weights, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled, "VLM per-layer text embeddings model");
    m_per_layer_embeddings_requests = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled.get_property(ov::optimal_number_of_infer_requests),
        [&compiled]() -> ov::InferRequest {
            return compiled.create_infer_request();
        });
}

std::vector<ov::genai::EncodedImage> InputsEmbedderGemma4::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;

    ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};

    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
    }

    return embeds;
}

NormalizedPrompt InputsEmbedderGemma4::normalize_prompt(const std::string& prompt,
                                                        size_t base_id,
                                                        const std::vector<EncodedImage>& images) const {
    const auto& boi = m_vlm_config.boi_token;
    const auto& eoi = m_vlm_config.eoi_token;
    const auto& img = m_vlm_config.image_token;

    auto [unified_prompt, images_sequence] = normalize(prompt, boi, boi, base_id, images.size());

    for (size_t new_image_id : images_sequence) {
        const ov::Tensor& image_embed = images.at(new_image_id - base_id).resized_source;
        size_t num_image_tokens = image_embed.get_shape().at(1);

        std::string expanded_tag = "\n\n" + boi;
        for (size_t i = 0; i < num_image_tokens; i++) {
            expanded_tag += img;
        }
        expanded_tag += eoi + "\n\n";

        size_t pos = unified_prompt.find(boi);
        OPENVINO_ASSERT(pos != std::string::npos, "Failed to find BOI token in prompt during normalization");
        unified_prompt.replace(pos, boi.length(), expanded_tag);
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderGemma4::compute_per_layer_embeddings(const ov::Tensor& input_ids) {
    OPENVINO_ASSERT(m_per_layer_embeddings_requests, "Per-layer embeddings model is not loaded");

    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_per_layer_embeddings_requests.get());
    ov::InferRequest& req = guard.get();
    req.set_tensor("input_ids", input_ids);
    req.infer();

    const ov::Tensor& output = req.get_output_tensor();
    ov::Tensor result(output.get_element_type(), output.get_shape());
    std::memcpy(result.data(), output.data(), output.get_byte_size());
    return result;
}

// Gemma4 Jinja chat template applies "| trim" to string content, stripping the
// leading "\n\n" that normalize_prompt places before the BOI token.
// Restore the "\n\n" padding in the templated string to match the HF prompt format.
ov::Tensor InputsEmbedderGemma4::apply_chat_template_tokenize(const std::string& prompt, VLMPerfMetrics& metrics) {
    if (!m_is_chat_conversation && m_apply_chat_template) {
        // Non-chat path: apply template, then re-insert "\n\n" before BOI, then tokenize.
        bool add_special_tokens_val = m_add_special_tokens_is_set ? m_add_special_tokens : false;
        auto start_tokenizer_time = std::chrono::steady_clock::now();

        ChatHistory history({{{"role", "user"}, {"content", prompt}}});
        std::string templated_prompt = m_tokenizer.apply_chat_template(history, true);

        const auto& boi = m_vlm_config.boi_token;
        size_t boi_pos = templated_prompt.find(boi);
        if (boi_pos != std::string::npos && (boi_pos < 2 || templated_prompt.substr(boi_pos - 2, 2) != "\n\n")) {
            templated_prompt.insert(boi_pos, "\n\n");
        }

        ov::Tensor encoded =
            m_tokenizer.encode(templated_prompt, ov::genai::add_special_tokens(add_special_tokens_val)).input_ids;
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(
            PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        return encoded;
    }

    // Chat path: template already applied before get_inputs_embeds, "\n\n" may be stripped.
    // No-template path: no stripping occurs.
    // In both cases, re-insert "\n\n" before BOI if missing, then delegate to base.
    std::string adjusted_prompt = prompt;
    const auto& boi = m_vlm_config.boi_token;
    size_t boi_pos = adjusted_prompt.find(boi);
    if (boi_pos != std::string::npos && (boi_pos < 2 || adjusted_prompt.substr(boi_pos - 2, 2) != "\n\n")) {
        adjusted_prompt.insert(boi_pos, "\n\n");
    }

    return IInputsEmbedder::apply_chat_template_tokenize(adjusted_prompt, metrics);
}

ov::Tensor InputsEmbedderGemma4::get_inputs_embeds(const std::string& prompt,
                                                   const std::vector<EncodedImage>& images,
                                                   VLMPerfMetrics& metrics,
                                                   bool recalculate_merged_embeddings,
                                                   const std::vector<size_t>& images_sequence) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
    }

    ov::Tensor input_ids = get_encoded_input_ids(prompt, metrics);

    m_lm_extra_inputs["per_layer_inputs"] = compute_per_layer_embeddings(input_ids);

    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor encoded_image_token =
        m_tokenizer.encode(m_vlm_config.image_token, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] +=
        ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];

    auto merged = utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);

    return merged;
}

const std::unordered_map<std::string, ov::Tensor>& InputsEmbedderGemma4::get_lm_extra_inputs() const {
    return m_lm_extra_inputs;
}

}  // namespace ov::genai
