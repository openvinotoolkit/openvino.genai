// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/idefics3/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

/**
 * Ensure the dimension is divisible by patch_size
 */
int ensure_divide(int length, int patch_size) {
    return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
}

/**
 * Preprocess image for Idefics3 vision encoder.
 * Handles resizing to longest edge constraint and normalization.
 */
clip_image_f32 preprocess_idefics3_image(const clip_image_u8& image, const ProcessorConfig& config) {
    // Idefics3 expects 378x378 images to produce 27x27 = 729 patches
    // The connector reshapes from [729, 1152] to [27, 9, 3456]
    int patch_size = config.patch_size > 0 ? config.patch_size : 14;
    constexpr int target_patches_per_side = 27;
    int target_size = target_patches_per_side * patch_size;  // 27 * 14 = 378
    
    // Resize image to exactly 378x378
    clip_image_u8 resized_image;
    bicubic_resize(image, resized_image, target_size, target_size);
    
    // Normalize
    clip_ctx ctx;
    ctx.image_size = target_size;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
    
    clip_image_f32 normalized_image = clip_image_preprocess(ctx, resized_image);
    return normalized_image;
}

ov::Tensor create_patch_attention_mask(int height, int width, int num_patches_h, int num_patches_w) {
    // Create attention mask for patches
    ov::Tensor mask(ov::element::boolean, {1, num_patches_h, num_patches_w});
    bool* mask_data = mask.data<bool>();
    std::fill(mask_data, mask_data + num_patches_h * num_patches_w, true);
    return mask;
}

ov::Tensor create_patch_position_ids(int num_patches) {
    // Create position IDs for patches
    ov::Tensor position_ids(ov::element::i64, {1, num_patches});
    int64_t* pos_data = position_ids.data<int64_t>();
    for (int i = 0; i < num_patches; ++i) {
        pos_data[i] = i;
    }
    return position_ids;
}

} // namespace

EncodedImage VisionEncoderIdefics3::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);
    
    // Preprocess image
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_f32 preprocessed = preprocess_idefics3_image(input_image, config);
    ov::Tensor pixel_values = clip_image_f32_to_tensor(preprocessed);
    
    // Get dimensions for patch calculations
    size_t height = pixel_values.get_shape()[2];
    size_t width = pixel_values.get_shape()[3];
    size_t patch_size = config.patch_size > 0 ? config.patch_size : 14;  // Default patch size for Idefics3
    
    size_t num_patches_h = height / patch_size;
    size_t num_patches_w = width / patch_size;
    size_t num_patches = num_patches_h * num_patches_w;
    
    // Create attention mask and position IDs
    ov::Tensor patch_attention_mask = create_patch_attention_mask(height, width, num_patches_h, num_patches_w);
    ov::Tensor patch_position_ids = create_patch_position_ids(num_patches);
    
    // Set inputs
    encoder.set_tensor("pixel_values", pixel_values);
    encoder.set_tensor("patch_attention_mask", patch_attention_mask);
    encoder.set_tensor("patch_position_ids", patch_position_ids);
    
    encoder.infer();
    
    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());
    
    ImageSize resized_source_size{num_patches_h, num_patches_w};
    
    return {std::move(image_features), resized_source_size};
}

InputsEmbedderIdefics3::InputsEmbedderIdefics3(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) { }

InputsEmbedderIdefics3::InputsEmbedderIdefics3(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

NormalizedPrompt InputsEmbedderIdefics3::normalize_prompt(
    const std::string& prompt, 
    size_t base_id, 
    const std::vector<EncodedImage>& images) const {
    
    // Idefics3 uses <image> tokens similar to LLaVA
    std::string image_token = m_vlm_config.im_start;
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());
    
    // Expand each <image> token to match the number of tokens from vision encoder
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    size_t searched_pos = 0;
    
    for (size_t new_image_id : images_sequence) {
        const auto& encoded_image = images.at(new_image_id - base_id);
        image_embeds.push_back(encoded_image.resized_source);
        
        // Each image produces image_seq_len tokens (81 for SmolVLM)
        size_t num_tokens = encoded_image.resized_source.get_shape().at(1);
        
        std::string expanded_tag;
        for (size_t idx = 0; idx < num_tokens; ++idx) {
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

ov::Tensor InputsEmbedderIdefics3::get_inputs_embeds(
    const std::string& unified_prompt, 
    const std::vector<ov::genai::EncodedImage>& images, 
    ov::genai::VLMPerfMetrics& metrics, 
    bool recalculate_merged_embeddings, 
    const std::vector<size_t>& images_sequence) {
    
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
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += 
        ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    
    int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];
    
    return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
}

} // namespace ov::genai
