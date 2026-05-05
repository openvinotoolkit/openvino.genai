// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/mistral3/classes.hpp"
#include "visual_language/clip.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ov::genai {

namespace {

/// Preprocess an image for the Pixtral vision encoder.
/// 1. Compute patch-aligned target dimensions (multiples of patch_size * spatial_merge_size).
/// 2. Resize directly to the target using bicubic interpolation.
/// 3. Normalize with CLIP mean/std.
/// Returns a float32 tensor in CHW layout.
ov::Tensor preprocess_image_mistral3(const ov::Tensor& image,
                                     const ProcessorConfig& config,
                                     size_t patch_size,
                                     size_t spatial_merge_size,
                                     size_t& out_h_patches,
                                     size_t& out_w_patches) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    const int orig_w = input_image.nx;
    const int orig_h = input_image.ny;
    const int longest = static_cast<int>(config.longest_edge);

    // Downscale if the image exceeds the longest_edge constraint
    int new_h = orig_h;
    int new_w = orig_w;
    const float ratio = static_cast<float>(std::max(orig_h, orig_w)) / static_cast<float>(longest);
    if (ratio > 1.0f) {
        new_h = static_cast<int>(orig_h / ratio);
        new_w = static_cast<int>(orig_w / ratio);
    }

    // Snap dimensions up to the nearest multiple of (patch_size * spatial_merge_size)
    // so the patch grid is divisible by spatial_merge_size (required by the unfold operation).
    const int effective_patch = static_cast<int>(patch_size * spatial_merge_size);
    const int target_h = ((new_h - 1) / effective_patch + 1) * effective_patch;
    const int target_w = ((new_w - 1) / effective_patch + 1) * effective_patch;

    // Resize directly to the patch-aligned target (no zero-padding)
    clip_image_u8 resized_image;
    bicubic_resize(input_image, resized_image, target_w, target_h);

    out_h_patches = static_cast<size_t>(target_h) / patch_size;
    out_w_patches = static_cast<size_t>(target_w) / patch_size;

    // Normalize with CLIP mean/std and convert to CHW
    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    clip_image_f32 normalized = clip_image_preprocess(ctx, resized_image);
    return clip_image_f32_to_tensor(normalized);
}

}  // namespace

EncodedImage VisionEncoderMistral3::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    const size_t patch_size = config.patch_size;
    const size_t spatial_merge_size = config_map.count("spatial_merge_size")
        ? config_map.at("spatial_merge_size").as<size_t>() : 2;
    size_t h_patches = 0;
    size_t w_patches = 0;
    ov::Tensor pixel_values = preprocess_image_mistral3(image, config, patch_size, spatial_merge_size, h_patches, w_patches);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    ImageSize grid_size{h_patches, w_patches};

    return {std::move(image_features), grid_size};
}

InputsEmbedderMistral3::InputsEmbedderMistral3(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, device, device_config) {
    auto compiled_model = utils::singleton_core().compile_model(
        model_dir / "openvino_multi_modal_projector_model.xml", device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM multi-modal projector model");
    m_ireq_queue_multi_modal_projector = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

InputsEmbedderMistral3::InputsEmbedderMistral3(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    const auto& projector_model = utils::get_model_weights_pair(models_map, "multi_modal_projector").first;
    const auto& projector_weights = utils::get_model_weights_pair(models_map, "multi_modal_projector").second;
    auto compiled_model = utils::singleton_core().compile_model(
        projector_model, projector_weights, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM multi-modal projector model");
    m_ireq_queue_multi_modal_projector = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

ov::Tensor InputsEmbedderMistral3::spatial_merge(const ov::Tensor& features,
                                                  size_t h_patches,
                                                  size_t w_patches) const {
    // Implements torch.nn.functional.unfold(kernel_size=spatial_merge_size, stride=spatial_merge_size)
    // Input features: [num_patches, hidden_size] where num_patches = h_patches * w_patches
    // Output: [h_merged * w_merged, hidden_size * spatial_merge_size^2]
    const size_t sms = m_vlm_config.spatial_merge_size;
    const auto& shape = features.get_shape();
    OPENVINO_ASSERT(shape.size() == 2, "Expected 2D tensor for spatial_merge, got ", shape.size(), "D");
    const size_t hidden_size = shape[1];
    OPENVINO_ASSERT(shape[0] == h_patches * w_patches,
                    "Patch count mismatch: ", shape[0], " vs ", h_patches, " * ", w_patches);
    OPENVINO_ASSERT(h_patches % sms == 0 && w_patches % sms == 0,
                    "Patch grid (", h_patches, ", ", w_patches, ") not divisible by spatial_merge_size ", sms);

    const size_t h_merged = h_patches / sms;
    const size_t w_merged = w_patches / sms;
    const size_t merged_hidden = hidden_size * sms * sms;

    ov::Tensor merged(features.get_element_type(), {h_merged * w_merged, merged_hidden});
    const float* src = features.data<const float>();
    float* dst = merged.data<float>();

    // For each merged spatial position, collect sms x sms patches and interleave features
    // to match PyTorch unfold layout: output[c * sms^2 + kh * sms + kw] = features[patch, c]
    const size_t kernel_area = sms * sms;
    for (size_t mh = 0; mh < h_merged; ++mh) {
        for (size_t mw = 0; mw < w_merged; ++mw) {
            const size_t out_idx = mh * w_merged + mw;
            float* out_ptr = dst + out_idx * merged_hidden;
            for (size_t dh = 0; dh < sms; ++dh) {
                for (size_t dw = 0; dw < sms; ++dw) {
                    const size_t src_h = mh * sms + dh;
                    const size_t src_w = mw * sms + dw;
                    const size_t src_idx = src_h * w_patches + src_w;
                    const float* patch_features = src + src_idx * hidden_size;
                    const size_t kernel_pos = dh * sms + dw;
                    for (size_t c = 0; c < hidden_size; ++c) {
                        out_ptr[c * kernel_area + kernel_pos] = patch_features[c];
                    }
                }
            }
        }
    }

    return merged;
}

std::vector<ov::genai::EncodedImage> InputsEmbedderMistral3::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> encoded_images;
    ov::AnyMap vision_config = {
        {"patch_size", m_vlm_config.vision_config_patch_size},
        {"spatial_merge_size", m_vlm_config.spatial_merge_size}
    };
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    encoded_images.reserve(single_images.size());

    for (const ov::Tensor& image : single_images) {
        // Vision encoder produces [num_patches, hidden_size] and grid dims
        EncodedImage enc = m_vision_encoder->encode(image, vision_config);

        const size_t h_patches = enc.resized_source_size.height;
        const size_t w_patches = enc.resized_source_size.width;

        // Squeeze batch dim if present: [1, N, D] -> [N, D]
        ov::Tensor features = enc.resized_source;
        if (features.get_shape().size() == 3 && features.get_shape()[0] == 1) {
            features.set_shape({features.get_shape()[1], features.get_shape()[2]});
        }

        // Spatial merge: [h*w, D] -> [h/sms * w/sms, D * sms^2]
        ov::Tensor merged = spatial_merge(features, h_patches, w_patches);

        // Multi-modal projector: [N_merged, D * sms^2] -> [N_merged, text_hidden_size]
        CircularBufferQueueElementGuard<ov::InferRequest> projector_guard(m_ireq_queue_multi_modal_projector.get());
        ov::InferRequest& projector = projector_guard.get();
        projector.set_tensor("image_features", merged);
        projector.infer();

        const ov::Tensor& proj_output = projector.get_output_tensor();
        ov::Tensor projected(proj_output.get_element_type(), proj_output.get_shape());
        std::memcpy(projected.data(), proj_output.data(), proj_output.get_byte_size());

        const size_t sms = m_vlm_config.spatial_merge_size;
        const size_t h_merged = h_patches / sms;
        const size_t w_merged = w_patches / sms;

        EncodedImage result;
        result.resized_source = std::move(projected);
        result.resized_source_size = {h_merged, w_merged};
        result.num_image_tokens = h_merged * w_merged;
        encoded_images.push_back(std::move(result));
    }

    return encoded_images;
}

NormalizedPrompt InputsEmbedderMistral3::normalize_prompt(
    const std::string& prompt,
    size_t base_id,
    const std::vector<EncodedImage>& images) const {
    // Pixtral uses [IMG] as image token. The image token is repeated for each merged patch,
    // with [IMG_BREAK] between rows and [IMG_END] at the end.
    // Pattern: [IMG]*w_merged [IMG_BREAK] [IMG]*w_merged [IMG_BREAK] ... [IMG]*w_merged [IMG_END]
    const std::string& img_token = m_vlm_config.img_token;
    const std::string& img_break = m_vlm_config.img_break;
    const std::string& img_end = m_vlm_config.img_end;

    auto [unified_prompt, images_sequence] = normalize(prompt, img_token, img_token, base_id, images.size());

    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const auto& enc_image = images.at(new_image_id - base_id);
        const size_t h_merged = enc_image.resized_source_size.height;
        const size_t w_merged = enc_image.resized_source_size.width;

        // Build expanded token sequence
        std::string expanded_tag;
        for (size_t row = 0; row < h_merged; ++row) {
            for (size_t col = 0; col < w_merged; ++col) {
                expanded_tag += img_token;
            }
            if (row < h_merged - 1) {
                expanded_tag += img_break;
            }
        }
        expanded_tag += img_end;

        OPENVINO_ASSERT(searched_pos < unified_prompt.length());
        searched_pos = unified_prompt.find(img_token, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos,
                        "Image token not found in prompt for image ", new_image_id);
        unified_prompt.replace(searched_pos, img_token.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderMistral3::get_inputs_embeds(
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

    const int64_t image_token_id = static_cast<int64_t>(m_vlm_config.image_token_index);
    return merge_image_embeddings(input_ids, text_embeds, image_embeds, image_token_id);
}

ov::Tensor InputsEmbedderMistral3::merge_image_embeddings(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds,
    const std::vector<ov::Tensor>& image_embeds,
    int64_t image_token_id) const {
    // Mistral3 uses [IMG]*w [IMG_BREAK] ... [IMG]*w [IMG_END] layout,
    // so image tokens are NOT contiguous. Use masked scatter (matching HF's
    // masked_scatter) instead of merge_text_and_image_embeddings_llava which
    // assumes one contiguous block per image.
    const auto text_embeds_shape = text_embeds.get_shape();
    const size_t seq_len = text_embeds_shape[1];
    const size_t hidden_size = text_embeds_shape[2];

    ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());

    const int64_t* ids = input_ids.data<const int64_t>();
    float* dst = inputs_embeds.data<float>();

    size_t img_idx = 0;       // index into current image_embed
    size_t embed_idx = 0;     // which image_embed we're consuming
    const float* src = image_embeds[0].data<const float>();
    size_t embed_len = image_embeds[0].get_shape().at(0);

    for (size_t pos = 0; pos < seq_len; ++pos) {
        if (ids[pos] == image_token_id) {
            OPENVINO_ASSERT(embed_idx < image_embeds.size() && img_idx < embed_len,
                "More [IMG] tokens in input than image embeddings available");
            std::memcpy(dst + pos * hidden_size,
                        src + img_idx * hidden_size,
                        hidden_size * sizeof(float));
            ++img_idx;
            if (img_idx >= embed_len && embed_idx + 1 < image_embeds.size()) {
                ++embed_idx;
                img_idx = 0;
                src = image_embeds[embed_idx].data<const float>();
                embed_len = image_embeds[embed_idx].get_shape().at(0);
            }
        }
    }

    return inputs_embeds;
}

}  // namespace ov::genai
