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

const std::string NATIVE_TAG = "[IMG]";
const std::string IMG_BREAK_TAG = "[IMG_BREAK]";
const std::string IMG_END_TAG = "[IMG_END]";

/// Preprocess an image for the Pixtral vision encoder.
/// 1. Compute patch-aligned target dimensions (multiples of patch_size * spatial_merge_size).
/// 2. Resize directly to the target using bicubic interpolation.
/// 3. Normalize with CLIP mean/std.
/// Returns a float32 tensor in CHW layout together with the resulting patch grid size.
std::pair<ov::Tensor, ImageSize> preprocess_image_mistral3(const ov::Tensor& image,
                                                           const ProcessorConfig& config,
                                                           size_t patch_size,
                                                           size_t spatial_merge_size) {
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

    const ImageSize grid_size{static_cast<size_t>(target_h) / patch_size,
                              static_cast<size_t>(target_w) / patch_size};

    // Normalize with CLIP mean/std and convert to CHW
    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    clip_image_f32 normalized = clip_image_preprocess(ctx, resized_image);
    return {clip_image_f32_to_tensor(normalized), grid_size};
}

/// Apply spatial merge (unfold) to vision encoder output.
/// Groups adjacent spatial_merge_size x spatial_merge_size patches and concatenates their features.
/// Input: [num_patches, hidden_size] with grid dims grid_size.
/// Output: [h_merged * w_merged, hidden_size * spatial_merge_size^2].
ov::Tensor spatial_merge(const ov::Tensor& features, ImageSize grid_size, size_t spatial_merge_size) {
    // Implements torch.nn.functional.unfold(kernel_size=spatial_merge_size, stride=spatial_merge_size)
    const size_t h_patches = grid_size.height;
    const size_t w_patches = grid_size.width;
    const auto& shape = features.get_shape();
    OPENVINO_ASSERT(shape.size() == 2, "Expected 2D tensor for spatial_merge, got ", shape.size(), "D");
    const size_t hidden_size = shape[1];
    OPENVINO_ASSERT(shape[0] == h_patches * w_patches,
                    "Patch count mismatch: ", shape[0], " vs ", h_patches, " * ", w_patches);
    OPENVINO_ASSERT(h_patches % spatial_merge_size == 0 && w_patches % spatial_merge_size == 0,
                    "Patch grid (", h_patches, ", ", w_patches, ") not divisible by spatial_merge_size ",
                    spatial_merge_size);

    const size_t h_merged = h_patches / spatial_merge_size;
    const size_t w_merged = w_patches / spatial_merge_size;
    const size_t merged_hidden = hidden_size * spatial_merge_size * spatial_merge_size;

    ov::Tensor merged(features.get_element_type(), {h_merged * w_merged, merged_hidden});
    const float* src = features.data<const float>();
    float* dst = merged.data<float>();

    // For each merged spatial position, collect sms x sms patches and interleave features
    // to match PyTorch unfold layout: output[c * sms^2 + kh * sms + kw] = features[patch, c]
    const size_t kernel_area = spatial_merge_size * spatial_merge_size;
    for (size_t mh = 0; mh < h_merged; ++mh) {
        for (size_t mw = 0; mw < w_merged; ++mw) {
            const size_t out_idx = mh * w_merged + mw;
            float* out_ptr = dst + out_idx * merged_hidden;
            for (size_t dh = 0; dh < spatial_merge_size; ++dh) {
                for (size_t dw = 0; dw < spatial_merge_size; ++dw) {
                    const size_t src_h = mh * spatial_merge_size + dh;
                    const size_t src_w = mw * spatial_merge_size + dw;
                    const size_t src_idx = src_h * w_patches + src_w;
                    const float* patch_features = src + src_idx * hidden_size;
                    const size_t kernel_pos = dh * spatial_merge_size + dw;
                    for (size_t c = 0; c < hidden_size; ++c) {
                        out_ptr[c * kernel_area + kernel_pos] = patch_features[c];
                    }
                }
            }
        }
    }

    return merged;
}

/// Merge text and image embeddings using masked scatter.
/// Replaces every position in text_embeds where input_ids == image_token_id
/// with the next image embedding in sequence. Handles non-contiguous image tokens
/// separated by [IMG_BREAK]/[IMG_END], which merge_text_and_image_embeddings_llava can't express.
ov::Tensor merge_image_embeddings(const ov::Tensor& input_ids,
                                  const ov::Tensor& text_embeds,
                                  const std::vector<ov::Tensor>& image_embeds,
                                  int64_t image_token_id) {
    const auto text_embeds_shape = text_embeds.get_shape();
    OPENVINO_ASSERT(text_embeds_shape.size() == 3,
                    "Expected text embeddings of rank 3 [B, S, H], got rank ", text_embeds_shape.size());
    const size_t seq_len = text_embeds_shape[1];
    const size_t hidden_size = text_embeds_shape[2];

    ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());

    const int64_t* ids = input_ids.data<const int64_t>();
    float* dst = inputs_embeds.data<float>();

    size_t img_idx = 0;    // index into current image_embed
    size_t embed_idx = 0;  // which image_embed is being consumed
    const float* src = image_embeds[0].data<const float>();
    size_t embed_len = image_embeds[0].get_shape().at(0);

    for (size_t pos = 0; pos < seq_len; ++pos) {
        if (ids[pos] == image_token_id) {
            OPENVINO_ASSERT(embed_idx < image_embeds.size() && img_idx < embed_len,
                            "More [IMG] tokens in input than image embeddings available");
            std::memcpy(dst + pos * hidden_size, src + img_idx * hidden_size, hidden_size * sizeof(float));
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

}  // namespace

EncodedImage VisionEncoderMistral3::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    const ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    auto [pixel_values, grid_size] =
        preprocess_image_mistral3(image, config, config.patch_size, config.spatial_merge_size);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    return {std::move(image_features), grid_size};
}

InputsEmbedderMistral3::InputsEmbedderMistral3(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {
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

std::vector<ov::genai::EncodedImage> InputsEmbedderMistral3::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> encoded_images;
    const size_t spatial_merge_size = m_vision_encoder->get_processor_config().spatial_merge_size;
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    encoded_images.reserve(single_images.size());

    for (const ov::Tensor& image : single_images) {
        // Vision encoder produces [num_patches, hidden_size] and grid dims
        EncodedImage enc = m_vision_encoder->encode(image, {});

        const ImageSize grid_size = enc.resized_source_size;

        // Squeeze batch dim if present: [1, N, D] -> [N, D]
        ov::Tensor features = enc.resized_source;
        if (features.get_shape().size() == 3 && features.get_shape()[0] == 1) {
            features.set_shape({features.get_shape()[1], features.get_shape()[2]});
        }

        // Spatial merge: [h*w, D] -> [h/sms * w/sms, D * sms^2]
        ov::Tensor merged = spatial_merge(features, grid_size, spatial_merge_size);

        // Multi-modal projector: [N_merged, D * sms^2] -> [N_merged, text_hidden_size]
        CircularBufferQueueElementGuard<ov::InferRequest> projector_guard(m_ireq_queue_multi_modal_projector.get());
        ov::InferRequest& projector = projector_guard.get();
        projector.set_tensor("image_features", merged);
        projector.infer();

        const ov::Tensor& proj_output = projector.get_output_tensor();
        ov::Tensor projected(proj_output.get_element_type(), proj_output.get_shape());
        std::memcpy(projected.data(), proj_output.data(), proj_output.get_byte_size());

        const size_t h_merged = grid_size.height / spatial_merge_size;
        const size_t w_merged = grid_size.width / spatial_merge_size;

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
    // Pixtral repeats [IMG] for each merged patch, with [IMG_BREAK] between rows and [IMG_END] at the end.
    // Pattern: [IMG]*w_merged [IMG_BREAK] [IMG]*w_merged [IMG_BREAK] ... [IMG]*w_merged [IMG_END]
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG, base_id, images.size());

    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const auto& enc_image = images.at(new_image_id - base_id);
        const size_t h_merged = enc_image.resized_source_size.height;
        const size_t w_merged = enc_image.resized_source_size.width;

        std::string expanded_tag;
        expanded_tag.reserve(h_merged * w_merged * NATIVE_TAG.size() + h_merged * IMG_BREAK_TAG.size() +
                             IMG_END_TAG.size());
        for (size_t row = 0; row < h_merged; ++row) {
            for (size_t col = 0; col < w_merged; ++col) {
                expanded_tag += NATIVE_TAG;
            }
            if (row < h_merged - 1) {
                expanded_tag += IMG_BREAK_TAG;
            }
        }
        expanded_tag += IMG_END_TAG;

        searched_pos = unified_prompt.find(NATIVE_TAG, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos,
                        "Image token not found in prompt for image ", new_image_id);
        unified_prompt.replace(searched_pos, NATIVE_TAG.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

void InputsEmbedderMistral3::encode_image_token_id() {
    std::call_once(m_image_token_id_once_flag, [this]() {
        const ov::Tensor encoded_image_token =
            m_tokenizer.encode(NATIVE_TAG, ov::genai::add_special_tokens(false)).input_ids;
        OPENVINO_ASSERT(encoded_image_token.get_size() == 1, "Encoded image token must contain a single token");
        m_image_token_id = encoded_image_token.data<int64_t>()[0];
    });
}

ov::Tensor InputsEmbedderMistral3::get_inputs_embeds(
    const std::string& unified_prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence) {
    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);

    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = get_text_embedding(req, input_ids, metrics);

    if (images.empty() || images_sequence.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
    }

    encode_image_token_id();
    return merge_image_embeddings(input_ids, text_embeds, image_embeds, m_image_token_id);
}

}  // namespace ov::genai
