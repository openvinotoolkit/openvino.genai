// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/glm/classes.hpp"

#include "visual_language/clip.hpp"
#include "utils.hpp"
#include "logger.hpp"

namespace ov::genai {

namespace {

/// @brief Resize and normalize an image for the GLM SigLIP vision encoder.
///
/// Preprocessing mirrors SiglipImageProcessor with:
///   - Bicubic resize to (image_size x image_size)
///   - Pixel normalization: divide by 255, subtract mean 0.5, divide by std 0.5
///     → equivalent to (pixel/255.0 - 0.5) / 0.5
ov::Tensor get_pixel_values_glm(const ov::Tensor& image, size_t image_size) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    // Bicubic resize to image_size x image_size
    clip_image_u8 resized_image;
    bicubic_resize(input_image, resized_image, static_cast<int>(image_size), static_cast<int>(image_size));

    // Build clip context with SigLIP normalization: mean=0.5, std=0.5
    clip_ctx ctx;
    ctx.image_size = static_cast<int>(image_size);
    ctx.image_mean[0] = 0.5f;
    ctx.image_mean[1] = 0.5f;
    ctx.image_mean[2] = 0.5f;
    ctx.image_std[0] = 0.5f;
    ctx.image_std[1] = 0.5f;
    ctx.image_std[2] = 0.5f;

    clip_image_f32 preprocessed = clip_image_preprocess(ctx, resized_image);

    // Convert to [1, C, H, W] tensor
    ov::Tensor output(ov::element::f32, {1, 3, image_size, image_size});
    float* dst = output.data<float>();
    const float* src = preprocessed.buf.data();
    std::copy_n(src, preprocessed.buf.size(), dst);
    return output;
}

/// @brief Merge vision embeddings into text embeddings at boi_token positions.
///
/// GLM inserts N consecutive copies of boi_token_id per image. This function
/// overwrites those positions with the corresponding vision encoder features.
/// All images must have the same number of vision tokens.
ov::Tensor merge_text_and_vision_embeddings_glm(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds,
    const std::vector<ov::Tensor>& image_embeds,
    int64_t boi_token_id)
{
    const auto& shape = text_embeds.get_shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];
    size_t embed_dim = shape[2];

    const int64_t* ids_data = input_ids.data<const int64_t>();
    const float* text_data = text_embeds.data<const float>();

    ov::Tensor result(text_embeds.get_element_type(), shape);
    float* out_data = result.data<float>();

    // Copy text embeddings as base
    std::copy_n(text_data, batch_size * seq_len * embed_dim, out_data);

    // For each batch entry, replace boi_token spans with vision embeddings
    size_t image_idx = 0;
    for (size_t b = 0; b < batch_size; ++b) {
        const int64_t* row_ids = ids_data + b * seq_len;
        float* row_out = out_data + b * seq_len * embed_dim;

        size_t j = 0;
        while (j < seq_len) {
            if (row_ids[j] == boi_token_id) {
                // Find the run of boi tokens
                size_t run_start = j;
                while (j < seq_len && row_ids[j] == boi_token_id) {
                    ++j;
                }
                size_t run_len = j - run_start;

                if (image_idx < image_embeds.size()) {
                    const ov::Tensor& img_feat = image_embeds[image_idx];
                    // img_feat shape: [1, num_tokens, embed_dim] or [num_tokens, embed_dim]
                    const float* img_data = img_feat.data<const float>();
                    size_t img_tokens = img_feat.get_size() / embed_dim;
                    size_t copy_tokens = std::min(run_len, img_tokens);

                    std::copy_n(
                        img_data,
                        copy_tokens * embed_dim,
                        row_out + run_start * embed_dim);
                    ++image_idx;
                }
            } else {
                ++j;
            }
        }
    }

    return result;
}

} // namespace

//
// VisionEncoderGLM
//

EncodedImage VisionEncoderGLM::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(
        this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    // Read image_size from config_map, fall back to 672 (GLM default)
    size_t image_size = 672;
    if (config_map.count("image_size")) {
        image_size = config_map.at("image_size").as<size_t>();
    }

    ov::Tensor pixel_values = get_pixel_values_glm(image, image_size);

    // The exported vision encoder accepts input named "image": [batch, C, H, W]
    encoder.set_tensor("image", pixel_values);
    encoder.infer();

    const ov::Tensor& output = encoder.get_output_tensor();
    ov::Tensor image_features(output.get_element_type(), output.get_shape());
    std::memcpy(image_features.data(), output.data(), output.get_byte_size());

    // patch_size=14, image_size=672 → grid = 672/14 = 48
    ImageSize grid{image_size / 14, image_size / 14};
    return {std::move(image_features), grid};
}

//
// InputsEmbedderGLM
//

InputsEmbedderGLM::InputsEmbedderGLM(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, device, device_config) {}

InputsEmbedderGLM::InputsEmbedderGLM(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

NormalizedPrompt InputsEmbedderGLM::normalize_prompt(
    const std::string& prompt,
    size_t base_id,
    const std::vector<EncodedImage>& images) const
{
    // The GLM chat template already injects N=578 copies of <|begin_of_image|>
    // per image directly into the tokenized prompt. No further expansion is needed
    // here — the prompt is used as-is.
    // We record which images are referenced (all of them, in order).
    std::vector<size_t> images_sequence;
    images_sequence.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        images_sequence.push_back(base_id + i);
    }
    return {prompt, std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderGLM::get_inputs_embeds(
    const std::string& prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence)
{
    // Collect image embeddings referenced by this prompt
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t img_id : images_sequence) {
        image_embeds.push_back(images.at(img_id).resized_source);
    }

    // Tokenize the prompt
    ov::Tensor input_ids = get_encoded_input_ids(prompt, metrics);

    // Embed the token ids
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_guard(
        m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (image_embeds.empty()) {
        // Text-only path: return a copy of the text embeddings
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    // Image-text path: replace boi_token_id spans with vision embeddings
    int64_t boi_token_id = m_vlm_config.glm_boi_token_id;
    return merge_text_and_vision_embeddings_glm(input_ids, text_embeds, image_embeds, boi_token_id);
}

} // namespace ov::genai
