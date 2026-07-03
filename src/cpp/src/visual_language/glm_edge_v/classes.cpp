// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/glm_edge_v/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ov::genai {

namespace {

// GLM-Edge-V uses an Mllama-style image processor: resize preserving aspect
// ratio to fit a single (size_width x size_height) tile, pad the remainder with
// zeros, then rescale + normalize.
clip_image_f32 preprocess_clip_image_glm(const clip_image_u8& image, const ProcessorConfig& config) {
    int target_w = static_cast<int>(config.size_width);
    int target_h = static_cast<int>(config.size_height);

    float scale = std::min(static_cast<float>(target_w) / image.nx,
                           static_cast<float>(target_h) / image.ny);
    int new_w = std::max(1, static_cast<int>(std::round(image.nx * scale)));
    int new_h = std::max(1, static_cast<int>(std::round(image.ny * scale)));
    new_w = std::min(new_w, target_w);
    new_h = std::min(new_h, target_h);

    clip_image_u8 resized_image;
    bicubic_resize(image, resized_image, new_w, new_h);

    // Pad bottom/right with zeros to the full tile size.
    clip_image_u8 padded_image;
    padded_image.nx = target_w;
    padded_image.ny = target_h;
    padded_image.buf.assign(static_cast<size_t>(target_w) * target_h * 3, 0);
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            for (int c = 0; c < 3; ++c) {
                padded_image.buf[(static_cast<size_t>(y) * target_w + x) * 3 + c] =
                    resized_image.buf[(static_cast<size_t>(y) * new_w + x) * 3 + c];
            }
        }
    }

    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    return clip_image_preprocess(ctx, padded_image);
}

ov::Tensor get_pixel_values_glm(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_f32 preprocessed_image = preprocess_clip_image_glm(input_image, config);
    return clip_image_f32_to_tensor(preprocessed_image);
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

    // Materialize an f32 copy of the image features for the llava-style merge.
    //
    // NOTE: the GLM-Edge-V SigLIP tower IR declares its ``last_hidden_state``
    // output as BF16, but the OpenVINO CPU plugin populates that tensor with
    // F16-encoded bytes. The reference stack (OpenVINO Python bindings used by
    // optimum-intel) reads those bytes reinterpreted as F16, so we must do the
    // same here to stay numerically consistent with the reference inference.
    ov::Tensor image_features(ov::element::f32, infer_output.get_shape());
    const size_t n = infer_output.get_size();
    float* dst = image_features.data<float>();
    const ov::element::Type out_type = infer_output.get_element_type();
    if (out_type == ov::element::f32) {
        std::memcpy(dst, infer_output.data(), infer_output.get_byte_size());
    } else if (out_type == ov::element::f16 || out_type == ov::element::bf16) {
        // Reinterpret the raw 16-bit payload as F16 (see note above).
        const ov::float16* src = reinterpret_cast<const ov::float16*>(infer_output.data());
        for (size_t i = 0; i < n; ++i) {
            dst[i] = static_cast<float>(src[i]);
        }
    } else {
        OPENVINO_THROW("Unsupported GLM-Edge-V vision encoder output type: ", out_type);
    }

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
    std::string image_token = m_vlm_config.begin_of_image;
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());

    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const ov::Tensor& image_embed = images.at(new_image_id - base_id).resized_source;
        size_t num_image_tokens = image_embed.get_shape().at(1);

        std::string expanded_tag;
        expanded_tag.reserve(num_image_tokens * image_token.size());
        for (size_t idx = 0; idx < num_image_tokens; ++idx) {
            expanded_tag += image_token;
        }

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
    ov::Tensor encoded_image_token = m_tokenizer.encode(m_vlm_config.begin_of_image, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];

    return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
}

} // namespace ov::genai
