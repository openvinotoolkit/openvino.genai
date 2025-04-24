// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/llava/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {

clip_image_f32 preprocess_clip_image_llava(const clip_image_u8& image, const ProcessorConfig& config) {
    bool do_resize = true;
    bool do_center_crop = true;

    // Resize
    clip_image_u8 resized_image;
    if (do_resize) {
        int target_size = config.size_shortest_edge;
        float scale = static_cast<float>(target_size) / std::min(image.nx, image.ny);
        int new_width = static_cast<int>(image.nx * scale);
        int new_height = static_cast<int>(image.ny * scale);
        bicubic_resize(image, resized_image, new_width, new_height);
    } else {
        resized_image = image;
    }

    // Center crop
    clip_image_u8 cropped_image;
    if (do_center_crop) {
        int crop_height = config.crop_size_height;
        int crop_width = config.crop_size_width;
        int start_x = (resized_image.nx - crop_width) / 2;
        int start_y = (resized_image.ny - crop_height) / 2;

        cropped_image.nx = crop_width;
        cropped_image.ny = crop_height;
        cropped_image.buf.resize(3 * crop_width * crop_height);

        for (int y = 0; y < crop_height; ++y) {
            for (int x = 0; x < crop_width; ++x) {
                for (int c = 0; c < 3; ++c) {
                    cropped_image.buf[(y * crop_width + x) * 3 + c] =
                        resized_image.buf[((start_y + y) * resized_image.nx + (start_x + x)) * 3 + c];
                }
            }
        }
    } else {
        cropped_image = resized_image;
    }

    // Normalize
    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    clip_image_f32 normalized_image = clip_image_preprocess(ctx, cropped_image);
    return normalized_image;
}

namespace {

ov::Tensor get_pixel_values_llava(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_f32 preprocessed_image = preprocess_clip_image_llava(input_image, config);
    return clip_image_f32_to_tensor(preprocessed_image);
}

} // namespace

EncodedImage VisionEncoderLLaVA::encode( const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values = get_pixel_values_llava(image, config);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    ImageSize resized_source_size{config.crop_size_height / config.patch_size, config.crop_size_width / config.patch_size};

    return {std::move(image_features), resized_source_size};
}

InputsEmbedderLLaVA::InputsEmbedderLLaVA(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) { }

InputsEmbedderLLaVA::InputsEmbedderLLaVA(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

std::vector<ov::genai::EncodedImage> InputsEmbedderLLaVA::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
    }
    return embeds;
}

ov::Tensor InputsEmbedderLLaVA::get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings) {
    std::string image_token = m_vlm_config.im_start;
    auto [unified_prompt, images_sequence] = normalize_prompt(prompt, image_token, image_token, m_image_id, images.size());

    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(std::move(images.at(new_image_id - m_image_id).resized_source));
        std::string expanded_tag;
        for (size_t idx = 0; idx < image_embeds.back().get_shape().at(1); ++idx) {
            expanded_tag += image_token;
        }
        expanded_tag += '\n';
        OPENVINO_ASSERT(searched_pos < unified_prompt.length());
        searched_pos = unified_prompt.find(image_token, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos);
        unified_prompt.replace(searched_pos, image_token.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }
    m_image_id = images_sequence.empty() ? m_image_id : *std::max_element(images_sequence.begin(), images_sequence.end()) + 1;

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
    if (!m_is_chat_conversation) {
        m_image_id = 0;
    }
    return merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
}

ov::Tensor InputsEmbedderLLaVA::merge_text_and_image_embeddings_llava(const ov::Tensor& input_ids,
                                                                      ov::Tensor& text_embeds,
                                                                      const std::vector<ov::Tensor>& image_embeds,
                                                                      int64_t image_token_id) {
    auto text_embeds_shape = text_embeds.get_shape();
    size_t text_embeds_seq_length = text_embeds_shape[1];
    size_t hidden_size = text_embeds_shape[2];

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    int token_offset = text_embeds_seq_length - 1;
    auto text_embeds_data = text_embeds.data<float>();
    const float* text_embeds_end = text_embeds_data + text_embeds_seq_length * hidden_size;

    // Copy in reversed order because a tokenizer may truncate the input removing the preffix.
    for (auto image_embed_it = image_embeds.rbegin(); image_embed_it != image_embeds.rend(); ++image_embed_it) {
        for (; token_offset != -1; --token_offset) {
            if (input_ids_data[token_offset] == image_token_id) {
                break;
            }
        }
        if (token_offset == -1) {
            break;
        }
        int changed_token_offset = token_offset;
        for (; changed_token_offset != -1; --changed_token_offset) {
            if (input_ids_data[changed_token_offset] != image_token_id) {
                break;
            }
        }
        size_t n_tokens = std::min(image_embed_it->get_shape().at(1), size_t(token_offset - changed_token_offset));
        size_t n_floats = n_tokens * hidden_size;
        auto text_embeds_idx = text_embeds_data + (changed_token_offset + 1) * hidden_size;
        OPENVINO_ASSERT(text_embeds_idx + n_floats <= text_embeds_end);
        std::copy_n(
            image_embed_it->data<const float>() + image_embed_it->get_size() - n_floats,
            n_floats,
            text_embeds_idx
        );
        token_offset -= n_tokens + 1;
    }
    // text_embeds is bound to infer request that can be used by another thread after leaving embeddings calculation scope
    // so we need to return a copy to make sure data does not get corrupted 
    ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
    return inputs_embeds;
}

} // namespace ov::genai
