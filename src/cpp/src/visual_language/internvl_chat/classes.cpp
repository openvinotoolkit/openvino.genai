// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/internvl_chat/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

std::string NATIVE_TAG = "<image>";

std::vector<clip_image_u8> split_image_internvl(
    const clip_image_u8& image,
    int image_size,
    int min_num = 1,
    int max_num = 12,
    bool use_thumbnail = true) {
    int orig_width = image.nx;
    int orig_height = image.ny;
    float aspect_ratio = static_cast<float>(orig_width) / orig_height;

    std::vector<std::pair<int, int>> target_ratios;
    for (int n = min_num; n <= max_num; ++n) {
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (i * j <= max_num && i * j >= min_num) {
                    target_ratios.emplace_back(i, j);
                }
            }
        }
    }
    std::sort(target_ratios.begin(), target_ratios.end(),
        [](const auto& a, const auto& b) { return a.first * a.second < b.first * b.second; });

    auto find_closest_aspect_ratio = [&](float ar, const std::vector<std::pair<int, int>>& ratios) {
        float best_ratio_diff = std::numeric_limits<float>::max();
        std::pair<int, int> best_ratio = {1, 1};
        int area = orig_width * orig_height;

        for (const auto& ratio : ratios) {
            float target_ar = static_cast<float>(ratio.first) / ratio.second;
            float ratio_diff = std::abs(ar - target_ar);
            if (ratio_diff < best_ratio_diff) {
                best_ratio_diff = ratio_diff;
                best_ratio = ratio;
            } else if (ratio_diff == best_ratio_diff && area > 0.5 * image_size * image_size * ratio.first * ratio.second) {
                best_ratio = ratio;
            }
        }
        return best_ratio;
    };

    auto target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios);

    int target_width = image_size * target_aspect_ratio.first;
    int target_height = image_size * target_aspect_ratio.second;
    int blocks = target_aspect_ratio.first * target_aspect_ratio.second;

    clip_image_u8 resized_img;
    bicubic_resize(image, resized_img, target_width, target_height);

    std::vector<clip_image_u8> processed_images;
    for (int i = 0; i < blocks; ++i) {
        int x = (i % (target_width / image_size)) * image_size;
        int y = (i / (target_width / image_size)) * image_size;

        clip_image_u8 split_img;
        split_img.nx = image_size;
        split_img.ny = image_size;
        split_img.buf.resize(3 * image_size * image_size);

        for (int dy = 0; dy < image_size; ++dy) {
            for (int dx = 0; dx < image_size; ++dx) {
                for (int c = 0; c < 3; ++c) {
                    int src_idx = ((y + dy) * target_width + (x + dx)) * 3 + c;
                    int dst_idx = (dy * image_size + dx) * 3 + c;
                    split_img.buf[dst_idx] = resized_img.buf[src_idx];
                }
            }
        }

        processed_images.push_back(std::move(split_img));
    }

    if (use_thumbnail && processed_images.size() != 1) {
        clip_image_u8 thumbnail_img;
        bicubic_resize(image, thumbnail_img, image_size, image_size);
        processed_images.push_back(std::move(thumbnail_img));
    }

    return processed_images;
}

ov::Tensor get_pixel_values_internvl(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    const size_t image_size = config.size_shortest_edge;

    clip_ctx ctx;
    ctx.image_size = image_size;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    std::vector<clip_image_u8> splitted_images = split_image_internvl(input_image, image_size);

    std::vector<clip_image_f32> processed_images;
    processed_images.reserve(splitted_images.size());
    for (const auto& image : splitted_images) {
        processed_images.push_back(clip_image_preprocess(ctx, image));
    }

    size_t batch_size = processed_images.size();
    size_t channels = 3;
    size_t height = processed_images[0].ny;
    size_t width = processed_images[0].nx;

    ov::Tensor output_tensor(ov::element::f32, {batch_size, channels, height, width});
    float* output_data = output_tensor.data<float>();

    for (size_t i = 0; i < batch_size; ++i) {
        const auto& img = processed_images[i];
        std::copy(img.buf.begin(), img.buf.end(), output_data + i * channels * height * width);
    }
    return output_tensor;
}

} // namespace

EncodedImage VisionEncoderInternVLChat::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values = get_pixel_values_internvl(image, config);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    ImageSize resized_source_size{config.crop_size_height / config.patch_size, config.crop_size_width / config.patch_size};

    return {std::move(image_features), resized_source_size};
}

namespace {

ov::Tensor merge_text_and_image_embeddings_internvl(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds,
    const std::vector<ov::Tensor>& image_embeds,
    int64_t image_context_token_id) {
    auto text_embeds_shape = text_embeds.get_shape();
    size_t batch_size = text_embeds_shape.at(0);
    size_t seq_len = text_embeds_shape.at(1);
    size_t embed_dim = text_embeds_shape.at(2);

    ov::Tensor merged_embeds(text_embeds.get_element_type(), text_embeds_shape);

    const float* text_embeds_data = text_embeds.data<float>();
    const int64_t* input_ids_data = input_ids.data<int64_t>();
    float* merged_embeds_data = merged_embeds.data<float>();

    size_t flattened_size = batch_size * seq_len;
    std::vector<bool> image_context_tokens_mask(flattened_size, false);
    size_t image_context_tokens_count = 0;

    for (size_t i = 0; i < flattened_size; ++i) {
        if (input_ids_data[i] == image_context_token_id) {
            image_context_tokens_mask[i] = true;
            ++image_context_tokens_count;
        }
    }

    OPENVINO_ASSERT(image_context_tokens_count > 0, "input_ids does not contain image context token ids");

    size_t image_idx = 0;
    size_t image_context_token_idx = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            size_t flat_idx = i * seq_len + j;
            size_t offset = flat_idx * embed_dim;

            if (image_context_tokens_mask[flat_idx]) {
                const ov::Tensor& single_image_embeds = image_embeds[image_idx];
                const size_t num_all_image_tokens = single_image_embeds.get_shape().at(0) * single_image_embeds.get_shape().at(1); // num_patches * num_image_tokens
                const float* image_embeds_data = single_image_embeds.data<float>();
                std::copy_n(image_embeds_data + image_context_token_idx * embed_dim,
                            embed_dim,
                            merged_embeds_data + offset);
                
                ++image_context_token_idx;

                if (image_context_token_idx == num_all_image_tokens) {
                    ++image_idx;
                    image_context_token_idx = 0;
                }
            } else {
                std::copy_n(text_embeds_data + offset, embed_dim, merged_embeds_data + offset);
            }
        }
    }

    return merged_embeds;
}

} // namespace

InputsEmbedderInternVLChat::InputsEmbedderInternVLChat(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) { }

InputsEmbedderInternVLChat::InputsEmbedderInternVLChat(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }


NormlizedPrompt InputsEmbedderInternVLChat::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG + '\n', base_id, images.size());
    
    std::string image_start_token = m_vlm_config.image_start_token;
    std::string image_context_token = m_vlm_config.image_context_token;
    std::string image_end_token = m_vlm_config.image_end_token;
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id - base_id).resized_source);

        const size_t num_patches = image_embeds.back().get_shape().at(0);
        const size_t num_image_tokens = image_embeds.back().get_shape().at(1);
        
        std::string expanded_tag{image_start_token};
        for (size_t idx = 0; idx < num_patches * num_image_tokens; ++idx) {
            expanded_tag += image_context_token;
        }
        expanded_tag += image_end_token;
        OPENVINO_ASSERT(searched_pos < unified_prompt.length());
        searched_pos = unified_prompt.find(NATIVE_TAG, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos);
        unified_prompt.replace(searched_pos, NATIVE_TAG.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderInternVLChat::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
    }
    std::string image_context_token = m_vlm_config.image_context_token;

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
    ov::Tensor encoded_image_context_token = m_tokenizer.encode(image_context_token, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_context_token_id = encoded_image_context_token.data<int64_t>()[encoded_image_context_token.get_size() - 1];
    return merge_text_and_image_embeddings_internvl(input_ids, text_embeds, image_embeds, image_context_token_id);
}

} // namespace ov::genai
