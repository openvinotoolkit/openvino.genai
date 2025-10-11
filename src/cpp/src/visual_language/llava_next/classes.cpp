
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/llava_next/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {

// forward declaration
clip_image_f32 preprocess_clip_image_llava(const clip_image_u8& image, const ProcessorConfig& config);

ov::Tensor VisionEncoderLLaVANext::get_pixel_values_llava_next(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    std::pair<int, int> size{config.size_shortest_edge, config.size_shortest_edge};
    auto patch_size = config.crop_size_height;
    auto image_patches = get_image_patches(input_image, config.image_grid_pinpoints, size, patch_size);

    // Preprocess image patches
    std::vector<clip_image_f32> processed_patches;
    processed_patches.reserve(image_patches.size());

    for (const auto& patch : image_patches) {
        processed_patches.push_back(preprocess_clip_image_llava(patch, config));
    }

    size_t num_patches = processed_patches.size();
    size_t channels = 3;
    size_t height = processed_patches[0].ny;
    size_t width = processed_patches[0].nx;

    ov::Tensor concatenated_tensor(ov::element::f32, {num_patches, channels, height, width});
    float* tensor_data = concatenated_tensor.data<float>();

    // Fill the tensor with the preprocessed patches data (each patch layout is [C * H * W])
    for (size_t i = 0; i < num_patches; ++i) {
        const auto& img = processed_patches[i];
        std::copy(img.buf.begin(), img.buf.end(), tensor_data + i * channels * height * width);
    }

    return concatenated_tensor;
}

EncodedImage VisionEncoderLLaVANext::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values = get_pixel_values_llava_next(image, config);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    ImageSize resized_source_size{config.crop_size_height / config.patch_size, config.crop_size_width / config.patch_size};

    // Gen number of patches
    ImageSize original_image_size{image.get_shape().at(1), image.get_shape().at(2)};
    auto best_resolution = select_best_resolution({original_image_size.width, original_image_size.height}, config.image_grid_pinpoints);
    int num_patches_w = best_resolution.first / config.size_shortest_edge;
    int num_patches_h = best_resolution.second / config.size_shortest_edge;

    EncodedImage encoded_image;
    encoded_image.resized_source = std::move(image_features);
    encoded_image.resized_source_size = resized_source_size;
    encoded_image.patches_grid = {num_patches_h, num_patches_w};
    encoded_image.original_image_size = original_image_size;
    return encoded_image;
}

namespace {

ov::Tensor unpad_image(const ov::Tensor& tensor, const ImageSize& original_size) {
    size_t original_height = original_size.height;
    size_t original_width = original_size.width;
    auto shape = tensor.get_shape();
    size_t embed_dim = shape[0];
    size_t current_height = shape[1];
    size_t current_width = shape[2];

    float original_aspect_ratio = static_cast<float>(original_width) / original_height;
    float current_aspect_ratio = static_cast<float>(current_width) / current_height;

    ov::Tensor unpadded_tensor;

    if (original_aspect_ratio > current_aspect_ratio) {
        float scale_factor = static_cast<float>(current_width) / original_width;
        size_t new_height = static_cast<size_t>(original_height * scale_factor);
        size_t padding = (current_height - new_height) / 2;
        size_t unpadded_height_dim = new_height + 1;
        unpadded_height_dim = std::min(unpadded_height_dim, current_height);
        unpadded_tensor = ov::Tensor(tensor.get_element_type(), {embed_dim, unpadded_height_dim, current_width});

        for (size_t e = 0; e < embed_dim; ++e) {
            for (int h = 0; h < unpadded_height_dim; ++h) {
                std::copy(
                    tensor.data<float>() + (e * current_height * current_width + (padding + h) * current_width),
                    tensor.data<float>() + (e * current_height * current_width + (padding + h) * current_width + current_width),
                    unpadded_tensor.data<float>() + (e * unpadded_height_dim * current_width + h * current_width)
                );
            }
        }
    } else {
        float scale_factor = static_cast<float>(current_height) / original_height;
        size_t new_width = static_cast<size_t>(original_width * scale_factor);
        size_t padding = (current_width - new_width) / 2;
        size_t unpadded_width_dim = new_width + 1;
        unpadded_width_dim = std::min(unpadded_width_dim, current_width);
        unpadded_tensor = ov::Tensor(tensor.get_element_type(), {embed_dim, current_height, unpadded_width_dim});

        for (size_t e = 0; e < embed_dim; ++e) {
            for (int h = 0; h < current_height; ++h) {
                std::copy(
                    tensor.data<float>() + (e * current_height * current_width + h * current_width + padding),
                    tensor.data<float>() + (e * current_height * current_width + h * current_width + padding + unpadded_width_dim),
                    unpadded_tensor.data<float>() + (e * current_height * unpadded_width_dim + h * unpadded_width_dim)
                );
            }
        }
    }

    return unpadded_tensor;
}

ov::Tensor reshape_and_rearrange_image_feature(
    const ov::Tensor& image_feature,
    int num_patch_height,
    int num_patch_width,
    int height,
    int width) {
    auto shape = image_feature.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "image_feature tensor must have 3 dimensions");

    size_t num_patches = shape[0];
    size_t patch_seq_len = shape[1];
    size_t embed_dim = shape[2];

    OPENVINO_ASSERT(
        num_patches == num_patch_height * num_patch_width,
        "Number of patches does not match the specified grid size"
    );

    OPENVINO_ASSERT(
        patch_seq_len == height * width,
        "Patch sequence length does not match the specified height and width"
    );

    // Reshape tensor data and permute dimensions
    // [num_patches, patch_seq_len, embed_dim] -> [embed_dim, num_patch_height, height, num_patch_width, width]
    std::vector<float> reshaped_data(num_patches * patch_seq_len * embed_dim);
    const float* image_feature_data = image_feature.data<float>();

    for (int p = 0; p < num_patches; ++p) {
        for (int i = 0; i < patch_seq_len; ++i) {
            for (int e = 0; e < embed_dim; ++e) {
                int h = i / width;
                int w = i % width;
                int ph = p / num_patch_width;
                int pw = p % num_patch_width;
                reshaped_data[((((e * num_patch_height + ph) * height + h) * num_patch_width + pw) * width + w)] =
                    image_feature_data[(p * patch_seq_len + i) * embed_dim + e];
            }
        }
    }

    ov::Tensor result(image_feature.get_element_type(),
                        {static_cast<size_t>(embed_dim),
                        static_cast<size_t>(num_patch_height * height),
                        static_cast<size_t>(num_patch_width * width)}
    );
    std::copy(reshaped_data.begin(), reshaped_data.end(), result.data<float>());
    return result;
}

/**
* @brief Flattens and transposes tensor.
* Used for packing image features of llava_next models.
*
* @param tensor A tensor with a shape (embed_dim, height, width)
* @return A tensor with a shape (height * width, embed_dim)
*/
ov::Tensor flatten_and_transpose(const ov::Tensor& tensor) {
    auto shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "Flattening tensor must have 3 dimensions");
    const float* data = tensor.data<float>();
    size_t embed_dim = shape[0];
    size_t height = shape[1];
    size_t width = shape[2];
    size_t flatten_dim = height * width;

    ov::Tensor flatten_feature(tensor.get_element_type(), {flatten_dim, embed_dim});
    float* flatten_feature_data = flatten_feature.data<float>();

    for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
            for (size_t e = 0; e < embed_dim; ++e) {
                flatten_feature_data[(h * width + w) * embed_dim + e] = data[e * flatten_dim + h * width + w];
            }
        }
    }

    return flatten_feature;
}

/**
 * @brief Adds image newline tensor to patches image feature tensor.
 * Used for packing image features of llava_next models.
 *
 * @param image_feature A tensor with a shape (embed_dim, height, width)
 * @param image_newline A tensor with a shape (embed_dim)
 * @return A tensor with a shape (embed_dim, height, width + 1)
 */
ov::Tensor add_image_newline(const ov::Tensor& image_feature, const ov::Tensor& image_newline) {
    auto shape = image_feature.get_shape();

    OPENVINO_ASSERT(shape.size() == 3, "Input image_feature must have 3 dimensions");

    size_t embed_dim = shape[0];
    size_t height = shape[1];
    size_t width = shape[2];

    OPENVINO_ASSERT(image_newline.get_shape()[0] == embed_dim, "image_newline dimension must match embed_dim of image_feature");

    const float* image_feature_data = image_feature.data<float>();
    const float* newline_data = image_newline.data<float>();

    ov::Tensor feature_with_newline{image_feature.get_element_type(), {embed_dim, height, width + 1}};
    float* feature_with_newline_data = feature_with_newline.data<float>();

    for (size_t e = 0; e < embed_dim; ++e) {
        for (size_t h = 0; h < height; ++h) {
            // Copy original image feature data
            std::copy(
                image_feature_data + (e * height * width + h * width),
                image_feature_data + (e * height * width + (h + 1) * width),
                feature_with_newline_data + (e * height * (width + 1) + h * (width + 1))
            );
            // Add image newline
            feature_with_newline_data[e * height * (width + 1) + h * (width + 1) + width] = newline_data[e];
        }
    }

    return feature_with_newline;
}
} // namespace

/**
 * @brief Processes base and patches image features extracted from encoded image.
 * Used in getting inputs embeds for llava_next models.
 *
 * @param encoded_image An encoded image retrieved from vision encoder
 * @param original_image_size A size of the original image
 * @param image_newline An image newline tensor with a shape (embed_dim)
 * @return A tensor with a shape (1, new_seq_len, embed_dim)
 */
ov::Tensor InputsEmbedderLLaVANext::pack_image_features_llava_next(
    const EncodedImage& encoded_image,
    const ov::Tensor& image_newline) const {
    auto image_feature = encoded_image.resized_source;
    auto image_feature_shape = image_feature.get_shape();
    size_t num_patches = image_feature_shape[0];
    size_t patch_seq_len = image_feature_shape[1];
    size_t embed_dim = image_feature_shape[2];

    const float* image_feature_data = image_feature.data<float>();
    const float* newline_data = image_newline.data<float>();

    if (num_patches > 1) {
        // Extract base image feature (first patch)
        ov::Tensor base_image_feature(image_feature.get_element_type(), {1, patch_seq_len, embed_dim});
        const float* src_data = image_feature.data<float>();
        float* dst_data = base_image_feature.data<float>();
        std::copy(src_data, src_data + patch_seq_len * embed_dim, dst_data);

        // Extract other grid patches
        ov::Tensor patches_image_feature(image_feature.get_element_type(), {num_patches - 1, patch_seq_len, embed_dim});
        dst_data = patches_image_feature.data<float>();
        std::copy(src_data + patch_seq_len * embed_dim,
                src_data + num_patches * patch_seq_len * embed_dim,
                dst_data);

        // Process grid patches image feature
        size_t height = encoded_image.resized_source_size.height;
        size_t width = encoded_image.resized_source_size.width;
        size_t num_patch_height = encoded_image.patches_grid.first;
        size_t num_patch_width = encoded_image.patches_grid.second;

        ov::Tensor reshaped_image_feature = reshape_and_rearrange_image_feature(patches_image_feature, num_patch_height, num_patch_width, height, width);

        ov::Tensor unpadded_image_feature = unpad_image(reshaped_image_feature, encoded_image.original_image_size);

        ov::Tensor image_feature_with_newline = add_image_newline(unpadded_image_feature, image_newline);

        ov::Tensor processed_image_feature = flatten_and_transpose(image_feature_with_newline);

        // Concatenate base image feature ([1, seq_len_1, emded_dim]) and patches image feature ([seq_len_2, embed_dim])
        auto base_shape = base_image_feature.get_shape();
        auto processed_shape = processed_image_feature.get_shape();

        const float* base_data = base_image_feature.data<float>();
        const float* processed_data = processed_image_feature.data<float>();

        ov::Tensor result(image_feature.get_element_type(), {1, base_shape[1] + processed_shape[0], embed_dim});
        // Copy base image feature data
        std::copy(base_data, base_data + base_shape[1] * embed_dim, result.data<float>());
        // Copy processed image feature data
        std::copy(processed_data,
                processed_data + processed_shape[0] * embed_dim,
                result.data<float>() + base_shape[1] * embed_dim);
        return result;
    } else {
        // If there is only one patch, return the original (base) image feature concatenated with image_newline
        ov::Tensor result(image_feature.get_element_type(), {1, patch_seq_len + 1, embed_dim});
        // Copy base image feature data
        std::copy(image_feature_data + embed_dim,
                image_feature_data + patch_seq_len * embed_dim,
                result.data<float>());
        // Append image_newline data
        std::copy(newline_data,
                newline_data + embed_dim,
                result.data<float>() + patch_seq_len * embed_dim);
        return result;
    }
}

std::vector<ov::genai::EncodedImage> InputsEmbedderLLaVANext::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
    }
    return embeds;
}

NormlizedPrompt InputsEmbedderLLaVANext::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    std::string image_token = m_vlm_config.im_start;
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    ov::Tensor image_newline;
    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& encoded_image = images.at(new_image_id - base_id);
        if (!image_newline) {
            size_t embed_dim = encoded_image.resized_source.get_shape().at(2);
            image_newline = ov::Tensor(encoded_image.resized_source.get_element_type(), {embed_dim});
            float* image_newline_data = image_newline.data<float>();
            std::copy(m_vlm_config.image_newline.begin(), m_vlm_config.image_newline.end(), image_newline_data);
        }

        image_embeds.push_back(pack_image_features_llava_next(encoded_image, image_newline));
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
    return {std::move(unified_prompt), std::move(images_sequence)};
}

ov::Tensor InputsEmbedderLLaVANext::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    
    ov::Tensor image_newline;
    size_t searched_pos = 0;
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& encoded_image = images.at(new_image_id);
        if (!image_newline) {
            size_t embed_dim = encoded_image.resized_source.get_shape().at(2);
            image_newline = ov::Tensor(encoded_image.resized_source.get_element_type(), {embed_dim});
            float* image_newline_data = image_newline.data<float>();
            std::copy(m_vlm_config.image_newline.begin(), m_vlm_config.image_newline.end(), image_newline_data);
        }

        image_embeds.push_back(pack_image_features_llava_next(encoded_image, image_newline));
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