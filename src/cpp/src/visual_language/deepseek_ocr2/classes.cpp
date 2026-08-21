// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/deepseek_ocr2/classes.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <regex>

#include "openvino/core/type/float16.hpp"
#include "utils.hpp"
#include "visual_language/clip.hpp"

namespace ov::genai {

namespace {

constexpr int kBaseImageSize = 1024;
constexpr int kTileImageSize = 768;
constexpr int kMinTileBlocks = 2;
constexpr int kMaxTileBlocks = 6;
constexpr uint8_t kPadValue = 127;
constexpr float kNormMean = 0.5f;
constexpr float kNormStd = 0.5f;

const std::string NATIVE_TAG = "<image>";

std::pair<int, int> find_closest_aspect_ratio(float aspect_ratio,
                                              const std::vector<std::pair<int, int>>& target_ratios,
                                              int width,
                                              int height,
                                              int image_size) {
    float best_ratio_diff = std::numeric_limits<float>::infinity();
    std::pair<int, int> best_ratio = {1, 1};
    const int area = width * height;

    for (const auto& ratio : target_ratios) {
        const float target_aspect_ratio = static_cast<float>(ratio.first) / static_cast<float>(ratio.second);
        const float ratio_diff = std::abs(aspect_ratio - target_aspect_ratio);
        if (ratio_diff < best_ratio_diff) {
            best_ratio_diff = ratio_diff;
            best_ratio = ratio;
        } else if (ratio_diff == best_ratio_diff) {
            if (area > 0.5f * image_size * image_size * ratio.first * ratio.second) {
                best_ratio = ratio;
            }
        }
    }
    return best_ratio;
}

std::pair<std::vector<clip_image_u8>, std::pair<int, int>> dynamic_preprocess(const clip_image_u8& image,
                                                                                int image_size,
                                                                                int min_num,
                                                                                int max_num) {
    const int orig_width = image.nx;
    const int orig_height = image.ny;
    const float aspect_ratio = static_cast<float>(orig_width) / static_cast<float>(orig_height);

    std::vector<std::pair<int, int>> target_ratios;
    for (int n = min_num; n <= max_num; ++n) {
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (min_num <= i * j && i * j <= max_num) {
                    target_ratios.emplace_back(i, j);
                }
            }
        }
    }

    std::sort(target_ratios.begin(), target_ratios.end());
    target_ratios.erase(std::unique(target_ratios.begin(), target_ratios.end()), target_ratios.end());

    std::sort(target_ratios.begin(), target_ratios.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first * lhs.second < rhs.first * rhs.second;
    });

    const auto target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                                target_ratios,
                                                                orig_width,
                                                                orig_height,
                                                                image_size);

    const int target_width = image_size * target_aspect_ratio.first;
    const int target_height = image_size * target_aspect_ratio.second;
    const int blocks = target_aspect_ratio.first * target_aspect_ratio.second;

    clip_image_u8 resized_img;
    bicubic_resize(image, resized_img, target_width, target_height);

    std::vector<clip_image_u8> processed_images;
    processed_images.reserve(blocks);
    for (int i = 0; i < blocks; ++i) {
        const int x = (i % (target_width / image_size)) * image_size;
        const int y = (i / (target_width / image_size)) * image_size;

        clip_image_u8 tile_img;
        tile_img.nx = image_size;
        tile_img.ny = image_size;
        tile_img.buf.resize(3 * image_size * image_size);

        for (int dy = 0; dy < image_size; ++dy) {
            for (int dx = 0; dx < image_size; ++dx) {
                for (int c = 0; c < 3; ++c) {
                    const int src_idx = ((y + dy) * target_width + (x + dx)) * 3 + c;
                    const int dst_idx = (dy * image_size + dx) * 3 + c;
                    tile_img.buf[dst_idx] = resized_img.buf[src_idx];
                }
            }
        }

        processed_images.emplace_back(std::move(tile_img));
    }

    return {std::move(processed_images), target_aspect_ratio};
}

clip_image_f32 preprocess_image(const clip_image_u8& image) {
    const size_t width = static_cast<size_t>(image.nx);
    const size_t height = static_cast<size_t>(image.ny);

    clip_image_f32 result;
    result.nx = image.nx;
    result.ny = image.ny;
    result.buf.resize(3 * width * height);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            for (size_t channel = 0; channel < 3; ++channel) {
                const uint8_t pixel = image.buf[3 * (y * width + x) + channel];
                const size_t index = (y * width + x) + channel * width * height;
                const float value = static_cast<float>(pixel) / 255.0f;
                result.buf[index] = (value - kNormMean) / kNormStd;
            }
        }
    }

    return result;
}

clip_image_u8 resize_and_pad_image_pil_contain(const clip_image_u8& image,
                                                const std::pair<int, int>& target_resolution,
                                                uint8_t pad_value) {
    const int target_width = target_resolution.first;
    const int target_height = target_resolution.second;

    const float scale_w = static_cast<float>(target_width) / static_cast<float>(image.nx);
    const float scale_h = static_cast<float>(target_height) / static_cast<float>(image.ny);
    const float scale = std::min(scale_w, scale_h);

    const int new_width = std::max(1, static_cast<int>(std::lround(static_cast<float>(image.nx) * scale)));
    const int new_height = std::max(1, static_cast<int>(std::lround(static_cast<float>(image.ny) * scale)));

    clip_image_u8 resized_image;
    bicubic_resize(image, resized_image, new_width, new_height);

    clip_image_u8 padded_image;
    padded_image.nx = target_width;
    padded_image.ny = target_height;
    padded_image.buf.resize(3 * target_width * target_height, pad_value);

    const int pad_x = (target_width - new_width) / 2;
    const int pad_y = (target_height - new_height) / 2;

    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                const int dst_index = 3 * ((y + pad_y) * target_width + (x + pad_x)) + c;
                const int src_index = 3 * (y * new_width + x) + c;
                padded_image.buf[dst_index] = resized_image.buf[src_index];
            }
        }
    }

    return padded_image;
}

ov::Tensor to_batch_tensor(const std::vector<clip_image_u8>& images) {
    OPENVINO_ASSERT(!images.empty(), "Cannot create a batch tensor from an empty image list");

    const clip_image_f32 first_processed = preprocess_image(images.front());
    const size_t channels = 3;
    const size_t height = static_cast<size_t>(first_processed.ny);
    const size_t width = static_cast<size_t>(first_processed.nx);

    ov::Tensor batch_tensor(ov::element::f32, {images.size(), channels, height, width});
    float* batch_data = batch_tensor.data<float>();

    std::copy_n(first_processed.buf.data(), first_processed.buf.size(), batch_data);
    size_t offset = first_processed.buf.size();

    for (size_t idx = 1; idx < images.size(); ++idx) {
        const clip_image_f32 processed = preprocess_image(images[idx]);
        OPENVINO_ASSERT(static_cast<size_t>(processed.nx) == width && static_cast<size_t>(processed.ny) == height,
                        "Inconsistent tile tensor shape during DeepSeek-OCR-2 preprocessing");
        std::copy_n(processed.buf.data(), processed.buf.size(), batch_data + offset);
        offset += processed.buf.size();
    }

    return batch_tensor;
}

ov::Tensor infer_and_copy(CircularBufferQueue<ov::InferRequest>* queue, const ov::Tensor& pixel_values) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(queue);
    ov::InferRequest& infer_request = infer_request_guard.get();

    infer_request.set_tensor("pixel_values", pixel_values);
    infer_request.infer();

    const ov::Tensor& output = infer_request.get_output_tensor();
    ov::Tensor copy(output.get_element_type(), output.get_shape());
    output.copy_to(copy);
    return copy;
}

ov::Tensor build_merged_visual_features(const ov::Tensor& global_features,
                                        const std::optional<ov::Tensor>& tile_features,
                                        const std::vector<float>& view_separator) {
    OPENVINO_ASSERT(global_features.get_shape().size() == 3 && global_features.get_shape().at(0) == 1,
                    "Unexpected global feature shape for DeepSeek-OCR-2");

    auto copy_tensor_values_to_f32 = [](const ov::Tensor& src, float* dst, size_t values_count) {
        if (src.get_element_type() == ov::element::f32) {
            std::copy_n(src.data<const float>(), values_count, dst);
            return;
        }

        OPENVINO_ASSERT(src.get_element_type() == ov::element::f16,
                        "DeepSeek-OCR-2 vision features are expected to be fp16/fp32 but got ",
                        src.get_element_type().to_string());

        const ov::float16* src_data = src.data<const ov::float16>();
        for (size_t i = 0; i < values_count; ++i) {
            dst[i] = static_cast<float>(src_data[i]);
        }
    };

    const ov::Shape& global_shape = global_features.get_shape();
    const size_t hidden_size = global_shape.at(2);
    OPENVINO_ASSERT(view_separator.size() == hidden_size,
                    "DeepSeek-OCR-2 view_separator size does not match hidden size");

    size_t total_tokens = global_shape.at(1) + 1;
    if (tile_features.has_value()) {
        OPENVINO_ASSERT(tile_features->get_shape().size() == 3,
                        "Unexpected tile feature shape for DeepSeek-OCR-2");
        OPENVINO_ASSERT(tile_features->get_shape().at(2) == hidden_size,
                        "DeepSeek-OCR-2 tile feature hidden size mismatch");
        total_tokens += tile_features->get_shape().at(0) * tile_features->get_shape().at(1);
    }

    ov::Tensor merged_features(ov::element::f32, {1, total_tokens, hidden_size});
    float* merged_data = merged_features.data<float>();
    size_t token_offset = 0;

    if (tile_features.has_value()) {
        const size_t tile_tokens = tile_features->get_shape().at(0) * tile_features->get_shape().at(1);
        const size_t tile_values = tile_tokens * hidden_size;
        copy_tensor_values_to_f32(*tile_features, merged_data, tile_values);
        token_offset += tile_tokens;
    }

    const size_t global_values = global_shape.at(1) * hidden_size;
    copy_tensor_values_to_f32(global_features, merged_data + token_offset * hidden_size, global_values);
    token_offset += global_shape.at(1);

    std::copy_n(view_separator.data(), hidden_size, merged_data + token_offset * hidden_size);

    return merged_features;
}

}  // namespace

VisionEncoderDeepseekOCR2::VisionEncoderDeepseekOCR2(const std::filesystem::path& model_dir,
                                                     const std::string& device,
                                                     const ov::AnyMap properties) {
    auto compiled_global = utils::singleton_core().compile_model(
        model_dir / "openvino_vision_embeddings_model.xml",
        device,
        utils::get_model_properties(properties, "vision_embeddings", device));

    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_global.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_global]() -> ov::InferRequest {
            return compiled_global.create_infer_request();
        });

    auto compiled_tiles = utils::singleton_core().compile_model(
        model_dir / "openvino_vision_embeddings_tiles_model.xml",
        device,
        utils::get_model_properties(properties, "vision_embeddings_tiles", device));

    m_ireq_queue_vision_encoder_tiles = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_tiles.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_tiles]() -> ov::InferRequest {
            return compiled_tiles.create_infer_request();
        });

    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
}

VisionEncoderDeepseekOCR2::VisionEncoderDeepseekOCR2(const ModelsMap& models_map,
                                                     const std::filesystem::path& config_dir_path,
                                                     const std::string& device,
                                                     const ov::AnyMap properties) {
    const auto& [global_model, global_weights] = utils::get_model_weights_pair(models_map, "vision_embeddings");
    auto compiled_global = utils::singleton_core().compile_model(
        global_model,
        global_weights,
        device,
        utils::get_model_properties(properties, "vision_embeddings", device));

    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_global.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_global]() -> ov::InferRequest {
            return compiled_global.create_infer_request();
        });

    const auto& [tiles_model, tiles_weights] = utils::get_model_weights_pair(models_map, "vision_embeddings_tiles");
    auto compiled_tiles = utils::singleton_core().compile_model(
        tiles_model,
        tiles_weights,
        device,
        utils::get_model_properties(properties, "vision_embeddings_tiles", device));

    m_ireq_queue_vision_encoder_tiles = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_tiles.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_tiles]() -> ov::InferRequest {
            return compiled_tiles.create_infer_request();
        });

    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
}

EncodedImage VisionEncoderDeepseekOCR2::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    (void)config_map;
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    std::vector<clip_image_u8> tile_images;
    if (input_image.nx > kTileImageSize || input_image.ny > kTileImageSize) {
        auto [tiles, tile_ratio] = dynamic_preprocess(input_image, kTileImageSize, kMinTileBlocks, kMaxTileBlocks);
        if (tile_ratio.first > 1 || tile_ratio.second > 1) {
            tile_images = std::move(tiles);
        }
    }

    const clip_image_u8 global_view = resize_and_pad_image_pil_contain(input_image, {kBaseImageSize, kBaseImageSize}, kPadValue);
    const ov::Tensor global_tensor = to_batch_tensor({global_view});
    const ov::Tensor global_features = infer_and_copy(m_ireq_queue_vision_encoder.get(), global_tensor);

    std::optional<ov::Tensor> tile_features;
    if (!tile_images.empty()) {
        const ov::Tensor tile_tensor = to_batch_tensor(tile_images);
        tile_features = infer_and_copy(m_ireq_queue_vision_encoder_tiles.get(), tile_tensor);
    }

    ov::Tensor merged_features = build_merged_visual_features(global_features, tile_features, m_vlm_config.view_separator);

    EncodedImage encoded_image;
    encoded_image.resized_source = std::move(merged_features);
    encoded_image.num_image_tokens = encoded_image.resized_source.get_shape().at(1);
    encoded_image.resized_source_size = ImageSize{static_cast<size_t>(kBaseImageSize), static_cast<size_t>(kBaseImageSize)};
    encoded_image.original_image_size = ImageSize{static_cast<size_t>(input_image.ny), static_cast<size_t>(input_image.nx)};
    return encoded_image;
}

InputsEmbedderDeepseekOCR2::InputsEmbedderDeepseekOCR2(const VLMConfig& vlm_config,
                                                       const std::filesystem::path& model_dir,
                                                       const Tokenizer& tokenizer,
                                                       const std::string& device,
                                                       const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {
    if (m_vlm_config.image_token_id >= 0) {
        m_image_token_id = m_vlm_config.image_token_id;
    } else {
        ov::Tensor encoded_image_token = m_tokenizer.encode(NATIVE_TAG, ov::genai::add_special_tokens(false)).input_ids;
        m_image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];
    }
}

InputsEmbedderDeepseekOCR2::InputsEmbedderDeepseekOCR2(const VLMConfig& vlm_config,
                                                       const ModelsMap& models_map,
                                                       const Tokenizer& tokenizer,
                                                       const std::filesystem::path& config_dir_path,
                                                       const std::string& device,
                                                       const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    if (m_vlm_config.image_token_id >= 0) {
        m_image_token_id = m_vlm_config.image_token_id;
    } else {
        ov::Tensor encoded_image_token = m_tokenizer.encode(NATIVE_TAG, ov::genai::add_special_tokens(false)).input_ids;
        m_image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];
    }
}

ov::Tensor InputsEmbedderDeepseekOCR2::apply_chat_template_tokenize(const std::string& prompt,
                                                                    ov::genai::VLMPerfMetrics& metrics) {
    // Skip templating unless the user has explicitly set a custom chat template on the tokenizer.
    const bool saved = m_apply_chat_template;
    m_apply_chat_template = saved && !m_tokenizer.get_chat_template().empty();
    struct Guard { bool& flag; bool value; ~Guard() { flag = value; } } guard{m_apply_chat_template, saved};
    return InputsEmbedder::IInputsEmbedder::apply_chat_template_tokenize(prompt, metrics);
}

NormalizedPrompt InputsEmbedderDeepseekOCR2::normalize_prompt(const std::string& prompt,
                                                              size_t base_id,
                                                              const std::vector<EncodedImage>& images) const {
    std::string prompt_with_tag = prompt;
    if (!images.empty() &&
        prompt.find(NATIVE_TAG) == std::string::npos &&
        !std::regex_search(prompt, UNIVERSAL_IMAGE_PATTERN)) {
        prompt_with_tag = NATIVE_TAG + "\n" + prompt;
    }

    auto [unified_prompt, images_sequence] = normalize(prompt_with_tag, NATIVE_TAG, NATIVE_TAG, base_id, images.size());

    size_t search_pos = 0;
    for (size_t image_id : images_sequence) {
        const EncodedImage& encoded_image = images.at(image_id - base_id);
        const size_t num_image_tokens = encoded_image.num_image_tokens > 0 ?
            encoded_image.num_image_tokens :
            encoded_image.resized_source.get_shape().at(1);

        std::string expanded_tag;
        expanded_tag.reserve(num_image_tokens * NATIVE_TAG.size());
        for (size_t token_idx = 0; token_idx < num_image_tokens; ++token_idx) {
            expanded_tag += NATIVE_TAG;
        }

        OPENVINO_ASSERT(search_pos < unified_prompt.length());
        search_pos = unified_prompt.find(NATIVE_TAG, search_pos);
        OPENVINO_ASSERT(search_pos != std::string::npos,
                        "Failed to locate <image> token while normalizing DeepSeek-OCR-2 prompt");

        unified_prompt.replace(search_pos, NATIVE_TAG.length(), expanded_tag);
        search_pos += expanded_tag.length();
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderDeepseekOCR2::get_inputs_embeds(const std::string& unified_prompt,
                                                         const std::vector<ov::genai::EncodedImage>& images,
                                                         ov::genai::VLMPerfMetrics& metrics,
                                                         bool recalculate_merged_embeddings,
                                                         const std::vector<size_t>& images_sequence) {
    (void)recalculate_merged_embeddings;
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t image_id : images_sequence) {
        image_embeds.push_back(images.at(image_id).resized_source);
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

    return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, m_image_token_id);
}

}  // namespace ov::genai
