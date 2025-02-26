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

EncodedImage VisionEncoderLLaVA::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values = get_pixel_values_llava(image, config);

    m_vision_encoder.set_tensor("pixel_values", pixel_values);
    m_vision_encoder.infer();

    const ov::Tensor& infer_output = m_vision_encoder.get_output_tensor();
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

ov::Tensor InputsEmbedderLLaVA::get_inputs_embeds(const std::string& prompt, const std::vector<ov::Tensor>& images, ov::genai::VLMPerfMetrics& metrics) {
    std::string image_token = m_vlm_config.im_start;
    
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);

    std::string formatted_prompt;
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(single_images.size());

    for (const auto& image : single_images) {
        ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};
        EncodedImage encoded_image = m_vision_encoder->encode(image, vision_config);
        image_embeds.push_back(std::move(encoded_image.resized_source));
        formatted_prompt += image_token + "\n";
    }
    formatted_prompt += prompt;

    ov::Tensor input_ids = get_encoded_input_ids(formatted_prompt, metrics);
    ov::Tensor text_embeds = m_embedding.infer(input_ids);

    if (images.empty()) {
        return text_embeds;
    }
    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor encoded_image_token = m_tokenizer.encode(m_vlm_config.im_start, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];
    return merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
}

ov::Tensor InputsEmbedderLLaVA::merge_text_and_image_embeddings_llava(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds,
    const std::vector<ov::Tensor>& image_embeds,
    int64_t image_token_id) {
    auto text_embeds_shape = text_embeds.get_shape();
    size_t text_embeds_seq_length = text_embeds_shape[1];
    size_t hidden_size = text_embeds_shape[2];

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    const float* text_embeds_data = text_embeds.data<const float>();

    size_t num_image_tokens = 0;
    for (size_t s = 0; s < text_embeds_seq_length; ++s) {
        if (input_ids_data[s] == image_token_id) {
            num_image_tokens++;
        }
    }
    auto num_images = image_embeds.size();
    OPENVINO_ASSERT(
        num_image_tokens == num_images,
        "Number of image tokens in input_ids different from num_images."
    );

    size_t total_image_seq_length = 0;
    for (const auto& single_image_embeds : image_embeds) {
        OPENVINO_ASSERT(
            text_embeds_shape[2] == single_image_embeds.get_shape().at(2),
            "Incompatible shapes between text_embeds and image_embeds"
        );
        total_image_seq_length += single_image_embeds.get_shape().at(1);
    }
    size_t merged_seq_length = text_embeds_seq_length + total_image_seq_length - num_image_tokens;

    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor merged_embeds(text_embeds.get_element_type(), {BATCH_SIZE, merged_seq_length, hidden_size});
    float* merged_data = merged_embeds.data<float>();

    size_t merged_idx = 0;
    size_t image_idx = 0;
    for (size_t s = 0; s < text_embeds_seq_length; ++s) {
        if (input_ids_data[s] == image_token_id) {
            const float* image_embeds_data = image_embeds[image_idx].data<const float>();
            size_t image_seq_length = image_embeds[image_idx].get_shape()[1];

            std::copy_n(image_embeds_data,
                        image_seq_length * hidden_size,
                        merged_data + merged_idx * hidden_size);
            merged_idx += image_seq_length;
            image_idx++;
        } else {
            std::copy_n(text_embeds_data + s * hidden_size,
                        hidden_size,
                        merged_data + merged_idx * hidden_size);
            merged_idx++;
        }
    }
    return merged_embeds;
}

} // namespace ov::genai