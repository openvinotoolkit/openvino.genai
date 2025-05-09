// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/phi4mm/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

// TODO Remove debug utils
#include "print_tensor.hpp"
#include "compare_tensors.hpp"

namespace ov::genai {

namespace {
    
constexpr size_t INPUT_IMAGE_SIZE = 336;
const std::regex NATIVE_PATTERN{R"(<\|image_(\d+)\|>)"};

// Similar to Phi3, but without newline after image tag
void write_native(std::ostream& os, size_t idx) {
    os << "<|image_" << idx + 1 << "|>";
}

// FIXME Copied from Phi3 - reuse
std::string normalize_prompt(
    const std::string& prompt, size_t base_id, size_t n_images
) {
    std::smatch match;
    std::regex_search(prompt, match, NATIVE_PATTERN);
    auto [image_prompt, image_sequence] = universal_to_native(prompt, write_native);
    if (!image_sequence.empty()) {
        OPENVINO_ASSERT(match.empty(), "Prompt can contain only one type of image tags.");
        verify_ids(image_sequence, base_id, n_images);
        return image_prompt;
    }
    // Restore ids from native tags
    if (!match.empty()) {
        size_t image_id = std::stoul(match.str(1));
        OPENVINO_ASSERT(image_id != 0, "Image tags must be greater than 0");
        image_sequence.push_back(image_id - 1);
        constexpr int submatch_id_to_return = 1;
        for (std::sregex_token_iterator iter{
            match.suffix().first,
            prompt.end(),
            NATIVE_PATTERN,
            submatch_id_to_return
        }; iter != std::sregex_token_iterator{}; ++iter) {
            size_t image_id = std::stoul(*iter);
            OPENVINO_ASSERT(image_id != 0, "Image tags must be greater than 0");
            image_sequence.push_back(image_id - 1);
        }
        if (!image_sequence.empty()) {
            verify_ids(image_sequence, base_id, n_images);
            return image_prompt;
        }
    }
    // Prepend native tags
    std::stringstream stream;
    for (size_t relative_id = 0; relative_id < n_images; relative_id++) {
        image_sequence.push_back(base_id + relative_id);
        write_native(stream, image_sequence.back());
    }
    stream << prompt;
    return stream.str();
}

// FIXME Copied from Phi3 - reuse
/// @brief ov::Tensor is tokenized text, size_t is image tag
std::vector<std::variant<ov::Tensor, size_t>> split_tokenize(const std::string& text, ov::genai::Tokenizer& tokenizer) {
    std::vector<std::variant<ov::Tensor, size_t>> tokenized;
    auto prefix_begin = text.begin();
    bool is_submatch = false;
    for (std::sregex_token_iterator iter{
        prefix_begin,
        text.end(),
        NATIVE_PATTERN,
        {0, 1}  // Every match emits two values: whole match and submatch
    }; iter != std::sregex_token_iterator{}; ++iter) {
        if (is_submatch) {
            tokenized.push_back(std::stoul(iter->str()) - 1);
        } else {
            std::string regular_text{prefix_begin, iter->first};
            if (!regular_text.empty()) {
                tokenized.push_back(tokenizer.encode(regular_text, ov::genai::add_special_tokens(true)).input_ids);
            }
            prefix_begin = iter->second;
        }
        is_submatch = !is_submatch;
    }
    std::string regular_text{prefix_begin, text.end()};
    if (!regular_text.empty()) {
        tokenized.push_back(tokenizer.encode(regular_text, ov::genai::add_special_tokens(true)).input_ids);
    }
    return tokenized;
}

// FIXME Copied from Phi3 - reuse
ov::Tensor insert_image_placeholders(
    const std::vector<std::variant<ov::Tensor, size_t>>& chunks,
    const std::vector<size_t>& tokens_per_images
) {
    size_t merged_length = 0;
    for (const std::variant<ov::Tensor, size_t>& chunk : chunks) {
        merged_length += std::visit(utils::overloaded{
            [](const ov::Tensor& chunk) {
                return chunk.get_shape().at(1);
            },
            [&](size_t image_id) {
                return tokens_per_images.at(image_id);
            }
        }, chunk);
    }
    ov::Tensor merged{ov::element::i64, {1, merged_length}};
    size_t offset = 0;
    for (const std::variant<ov::Tensor, size_t>& chunk : chunks) {
        offset += std::visit(utils::overloaded{
            [&](const ov::Tensor& chunk) {
                size_t length = chunk.get_shape().at(1);
                std::copy_n(
                    chunk.data<int64_t>(),
                    length,
                    merged.data<int64_t>() + offset
                );
                return length;
            },
            [&](size_t image_id) {
                int64_t fill_value = -(static_cast<int64_t>(image_id)) - 1;
                std::fill_n(
                    merged.data<int64_t>() + offset,
                    tokens_per_images.at(image_id),
                    fill_value  // -1 to distinguish 0 token and 0 image id.
                );
                return tokens_per_images.at(image_id);
            }
        }, chunk);
    }
    return merged;
}

// FIXME Copied from Phi3 - reuse
std::vector<std::variant<ov::Tensor, size_t>> drop_image_placeholders(const ov::Tensor& tokens) {
    std::vector<std::variant<ov::Tensor, size_t>> chunks;
    int64_t last_token = tokens.data<int64_t>()[0];
    size_t text_start = 0;
    for (size_t offset = 1; offset < tokens.get_shape().at(1); ++offset) {
        // If last_token and next_token are not negative, it's continuation of the current chunk text - skip
        // If last_token is negative and next_token is not negative, it's a start of text - save the offset, add image placeholder
        // If last token is not negative and next_token is negative, it's an end of text - push_back a chunk
        // If last_token and next_token are negative, it's continuation of an image placeholder - skip
        // if last_token and next_token are negative but different, it's a start of a new image placeholder - save the previous image placeholder
        int64_t next_token = tokens.data<int64_t>()[offset];
        if (last_token < 0 && next_token >= 0) {
            text_start = offset;
            chunks.push_back(size_t(-(last_token + 1)));
        } else if (last_token >= 0 && next_token < 0) {
            chunks.emplace_back(
                std::in_place_type<ov::Tensor>,
                ov::element::i64,
                ov::Shape{1, offset - text_start},
                tokens.data<int64_t>() + text_start
            );
        } else if (last_token < 0 && next_token < 0 && last_token != next_token) {
            chunks.push_back(size_t(-(last_token + 1)));
        }
        last_token = next_token;
    }
    // Add the last chunk
    size_t full_length = tokens.get_shape().at(1);
    if (last_token >= 0) {
        chunks.emplace_back(
            std::in_place_type<ov::Tensor>,
            ov::element::i64,
            ov::Shape{1, full_length - text_start},
            tokens.data<int64_t>() + text_start
        );
    } else {
        chunks.push_back(size_t(-(last_token + 1)));
    }
    return chunks;
}

} // namespace


VisionEncoderPhi4MM::VisionEncoderPhi4MM(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap properties) : VisionEncoder(model_dir, device, properties) {
    
    auto compiled_model = utils::singleton_core().compile_model(model_dir / "openvino_vision_projection_model.xml", device, {});
    m_ireq_queue_vision_projection = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
}

VisionEncoderPhi4MM::VisionEncoderPhi4MM(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap properties) : VisionEncoder(models_map, config_dir_path, device, properties) {
    
    const auto& vision_projection_model = utils::get_model_weights_pair(models_map, "vision_projection").first;
    const auto& vision_projection_weights = utils::get_model_weights_pair(models_map, "vision_projection").second;
    auto compiled_model = utils::singleton_core().compile_model(vision_projection_model, vision_projection_weights, device, properties);
    m_ireq_queue_vision_projection = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
}

// FIXME Replace tensor mocks with real encoding logic
EncodedImage VisionEncoderPhi4MM::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    // CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    // ov::InferRequest& encoder = infer_request_guard.get();
    // ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    // // Create attention mask for image patches
    // int height = image.get_shape()[1];
    // int width = image.get_shape()[2];
    // ov::Tensor patch_attention_mask = ov::Tensor{ov::element::boolean, {1, height, width}};
    // std::fill_n(patch_attention_mask.data<bool>(), patch_attention_mask.get_size(), true);

    // // Get position IDs for the image
    // ov::Tensor patch_position_ids = get_vision_position_ids(
    //     image, 
    //     patch_attention_mask, 
    //     config.patch_size, 
    //     config.vision_config_image_size / config.patch_size
    // );
    
    // encoder.set_input_tensor("pixel_values", image);
    // encoder.set_input_tensor("patch_attention_mask", patch_attention_mask);
    // encoder.set_input_tensor("patch_position_ids", patch_position_ids);
    // encoder.infer();
    // ov::Tensor vision_features = encoder.get_output_tensor();

    // // Create encoded image result
    // EncodedImage encoded_image;
    // encoded_image.resized_source = vision_features;
    // encoded_image.resized_source_size = {
    //     static_cast<size_t>(height / config.patch_size),
    //     static_cast<size_t>(width / config.patch_size)
    // };
    
    // CircularBufferQueueElementGuard<ov::InferRequest> vision_projection_ireq_guard(this->m_ireq_queue_vision_projection.get());
    // ov::InferRequest& vision_projection = vision_projection_ireq_guard.get();
    // vision_projection.set_input_tensor(vision_features);
    // vision_projection.infer();
    // encoded_image.images_features_projection = vision_projection.get_output_tensor();
    
    // return encoded_image;


    // Using mocked tensors
    EncodedImage encoded_image;

    ov::Tensor img_features = read_tensor_from_file("./temp/tensors/phi4mm/img_features.bin");

    encoded_image.resized_source = img_features;
    
    ov::Shape shape = img_features.get_shape();
    encoded_image.resized_source_size = {
        static_cast<size_t>(shape[1] / m_processor_config.patch_size),
        static_cast<size_t>(shape[2] / m_processor_config.patch_size)
    };
    
    encoded_image.original_image_size = {
        static_cast<size_t>(image.get_shape()[2]),
        static_cast<size_t>(image.get_shape()[1])
    };

    ov::Tensor img_feature_proj = read_tensor_from_file("./temp/tensors/phi4mm/img_feature_proj.bin");
    
    encoded_image.images_features_projection = img_feature_proj;
    
    return encoded_image;
}


InputsEmbedderPhi4MM::InputsEmbedderPhi4MM(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config
) : IInputsEmbedder(vlm_config, model_dir, device, device_config) {}


InputsEmbedderPhi4MM::InputsEmbedderPhi4MM(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

// FIXME Copied from Phi3 (except debug tensors printing and comparing) - reuse
ov::Tensor InputsEmbedderPhi4MM::get_inputs_embeds(
    const std::string& prompt, 
    const std::vector<ov::genai::EncodedImage>& images, 
    ov::genai::VLMPerfMetrics& metrics, 
    bool recalculate_merged_embeddings
) {
    size_t base_id = m_tokens_per_images.size();
    std::string image_prompt = normalize_prompt(prompt, base_id, images.size());

    std::vector<ov::Tensor> images_features_proj;
    for (const ov::genai::EncodedImage& encoded_image : images) {
        images_features_proj.push_back(encoded_image.images_features_projection);
        m_tokens_per_images.push_back(images_features_proj.back().get_shape().at(1));
    }

    std::vector<std::variant<ov::Tensor, size_t>> new_chat_tokens;
    if (m_is_chat_conversation) {
        m_history.push_back({{"role", "user"}, {"content", std::move(image_prompt)}});
        constexpr bool add_generation_prompt = true;
        std::string new_templated_chat_history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        new_chat_tokens = split_tokenize(new_templated_chat_history, m_tokenizer);
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    } else {
        std::string templated_prompt;
        if (m_apply_chat_template) {
            ChatHistory history({{{"role", "user"}, {"content", std::move(image_prompt)}}});
            constexpr bool add_generation_prompt = true;
            templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
        } else {
            templated_prompt = std::move(image_prompt);
        }
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        new_chat_tokens = split_tokenize(templated_prompt, m_tokenizer);
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    }
    
    ov::Tensor new_merged_tokens = insert_image_placeholders(new_chat_tokens, m_tokens_per_images);
    ov::Tensor new_tokens = update_history(new_merged_tokens);
    m_prev_hist_length = m_kv_cache_state.get_state().size();
    m_kv_cache_state.add_inputs(new_tokens);

    ov::Tensor input_ids_python = read_tensor_from_file("./temp/tensors/phi4mm/input_ids.bin");
    print_tensor(new_tokens, "new_tokens");
    print_tensor(input_ids_python, "input_ids_python");
    compare_tensors_auto(new_tokens, input_ids_python, "new_tokens", "input_ids_python");
    
    std::vector<std::variant<ov::Tensor, size_t>> tokens = drop_image_placeholders(new_tokens);
    ov::Tensor inputs_embeds{ov::element::f32, {1, new_tokens.get_shape().at(1), m_vlm_config.hidden_size}};
    size_t offset = 0;
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    for (const std::variant<ov::Tensor, size_t>& chunk : tokens) {
        offset += std::visit(utils::overloaded{
            [&](const ov::Tensor& chunk) {
                const ov::Tensor& text_embeds = m_embedding->infer(req, chunk);
                size_t text_length = text_embeds.get_shape().at(1);
                std::copy_n(
                    text_embeds.data<float>(),
                    text_embeds.get_size(),
                    inputs_embeds.data<float>() + offset * m_vlm_config.hidden_size
                );
                return text_length;
            },
            [&](size_t image_id) {
                const ov::Tensor& image_embeds = images_features_proj.at(image_id - base_id);
                size_t im_length = image_embeds.get_shape().at(1);
                std::copy_n(
                    image_embeds.data<float>(),
                    image_embeds.get_size(),
                    inputs_embeds.data<float>() + offset * m_vlm_config.hidden_size
                );
                return im_length;
            }
        }, chunk);
    }
    
    if (!m_is_chat_conversation) {
        m_tokens_per_images.clear();
    }

    print_tensor(inputs_embeds, "inputs_embeds");
    
    return inputs_embeds;
}

// FIXME Copied from Phi3 - reuse
void InputsEmbedderPhi4MM::update_chat_history(
    const std::string& decoded_results, 
    const ov::genai::GenerationStatus generation_finish_status
) {
    IInputsEmbedder::update_chat_history(decoded_results, generation_finish_status);
    if (generation_finish_status == ov::genai::GenerationStatus::CANCEL) {
        m_tokens_per_images = m_prev_tokens_per_images;
    } else {
        m_prev_tokens_per_images = m_tokens_per_images;
    }
}

// FIXME Copied from Phi3 - reuse
void InputsEmbedderPhi4MM::start_chat(const std::string& system_message) {
    IInputsEmbedder::start_chat(system_message);
    m_tokens_per_images.clear();
}

// FIXME Copied from Phi3 - reuse
void InputsEmbedderPhi4MM::finish_chat() {
    IInputsEmbedder::finish_chat();
    m_tokens_per_images.clear();
}

// FIXME Copied from Phi3 - reuse
bool InputsEmbedderPhi4MM::prompt_has_image_tag(const std::string& prompt) const {
    return IInputsEmbedder::prompt_has_image_tag(prompt) || std::regex_search(prompt, NATIVE_PATTERN);
}

} // namespace ov::genai
