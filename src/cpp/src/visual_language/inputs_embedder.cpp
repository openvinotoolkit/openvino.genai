// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "visual_language/inputs_embedder.hpp"

#include "visual_language/clip.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/embedding_model.hpp"

#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/qwen2_5_vl/classes.hpp"
#include "visual_language/phi3_vision/classes.hpp"
#include "visual_language/phi4mm/classes.hpp"
#include "visual_language/minicpm/classes.hpp"
#include "visual_language/llava/classes.hpp"
#include "visual_language/nanollava/classes.hpp"
#include "visual_language/llava_next/classes.hpp"
#include "visual_language/llava_next_video/classes.hpp"
#include "visual_language/internvl_chat/classes.hpp"
#include "visual_language/gemma3/classes.hpp"

#include "utils.hpp"

namespace ov::genai {

// Base InputsEmbedder class

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedder::IInputsEmbedder::get_position_ids(const size_t inputs_embeds_size, const size_t history_size) {
    ov::Tensor position_ids = ov::Tensor{ov::element::i64, { 1, inputs_embeds_size }};
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), history_size);
    return {position_ids, std::nullopt};
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedder::IInputsEmbedder::get_generation_phase_position_ids(const size_t inputs_embeds_size, const size_t history_size, int64_t rope_delta) {
    return get_position_ids(inputs_embeds_size, history_size);
}

void InputsEmbedder::IInputsEmbedder::start_chat(const std::string& system_message) {
    m_is_chat_conversation = true;
    if (!m_kv_cache_state.get_state().empty()) {
        m_kv_cache_state.reset_state();
    }
    if (system_message.empty()) {
        return;
    }
}

void InputsEmbedder::IInputsEmbedder::update_chat_history(const std::string& decoded_results, const ov::genai::GenerationStatus generation_finish_status) {
    m_kv_cache_state.num_tokens_to_trim = 0;
    if (generation_finish_status == ov::genai::GenerationStatus::CANCEL) {
        // If chat generation process was cancelled by user, let's rollback to previous state of kv cache
        std::vector<int64_t>& state = m_kv_cache_state.get_state();

        m_kv_cache_state.num_tokens_to_trim = state.size() - m_prev_hist_length;
        state.resize(m_prev_hist_length);
        m_kv_cache_state.reset_mem_state = state.empty();
    }
}

void InputsEmbedder::IInputsEmbedder::finish_chat() {
    m_is_chat_conversation = false;
    m_kv_cache_state.reset_state();
}

bool InputsEmbedder::IInputsEmbedder::clean_historical_vision_tokens(ov::Tensor& input_ids,
                                                                     int64_t vision_pad_token_id,
                                                                     int64_t vision_start_token_id,
                                                                     int64_t vision_end_token_id,
                                                                     size_t expected_count) {
    if (!m_is_chat_conversation || m_prev_hist_length == 0) {
        return false;
    }

    std::vector<int64_t> input_ids_vec(input_ids.data<int64_t>(), input_ids.data<int64_t>() + input_ids.get_shape()[1]);
    size_t vision_pad_count = std::count(input_ids_vec.begin(), input_ids_vec.end(), vision_pad_token_id);

    if (vision_pad_count <= expected_count) {
        return false;
    }

    // Keep only the last expected_count vision_pads
    size_t num_pads_to_skip = vision_pad_count - expected_count;

    std::vector<int64_t> cleaned_ids;
    size_t pads_skipped = 0;

    for (int64_t token : input_ids_vec) {
        if (token == vision_pad_token_id) {
            if (pads_skipped < num_pads_to_skip) {
                pads_skipped++;
            } else {
                cleaned_ids.push_back(token);
            }
        } else {
            cleaned_ids.push_back(token);
        }
    }

    // Remove orphaned vision_start/vision_end tokens
    std::vector<int64_t> final_cleaned_ids;
    for (size_t i = 0; i < cleaned_ids.size(); ++i) {
        if (cleaned_ids[i] == vision_start_token_id) {
            bool has_pads = false;
            for (size_t j = i + 1; j < cleaned_ids.size() && cleaned_ids[j] != vision_end_token_id; ++j) {
                if (cleaned_ids[j] == vision_pad_token_id) {
                    has_pads = true;
                    break;
                }
            }
            if (has_pads) {
                final_cleaned_ids.push_back(cleaned_ids[i]);
            }
        } else if (cleaned_ids[i] == vision_end_token_id) {
            bool has_matching_start = false;
            for (int j = final_cleaned_ids.size() - 1; j >= 0; --j) {
                if (final_cleaned_ids[j] == vision_start_token_id) {
                    has_matching_start = true;
                    break;
                }
            }
            if (has_matching_start) {
                final_cleaned_ids.push_back(cleaned_ids[i]);
            }
        } else {
            final_cleaned_ids.push_back(cleaned_ids[i]);
        }
    }

    input_ids = ov::Tensor(ov::element::i64, {1, final_cleaned_ids.size()});
    std::copy(final_cleaned_ids.begin(), final_cleaned_ids.end(), input_ids.data<int64_t>());

    // Remove the same tokens from kv_cache_state
    std::vector<int64_t>& kv_cache = m_kv_cache_state.get_state();
    size_t tokens_removed = input_ids_vec.size() - final_cleaned_ids.size();
    if (kv_cache.size() >= tokens_removed) {
        kv_cache.erase(kv_cache.end() - tokens_removed, kv_cache.end());
    }

    return true;
}

InputsEmbedder::IInputsEmbedder::IInputsEmbedder(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config) :
    m_vlm_config{vlm_config},
    m_vision_encoder(VisionEncoder::create(model_dir, m_vlm_config.model_type, device, device_config)),
    m_embedding(EmbeddingsModel::create(model_dir, m_vlm_config.scale_emb, device, device_config)),
    m_tokenizer{model_dir, device_config},
    m_pruning_processor(std::make_shared<VisionTokenPruningProcessor>(device)) { }

InputsEmbedder::IInputsEmbedder::IInputsEmbedder(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config) :
    m_vlm_config{vlm_config},
    m_vision_encoder(VisionEncoder::create(
        models_map,
        config_dir_path,
        m_vlm_config.model_type,
        device,
        device_config
    )),
    m_embedding(EmbeddingsModel::create(
        utils::get_model_weights_pair(models_map, "text_embeddings").first,
        utils::get_model_weights_pair(models_map, "text_embeddings").second,
        m_vlm_config.scale_emb,
        device,
        device_config
    )),
    m_tokenizer(tokenizer),
    m_pruning_processor(std::make_shared<VisionTokenPruningProcessor>(device)) { }

ov::Tensor InputsEmbedder::IInputsEmbedder::apply_chat_template_tokenize(const std::string& prompt, ov::genai::VLMPerfMetrics& metrics) {
    bool add_special_tokens = m_add_special_tokens_is_set ? m_add_special_tokens : !(m_is_chat_conversation || m_apply_chat_template);
    if (m_is_chat_conversation) {
        std::string prompt_to_encode = prompt;
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        ov::Tensor new_chat_tokens = m_tokenizer.encode(prompt_to_encode, ov::genai::add_special_tokens(add_special_tokens)).input_ids;
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        return new_chat_tokens;
    } else {
        ov::Tensor encoded_input_ids;
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        if (m_apply_chat_template) {
            std::string templated_prompt;
            ChatHistory history({{{"role", "user"}, {"content", prompt}}});
            constexpr bool add_generation_prompt = true;

            templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
            encoded_input_ids = m_tokenizer.encode(templated_prompt, ov::genai::add_special_tokens(add_special_tokens)).input_ids;
        } else {
            encoded_input_ids = m_tokenizer.encode(prompt, ov::genai::add_special_tokens(add_special_tokens)).input_ids;
        }
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        return encoded_input_ids;
    }
}

ov::Tensor InputsEmbedder::IInputsEmbedder::update_history(const ov::Tensor& new_chat_tokens) {
    ov::Tensor encoded_inputs;
    if (m_is_chat_conversation) {
        ov::genai::align_kv_cache_and_history(new_chat_tokens, m_kv_cache_state);

        // Sync m_prev_hist_length after align to prevent state mismatch when pruning is enabled.
        size_t kv_cache_after_align = m_kv_cache_state.get_state().size();
        if (m_prev_hist_length > kv_cache_after_align) {
            m_prev_hist_length = kv_cache_after_align;
        }

        encoded_inputs = get_chat_encoded_input(new_chat_tokens, m_kv_cache_state).input_ids;
    } else {
        encoded_inputs = new_chat_tokens;
    }

    return encoded_inputs;
}

ov::Tensor InputsEmbedder::IInputsEmbedder::get_encoded_input_ids(const std::string& prompt, ov::genai::VLMPerfMetrics& metrics) {
    const auto new_chat_tokens = apply_chat_template_tokenize(prompt, metrics);
    auto new_input_ids = update_history(new_chat_tokens);
    m_prev_hist_length = m_kv_cache_state.get_state().size();
    m_kv_cache_state.add_inputs(new_input_ids);
    return new_input_ids;
}

std::vector<ov::Tensor> InputsEmbedder::IInputsEmbedder::to_single_image_tensors(const std::vector<ov::Tensor>& images) {
    std::vector<ov::Tensor> single_image_tensors;
    for (const auto& image : images) {
        ov::Tensor reshaped_image = image;
        ov::Shape image_shape = image.get_shape();
        switch (image_shape.size()) {
            case 3:
                reshaped_image.set_shape({1, image_shape.at(0), image_shape.at(1), image_shape.at(2)});
                break;
            case 4: break;
            default: OPENVINO_THROW("Input image must have [NHWC] or [HWC] layout, given image shape is ", image_shape);
        }
        ov::Shape reshaped_image_shape = reshaped_image.get_shape();
        for (size_t batch_idx = 0; batch_idx < reshaped_image_shape.at(0); ++batch_idx) {
            ov::Tensor single_image{
                reshaped_image.get_element_type(),
                {1, reshaped_image_shape.at(1), reshaped_image_shape.at(2), reshaped_image_shape.at(3)},
                reshaped_image.data<uint8_t>() + batch_idx * reshaped_image_shape.at(1) * reshaped_image_shape.at(2) * reshaped_image_shape.at(3)
            };
            single_image_tensors.push_back(std::move(single_image));
        }
    }
    return single_image_tensors;
}

std::vector<ov::genai::EncodedImage> InputsEmbedder::IInputsEmbedder::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image));
    }
    return embeds;
}

ov::Tensor InputsEmbedder::IInputsEmbedder::get_inputs_embeds(
    const std::string& prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    const std::vector<ov::genai::EncodedVideo>& videos,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    if (!videos.size()) {
        return get_inputs_embeds(prompt, images, metrics, recalculate_merged_embeddings, images_sequence);
    }
    OPENVINO_THROW("Current model doesn't support video preprocess currently. Input images are processed as separate images.");
}

std::vector<ov::genai::EncodedVideo> InputsEmbedder::IInputsEmbedder::encode_videos(const std::vector<ov::Tensor>& videos) {
    if (!videos.size()) {
        return {};
    }
    OPENVINO_THROW("Current model doesn't support video preprocess currently. Input images are processed as separate images.");
}

NormalizedPrompt InputsEmbedder::IInputsEmbedder::normalize_prompt(
    const std::string& prompt,
    size_t base_image_id,
    size_t base_video_id,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos) const {
    if (!videos.size()) {
        return normalize_prompt(prompt, base_image_id, images);
    }
    OPENVINO_THROW("Current model doesn't support video preprocess currently. Input images are processed as separate images.");
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedder::IInputsEmbedder::get_inputs_embeds_with_token_type_ids(
    const std::string& prompt,
    const std::vector<EncodedImage>& images,
    VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence) {
    OPENVINO_THROW("This model does not support token_type_ids.");
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedder::IInputsEmbedder::get_inputs_embeds_with_token_type_ids(
    const std::string& prompt,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos,
    VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    OPENVINO_ASSERT(videos.size() == 0U, "The model doesn't support 'videos' preprocessing yet. Please use 'images' instead.");

    return get_inputs_embeds_with_token_type_ids(prompt, images, metrics, recalculate_merged_embeddings, image_sequence);
}

bool InputsEmbedder::IInputsEmbedder::has_token_type_ids() const { return false; }

/// Public InputsEmbedder class

InputsEmbedder::InputsEmbedder(const std::filesystem::path& model_dir,
                               const std::string& device,
                               const ov::AnyMap device_config) {
    auto vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");

    if (vlm_config.model_type == VLMModelType::MINICPM) {
        m_impl = std::make_shared<InputsEmbedderMiniCPM>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA) {
        m_impl = std::make_shared<InputsEmbedderLLaVA>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::NANOLLAVA) {
        m_impl = std::make_shared<InputsEmbedderNanoLLaVA>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA_NEXT) {
        m_impl = std::make_shared<InputsEmbedderLLaVANext>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA_NEXT_VIDEO) {
        m_impl = std::make_shared<InputsEmbedderLLaVANextVideo>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::INTERNVL_CHAT) {
        m_impl = std::make_shared<InputsEmbedderInternVLChat>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::PHI3_V) {
        m_impl = std::make_shared<InputsEmbedderPhi3V>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::PHI4MM) {
        m_impl = std::make_shared<InputsEmbedderPhi4MM>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::QWEN2_VL) {
        m_impl = std::make_shared<InputsEmbedderQwen2VL>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::QWEN2_5_VL) {
        m_impl = std::make_shared<InputsEmbedderQwen2_5_VL>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::GEMMA3) {
        m_impl = std::make_shared<InputsEmbedderGemma3>(vlm_config, model_dir, device, device_config);
    } else {
        OPENVINO_THROW("Unsupported model type in VLM InputsEmbedder class. Please, create feature request on new model support");
    }
}

InputsEmbedder::InputsEmbedder(const ModelsMap& models_map,
                               const Tokenizer& tokenizer,
                               const std::filesystem::path& config_dir_path,
                               const std::string& device,
                               const ov::AnyMap device_config) {
    auto vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");

    if (vlm_config.model_type == VLMModelType::MINICPM) {
        m_impl = std::make_shared<InputsEmbedderMiniCPM>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA) {
        m_impl = std::make_shared<InputsEmbedderLLaVA>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::NANOLLAVA) {
        m_impl = std::make_shared<InputsEmbedderNanoLLaVA>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA_NEXT) {
        m_impl = std::make_shared<InputsEmbedderLLaVANext>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA_NEXT_VIDEO) {
        m_impl = std::make_shared<InputsEmbedderLLaVANextVideo>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::INTERNVL_CHAT) {
        m_impl = std::make_shared<InputsEmbedderInternVLChat>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::PHI3_V) {
        m_impl = std::make_shared<InputsEmbedderPhi3V>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::PHI4MM) {
        m_impl = std::make_shared<InputsEmbedderPhi4MM>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::QWEN2_VL) {
        m_impl = std::make_shared<InputsEmbedderQwen2VL>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::QWEN2_5_VL) {
        m_impl = std::make_shared<InputsEmbedderQwen2_5_VL>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::GEMMA3) {
        m_impl = std::make_shared<InputsEmbedderGemma3>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else {
        OPENVINO_THROW("Unsupported model type in VLM InputsEmbedder class. Please, create feature request on new model support");
    }
}

ov::Tensor InputsEmbedder::get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& image_sequence) {
    return m_impl->get_inputs_embeds(prompt, images, metrics, recalculate_merged_embeddings, image_sequence);
}

ov::Tensor InputsEmbedder::get_inputs_embeds(const std::string& prompt,
                                             const std::vector<ov::genai::EncodedImage>& images,
                                             const std::vector<ov::genai::EncodedVideo>& videos,
                                             ov::genai::VLMPerfMetrics& metrics,
                                             bool recalculate_merged_embeddings,
                                             const std::vector<size_t>& images_sequence,
                                             const std::vector<size_t>& videos_sequence,
                                             const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    return m_impl->get_inputs_embeds(prompt,
                                     images,
                                     videos,
                                     metrics,
                                     recalculate_merged_embeddings,
                                     images_sequence,
                                     videos_sequence,
                                     history_vision_count);
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedder::get_inputs_embeds_with_token_type_ids(
    const std::string& prompt,
    const std::vector<EncodedImage>& images,
    VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence) {
    return m_impl->get_inputs_embeds_with_token_type_ids(
        prompt, images, metrics, recalculate_merged_embeddings, image_sequence);
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedder::get_inputs_embeds_with_token_type_ids(
    const std::string& prompt,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos,
    VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    return m_impl->get_inputs_embeds_with_token_type_ids(prompt,
                                                         images,
                                                         videos,
                                                         metrics,
                                                         recalculate_merged_embeddings,
                                                         image_sequence,
                                                         videos_sequence,
                                                         history_vision_count);
}

bool InputsEmbedder::has_token_type_ids() const {
    return m_impl->has_token_type_ids();
}

std::vector<ov::genai::EncodedImage> InputsEmbedder::encode_images(const std::vector<ov::Tensor>& images) {
    return m_impl->encode_images(images);
}

std::vector<ov::genai::EncodedVideo> InputsEmbedder::encode_videos(const std::vector<ov::Tensor>& videos) {
    return m_impl->encode_videos(videos);
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedder::get_position_ids(const size_t inputs_embeds_size, const size_t history_size) {
    return m_impl->get_position_ids(inputs_embeds_size, history_size);
}

void InputsEmbedder::set_position_ids(const ov::Tensor& position_ids) {
    m_impl->set_position_ids(position_ids);
}

void InputsEmbedder::set_rope_delta(int64_t rope_delta) {
    m_impl->set_rope_delta(rope_delta);
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedder::get_generation_phase_position_ids(const size_t inputs_embeds_size, const size_t history_size, int64_t rope_delta) {
    return m_impl->get_generation_phase_position_ids(inputs_embeds_size, history_size, rope_delta);
}

EmbeddingsModel::Ptr InputsEmbedder::get_embedding_model() const {
    return m_impl->get_embedding_model();
}

ov::genai::utils::KVCacheState& InputsEmbedder::get_kv_cache_state() {
    return  m_impl->get_kv_cache_state();
}

Tokenizer InputsEmbedder::get_tokenizer() const {
    return m_impl->get_tokenizer();
}

void InputsEmbedder::start_chat(const std::string& system_message) {
    return m_impl->start_chat(system_message);
}

void InputsEmbedder::update_chat_history(const std::string& decoded_results, const ov::genai::GenerationStatus generation_finish_status) {
    return m_impl->update_chat_history(decoded_results, generation_finish_status);
}

void InputsEmbedder::set_apply_chat_template_status(bool apply_chat_template) {
    return m_impl->set_apply_chat_template_status(apply_chat_template);
}

void InputsEmbedder::finish_chat() {
    return m_impl->finish_chat();
}

void InputsEmbedder::set_vision_token_pruning_config(size_t pruning_ratio, float relevance_weight) {
    return m_impl->set_vision_token_pruning_config(pruning_ratio, relevance_weight);
}

NormalizedPrompt InputsEmbedder::normalize_prompt(
    const std::string& prompt,
    size_t base_id,
    const std::vector<EncodedImage>& images
) const {
    auto norm_prompt = m_impl->normalize_prompt(prompt, base_id, 0, images, {});
    return {norm_prompt.unified_prompt, norm_prompt.images_sequence};
}

NormalizedPrompt InputsEmbedder::normalize_prompt(const std::string& prompt,
    size_t base_image_id,
    size_t base_video_id,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos
) const {
     return m_impl->normalize_prompt(prompt, base_image_id, base_video_id, images, videos);
}

void verify_ids(const std::vector<size_t>& image_ids, size_t base_id, size_t n_images) {
    for (size_t idx : image_ids) {
        OPENVINO_ASSERT(base_id <= idx, "Referring to older images isn't implemented");
        OPENVINO_ASSERT(idx < base_id + n_images, "Missing image ", idx);
    }
}

std::pair<std::string, std::vector<size_t>> InputsEmbedder::IInputsEmbedder::normalize(
    const std::string& prompt,
    const std::string& native_tag,
    const std::string& automatic_tag,
    size_t base_id,
    size_t n_images
) const {
    size_t pos = prompt.find(native_tag);
    auto [image_prompt, image_sequence] = universal_to_native(prompt, [&](std::ostream& os, size_t) {
        os << automatic_tag;
    });
    if (!image_sequence.empty()) {
        OPENVINO_ASSERT(pos == std::string::npos, "Prompt can contain only one type of image tags.");
        verify_ids(image_sequence, base_id, n_images);
        return {std::move(image_prompt), std::move(image_sequence)};
    }
    // Restore ids from native tags
    while (pos != std::string::npos) {
        image_sequence.push_back(base_id + image_sequence.size());
        pos = prompt.find(native_tag, pos + native_tag.length());
    }
    if (!image_sequence.empty()) {
        OPENVINO_ASSERT(image_sequence.size() == n_images, "The number of native image tags and provided images must match because it's ambiguous which image should be ignored.");
        return {std::move(image_prompt), std::move(image_sequence)};
    }
    // Prepend automatic tags
    std::stringstream stream;
    for (size_t relative_id = 0; relative_id < n_images; relative_id++) {
        image_sequence.push_back(base_id + relative_id);
        stream << automatic_tag;
    }
    stream << prompt;
    return {stream.str(), std::move(image_sequence)};
}

} // namespace ov::genai
