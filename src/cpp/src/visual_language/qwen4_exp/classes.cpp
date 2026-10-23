// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen4_exp/classes.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>

namespace ov::genai {

InputsEmbedderQwen4_Exp::InputsEmbedderQwen4_Exp(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config
) : InputsEmbedderQwen3_5(vlm_config, model_dir, tokenizer, device, device_config) {
    initialize_ngram_embeddings_model(
        utils::singleton_core().read_model(model_dir / "openvino_ngram_embeddings_model.xml"),
        device,
        device_config
    );
}

InputsEmbedderQwen4_Exp::InputsEmbedderQwen4_Exp(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config
) : InputsEmbedderQwen3_5(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    const auto& [model_str, weights] = utils::get_model_weights_pair(models_map, "ngram_embeddings");
    initialize_ngram_embeddings_model(
        utils::singleton_core().read_model(model_str, weights),
        device,
        device_config
    );
}

void InputsEmbedderQwen4_Exp::initialize_ngram_embeddings_model(
    const std::shared_ptr<ov::Model>& model,
    const std::string& device,
    const ov::AnyMap& device_config
) {
    // Set m_ngram_context_length
    const auto& ngram_history_input_pshape = model->input("ngram_history").get_partial_shape();
    OPENVINO_ASSERT(ngram_history_input_pshape.rank().is_static()
        && ngram_history_input_pshape.rank().get_length() == 2
        && ngram_history_input_pshape[1].is_static(),
        "Qwen4Exp ngram_history input shape must match text_config.ngram_size - 1.");
    m_ngram_context_length = static_cast<size_t>(ngram_history_input_pshape[1].get_length());

    m_ngram_embeddings_request = utils::singleton_core().compile_model(
        model,
        device,
        utils::get_model_properties(device_config, "ngram_embeddings", device)
    ).create_infer_request();
    m_lm_extra_inputs.clear();
}

ov::Tensor InputsEmbedderQwen4_Exp::get_ngram_embeddings(const ov::Tensor& input_ids) {
    m_ngram_embeddings_request.set_tensor("input_ids", input_ids);
    m_ngram_embeddings_request.set_tensor("ngram_history", m_ngram_history);
    m_ngram_embeddings_request.infer();

    // Update ngram_history from ngram_embeddings model output
    const ov::Tensor new_ngram_history_view = m_ngram_embeddings_request.get_tensor("new_ngram_history");
    ov::Tensor new_ngram_history(new_ngram_history_view.get_element_type(), new_ngram_history_view.get_shape());
    new_ngram_history_view.copy_to(new_ngram_history);
    m_ngram_history = std::move(new_ngram_history);

    const ov::Tensor ngram_embeddings_view = m_ngram_embeddings_request.get_tensor("ngram_embeddings");
    ov::Tensor ngram_embeddings(ngram_embeddings_view.get_element_type(), ngram_embeddings_view.get_shape());
    ngram_embeddings_view.copy_to(ngram_embeddings);

    return ngram_embeddings;
}

ov::Tensor InputsEmbedderQwen4_Exp::get_inputs_embeds(
    const std::string& prompt,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos,
    VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<size_t, size_t>>& history_vision_count
) {
    OPENVINO_ASSERT(!is_cdpruner_active(), "Qwen4Exp n-gram embeddings do not support vision token pruning.");
    const int64_t previous_rope_delta = m_rope_delta;

    ov::Tensor inputs_embeds = InputsEmbedderQwen3VL::get_inputs_embeds(
        prompt,
        images,
        videos,
        metrics,
        recalculate_merged_embeddings,
        images_sequence,
        videos_sequence,
        history_vision_count
    );

    const int64_t ngram_eos_token_id = m_tokenizer.get_pad_token_id();
    OPENVINO_ASSERT(ngram_eos_token_id > 0, "EOS token id for ngram history padding must be set.");

    const auto& input_shape = m_last_input_ids.get_shape();
    OPENVINO_ASSERT(input_shape.size() == 2 && input_shape[0] == 1,
        "Qwen4Exp prompt embedding currently supports batch 1.");

    m_ngram_history = ov::Tensor(ov::element::i64, {1, m_ngram_context_length});
    int64_t* ngram_history_data = m_ngram_history.data<int64_t>();
    std::fill_n(ngram_history_data, m_ngram_context_length, ngram_eos_token_id);

    if (m_is_chat_conversation && !m_cache_state.needs_reset()) {
        const auto& cached_ids = m_cache_state.get_state();
        OPENVINO_ASSERT(cached_ids.size() >= input_shape[1], "Qwen4Exp cache is shorter than the current prompt.");
        const size_t prefix_length = cached_ids.size() - input_shape[1];
        int64_t* positions = m_position_ids.data<int64_t>();
        for (size_t axis = 0; axis < 4; ++axis) {
            const int64_t offset = static_cast<int64_t>(prefix_length) +
                                   (axis > 0 && prefix_length > 0 ? previous_rope_delta : 0);
            for (size_t token_index = 0; token_index < input_shape[1]; ++token_index) {
                positions[axis * input_shape[1] + token_index] += offset;
            }
        }
        if (prefix_length > 0) {
            m_rope_delta += previous_rope_delta;
        }
        const size_t context_length = std::min(prefix_length, m_ngram_context_length);
        std::copy_n(cached_ids.begin() + prefix_length - context_length,
                    context_length,
                    ngram_history_data + m_ngram_context_length - context_length);
    }
    
    m_lm_extra_inputs["ngram_embeddings"] = get_ngram_embeddings(m_last_input_ids);

    return inputs_embeds;
}

LMExtraInputsCallback InputsEmbedderQwen4_Exp::get_lm_extra_inputs_callback() {
    return [this](const ov::Tensor& input_ids, const std::vector<int32_t>& beam_indices) {
        OPENVINO_ASSERT(input_ids.get_shape()[0] == beam_indices.size(),
                        "Qwen4Exp beam indices must match the current token batch.");
        ov::Tensor reordered_ngram_history(ov::element::i64, {beam_indices.size(), m_ngram_context_length});
        for (size_t batch = 0; batch < beam_indices.size(); ++batch) {
            const int32_t parent = beam_indices[batch];
            OPENVINO_ASSERT(parent >= 0 && static_cast<size_t>(parent) < m_ngram_history.get_shape()[0],
                            "Qwen4Exp parent beam index is outside the n-gram history batch.");
            std::copy_n(m_ngram_history.data<const int64_t>() + parent * m_ngram_context_length,
                        m_ngram_context_length,
                        reordered_ngram_history.data<int64_t>() + batch * m_ngram_context_length);
        }
        m_ngram_history = std::move(reordered_ngram_history);
        return std::unordered_map<std::string, ov::Tensor>{{"ngram_embeddings", get_ngram_embeddings(input_ids)}};
    };
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderQwen4_Exp::get_position_ids(
    size_t inputs_embeds_size, size_t
) {
    OPENVINO_ASSERT(m_position_ids.get_shape()[2] == inputs_embeds_size,
                    "Qwen4Exp prompt positions must match the input embedding sequence.");
    return {m_position_ids, m_rope_delta};
}

void InputsEmbedderQwen4_Exp::start_chat(const std::string& system_message) {
    InputsEmbedderQwen3_5::start_chat(system_message);
    m_ngram_history = ov::Tensor();
    m_lm_extra_inputs.clear();
}

void InputsEmbedderQwen4_Exp::finish_chat() {
    InputsEmbedderQwen3_5::finish_chat();
    m_ngram_history = ov::Tensor();
    m_lm_extra_inputs.clear();
}

std::pair<ov::Tensor, int64_t> InputsEmbedderQwen4_Exp::create_position_ids(
    const ov::Tensor& input_ids_tensor,
    const std::vector<std::array<size_t, 3>>& images_grid_thw,
    const std::vector<size_t>& images_sequence,
    const size_t image_id,
    const std::vector<std::array<size_t, 3>>& videos_grid_thw,
    const std::vector<size_t>& videos_sequence,
    const size_t video_id,
    const int64_t,
    const std::vector<std::pair<std::size_t, std::size_t>>&
) {
    OPENVINO_ASSERT(input_ids_tensor.get_shape().size() == 2 && input_ids_tensor.get_shape()[0] == 1,
                    "Qwen4Exp prompt positions currently support batch size one.");
    const size_t sequence_length = input_ids_tensor.get_shape()[1];
    OPENVINO_ASSERT(sequence_length > 0, "Qwen4Exp prompt must contain at least one token.");
    const size_t merge_size = m_vision_encoder->get_processor_config().merge_size;
    OPENVINO_ASSERT(merge_size > 0, "Qwen4Exp spatial merge size must be positive.");
    const int64_t image_token_id = m_vision_token_ids.at("image_pad");
    const int64_t video_token_id = m_vision_token_ids.at("video_pad");
    const int64_t* tokens = input_ids_tensor.data<const int64_t>();
    ov::Tensor position_ids(ov::element::i64, {4, 1, sequence_length});
    int64_t* positions = position_ids.data<int64_t>();
    std::iota(positions, positions + sequence_length, int64_t{0});

    std::vector<std::array<size_t, 3>> frame_grids;
    for (size_t video_index : videos_sequence) {
        const auto& grid = videos_grid_thw.at(video_index - video_id);
        for (size_t frame = 0; frame < grid[0]; ++frame) {
            frame_grids.push_back({1, grid[1], grid[2]});
        }
    }
    size_t image_index = 0;
    size_t frame_index = 0;
    int64_t current_position = 0;
    for (size_t start = 0; start < sequence_length;) {
        const int64_t token = tokens[start];
        if (token != image_token_id && token != video_token_id) {
            for (size_t axis = 1; axis < 4; ++axis) {
                positions[axis * sequence_length + start] = current_position;
            }
            ++current_position;
            ++start;
            continue;
        }
        size_t end = start + 1;
        while (end < sequence_length && tokens[end] == token) {
            ++end;
        }
        const auto& grid = token == image_token_id
                               ? images_grid_thw.at(images_sequence.at(image_index++) - image_id)
                               : frame_grids.at(frame_index++);
        OPENVINO_ASSERT(grid[1] % merge_size == 0 && grid[2] % merge_size == 0,
                        "Qwen4Exp vision grid must be divisible by spatial merge size.");
        const size_t grid_height = grid[1] / merge_size;
        const size_t grid_width = grid[2] / merge_size;
        OPENVINO_ASSERT(end - start == grid[0] * grid_height * grid_width,
                        "Qwen4Exp vision placeholder count does not match its grid.");
        for (size_t offset = 0; offset < end - start; ++offset) {
            positions[sequence_length + start + offset] = current_position + offset / (grid_height * grid_width);
            positions[2 * sequence_length + start + offset] = current_position + (offset / grid_width) % grid_height;
            positions[3 * sequence_length + start + offset] = current_position + offset % grid_width;
        }
        current_position += std::max(grid_height, grid_width);
        start = end;
    }
    OPENVINO_ASSERT(image_index == images_sequence.size() && frame_index == frame_grids.size(),
                    "Qwen4Exp vision grids do not match the prompt's image/video runs.");
    const int64_t maximum_position = *std::max_element(
        positions + sequence_length,
        positions + position_ids.get_size()
    );
    return {position_ids, maximum_position + 1 - static_cast<int64_t>(sequence_length)};
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderQwen4_Exp::get_generation_phase_position_ids(
    const size_t inputs_embeds_size,
    const size_t history_size,
    int64_t rope_delta
) {
    ov::Tensor position_ids(ov::element::i64, {4, 1, inputs_embeds_size});
    int64_t* positions = position_ids.data<int64_t>();
    std::iota(positions, positions + inputs_embeds_size, static_cast<int64_t>(history_size));
    for (size_t axis = 1; axis < 4; ++axis) {
        std::iota(positions + axis * inputs_embeds_size,
                  positions + (axis + 1) * inputs_embeds_size,
                  static_cast<int64_t>(history_size) + rope_delta);
    }
    return {position_ids, rope_delta};
}

const std::unordered_map<std::string, ov::Tensor>& InputsEmbedderQwen4_Exp::get_lm_extra_inputs() const {
    return m_lm_extra_inputs;
}

} // namespace ov::genai
