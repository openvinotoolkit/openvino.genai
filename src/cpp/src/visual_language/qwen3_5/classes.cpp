// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/qwen3_5/classes.hpp"

namespace ov::genai {

InputsEmbedderQwen3_5::InputsEmbedderQwen3_5(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config
) : InputsEmbedderQwen3VL(vlm_config, model_dir, device, device_config) {}

InputsEmbedderQwen3_5::InputsEmbedderQwen3_5(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config
) : InputsEmbedderQwen3VL(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

std::pair<ov::Tensor, int64_t> InputsEmbedderQwen3_5::create_position_ids(
    const ov::Tensor& input_ids_tensor,
    const std::vector<std::array<size_t, 3>>& images_grid_thw,
    const std::vector<size_t>& images_sequence,
    const size_t image_id,
    const std::vector<std::array<size_t, 3>>& videos_grid_thw,
    const std::vector<size_t>& videos_sequence,
    const size_t video_id,
    const int64_t vision_start_token_id,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count
) {
    const auto& [vision_position_ids, rope_delta] = InputsEmbedderQwen2VL::create_position_ids(
        input_ids_tensor,
        images_grid_thw,
        images_sequence,
        image_id,
        videos_grid_thw,
        videos_sequence,
        video_id,
        vision_start_token_id,
        history_vision_count
    );

    const auto& vision_position_ids_shape = vision_position_ids.get_shape();
    const size_t batch_size = vision_position_ids_shape[1];
    const size_t seq_len = vision_position_ids_shape[2];

    ov::Tensor position_ids{vision_position_ids.get_element_type(), {4, batch_size, seq_len}};
    int64_t* dst = position_ids.data<int64_t>();
    const int64_t* src = vision_position_ids.data<const int64_t>();

    // Add text position ids to dim 0
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            dst[b * seq_len + s] = static_cast<int64_t>(s);
        }
    }

    // Append 3D vision position ids
    std::memcpy(dst + batch_size * seq_len, src, 3 * batch_size * seq_len * sizeof(int64_t));

    return {position_ids, rope_delta};
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderQwen3_5::get_generation_phase_position_ids(
    const size_t inputs_embeds_size,
    const size_t history_size,
    int64_t rope_delta
) {
    const auto& vision_position_ids = InputsEmbedderQwen2VL::get_generation_phase_position_ids(
        inputs_embeds_size,
        history_size,
        rope_delta
    ).first;

    ov::Tensor position_ids{vision_position_ids.get_element_type(), {4, 1, inputs_embeds_size}};
    int64_t* dst = position_ids.data<int64_t>();
    const int64_t* src = vision_position_ids.data<const int64_t>();

    // Add text position ids to dim 0
    const int64_t text_position_id = static_cast<int64_t>(history_size);
    std::fill_n(dst, inputs_embeds_size, text_position_id);

    // Append 3D vision position ids
    std::memcpy(dst + inputs_embeds_size, src, 3 * inputs_embeds_size * sizeof(int64_t));
    
    return {position_ids, rope_delta};
}

void InputsEmbedderQwen3_5::start_chat(const std::string& system_message) {
    InputsEmbedderQwen2VL::start_chat(system_message);
}

void InputsEmbedderQwen3_5::finish_chat() {
    InputsEmbedderQwen2VL::finish_chat();
}

const std::unordered_map<std::string, ov::Tensor>& InputsEmbedderQwen3_5::get_lm_extra_inputs() const {    
    return InputsEmbedder::IInputsEmbedder::get_lm_extra_inputs();
}

} // namespace ov::genai
