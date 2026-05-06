// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vlm_config.hpp"

#include <fstream>

#include "json_utils.hpp"

namespace ov::genai {

namespace {

VLMModelType to_vlm_model_type(const std::string& value) {
    static const std::unordered_map<std::string, VLMModelType> model_types_map = {
        {"minicpmv", VLMModelType::MINICPM},
        {"minicpmo", VLMModelType::MINICPM},
        {"llava", VLMModelType::LLAVA},
        {"llava-qwen2", VLMModelType::NANOLLAVA},
        {"llava_next", VLMModelType::LLAVA_NEXT},
        {"llava_next_video", VLMModelType::LLAVA_NEXT_VIDEO},
        {"internvl_chat", VLMModelType::INTERNVL_CHAT},
        {"phi3_v", VLMModelType::PHI3_V},
        {"phi4mm", VLMModelType::PHI4MM},
        {"qwen2_vl", VLMModelType::QWEN2_VL},
        {"qwen2_5_vl", VLMModelType::QWEN2_5_VL},
        {"qwen3_vl", VLMModelType::QWEN3_VL},
        {"gemma3", VLMModelType::GEMMA3},
        {"gemma4", VLMModelType::GEMMA4},
        {"videochat_flash_qwen", VLMModelType::VIDEOCHAT_FLASH_QWEN},
        {"qwen3_omni", VLMModelType::QWEN3_OMNI},
        {"qwen3_omni_moe", VLMModelType::QWEN3_OMNI},
    };

    auto it = model_types_map.find(value);
    if (it != model_types_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Unsupported '", value, "' VLM model type");
}

void assert_size(size_t size, VLMModelType model_type) {
    if (model_type == VLMModelType::PHI3_V) {
        OPENVINO_ASSERT(size == 4096, "Expected size 4096 for PHI3_V model type");
    }
}

}  // namespace

VLMConfig::VLMConfig(const std::filesystem::path& json_path) {
    std::ifstream stream(json_path);
    OPENVINO_ASSERT(stream.is_open(), "Failed to open '", json_path, "' with processor config");
    nlohmann::json parsed = nlohmann::json::parse(stream);
    using ov::genai::utils::read_json_param;
    model_type = to_vlm_model_type(parsed.at("model_type"));
    read_json_param(parsed, "hidden_size", hidden_size);
    read_json_param(parsed, "scale_emb", scale_emb);
    read_json_param(parsed, "query_num", query_num);
    read_json_param(parsed, "use_image_id", use_image_id);

    // Setting llava_next specific config params
    read_json_param(parsed, "image_newline", image_newline);
    read_json_param(parsed, "vision_config.patch_size", vision_config_patch_size);

    // phi3_v and phi4mm
    if (parsed.contains("sub_GN") && parsed.at("sub_GN").is_array()) {
        sub_GN = parsed.at("sub_GN").get<std::vector<std::vector<std::vector<std::vector<float>>>>>().at(0).at(0).at(0);
    }
    assert_size(sub_GN.size(), model_type);
    if (parsed.contains("glb_GN") && parsed.at("glb_GN").is_array()) {
        glb_GN = parsed.at("glb_GN").get<std::vector<std::vector<std::vector<float>>>>().at(0).at(0);
    }
    assert_size(glb_GN.size(), model_type);

    // Qwen2.5VL
    read_json_param(parsed, "vision_config.window_size", vision_config_window_size);
    read_json_param(parsed, "vision_config.tokens_per_second", vision_config_tokens_per_second);

    // Qwen3-VL
    read_json_param(parsed, "vision_config.num_position_embeddings", vision_config_num_position_embeddings);
    read_json_param(parsed, "vision_config.deepstack_visual_indexes", vision_config_deepstack_visual_indexes);

    // Qwen3-Omni: vision/audio configs are nested under thinker_config
    if (model_type == VLMModelType::QWEN3_OMNI) {
        read_json_param(parsed, "thinker_config.text_config.hidden_size", hidden_size);
        // Vision config nested under thinker_config
        read_json_param(parsed, "thinker_config.vision_config.num_position_embeddings", vision_config_num_position_embeddings);
        read_json_param(parsed, "thinker_config.vision_config.deepstack_visual_indexes", vision_config_deepstack_visual_indexes);
        // Audio encoder config
        read_json_param(parsed, "thinker_config.audio_config.num_mel_bins", audio_config_num_mel_bins);
        read_json_param(parsed, "thinker_config.audio_config.n_window", audio_config_n_window);
        read_json_param(parsed, "thinker_config.audio_config.n_window_infer", audio_config_n_window_infer);
        // Top-level flags and token IDs
        read_json_param(parsed, "enable_audio_output", enable_audio_output);
        read_json_param(parsed, "tts_bos_token_id", tts_bos_token_id);
        read_json_param(parsed, "tts_eos_token_id", tts_eos_token_id);
        read_json_param(parsed, "tts_pad_token_id", tts_pad_token_id);
        read_json_param(parsed, "im_start_token_id", im_start_token_id);
        read_json_param(parsed, "im_end_token_id", im_end_token_id);
        read_json_param(parsed, "system_token_id", system_token_id);
        read_json_param(parsed, "user_token_id", user_token_id);
        read_json_param(parsed, "assistant_token_id", assistant_token_id);
        // Audio/image/video token IDs from thinker config
        read_json_param(parsed, "thinker_config.audio_token_id", audio_token_id);
        read_json_param(parsed, "thinker_config.image_token_id", image_token_id);
        read_json_param(parsed, "thinker_config.video_token_id", video_token_id);
        // Talker config
        read_json_param(parsed, "talker_config.num_code_groups", talker_num_code_groups);
        read_json_param(parsed, "talker_config.thinker_hidden_size", talker_thinker_hidden_size);
        read_json_param(parsed, "talker_config.codec_bos_id", talker_codec_bos_id);
        read_json_param(parsed, "talker_config.codec_eos_token_id", talker_codec_eos_token_id);
        read_json_param(parsed, "talker_config.codec_pad_id", talker_codec_pad_id);
        read_json_param(parsed, "talker_config.codec_nothink_id", talker_codec_nothink_id);
        read_json_param(parsed, "talker_config.codec_think_bos_id", talker_codec_think_bos_id);
        read_json_param(parsed, "talker_config.codec_think_eos_id", talker_codec_think_eos_id);
        // Speaker IDs
        if (parsed.contains("talker_config") && parsed["talker_config"].contains("speaker_id")) {
            for (auto& [key, val] : parsed["talker_config"]["speaker_id"].items()) {
                speaker_ids[key] = val.get<int64_t>();
            }
        }
    }
}

}  // namespace ov::genai
