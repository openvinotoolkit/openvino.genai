// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/whisper_generation_config.hpp"

#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <openvino/runtime/core.hpp>

#include "json_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

WhisperGenerationConfig::WhisperGenerationConfig() {
    apply_chat_template = false;
}

WhisperGenerationConfig::WhisperGenerationConfig(const std::filesystem::path& json_path)
    : GenerationConfig::GenerationConfig(json_path) {
    using ov::genai::utils::read_json_param;

    std::ifstream f(json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '", json_path, "' with generation config");

    nlohmann::json data = nlohmann::json::parse(f);

    read_json_param(data, "begin_suppress_tokens", begin_suppress_tokens);
    read_json_param(data, "suppress_tokens", suppress_tokens);
    read_json_param(data, "decoder_start_token_id", decoder_start_token_id);
    read_json_param(data, "pad_token_id", pad_token_id);
    read_json_param(data, "no_timestamps_token_id", no_timestamps_token_id);
    read_json_param(data, "max_initial_timestamp_index", max_initial_timestamp_index);
    read_json_param(data, "prev_sot_token_id", prev_sot_token_id);

    read_json_param(data, "is_multilingual", is_multilingual);
    if (is_multilingual) {
        read_json_param(data, "task_to_id.transcribe", transcribe_token_id);
        read_json_param(data, "task_to_id.translate", translate_token_id);
    }

    read_json_param(data, "lang_to_id", lang_to_id);

    apply_chat_template = false;
}

void WhisperGenerationConfig::update_generation_config(const ov::AnyMap& config_map) {
    using ov::genai::utils::read_anymap_param;

    read_anymap_param(config_map, "begin_suppress_tokens", begin_suppress_tokens);
    read_anymap_param(config_map, "suppress_tokens", suppress_tokens);
    read_anymap_param(config_map, "decoder_start_token_id", decoder_start_token_id);
    read_anymap_param(config_map, "pad_token_id", pad_token_id);
    read_anymap_param(config_map, "transcribe_token_id", transcribe_token_id);
    read_anymap_param(config_map, "translate_token_id", translate_token_id);
    read_anymap_param(config_map, "no_timestamps_token_id", no_timestamps_token_id);
    read_anymap_param(config_map, "max_initial_timestamp_index", max_initial_timestamp_index);
    read_anymap_param(config_map, "is_multilingual", is_multilingual);
    read_anymap_param(config_map, "language", language);
    read_anymap_param(config_map, "lang_to_id", lang_to_id);
    read_anymap_param(config_map, "task", task);
    read_anymap_param(config_map, "return_timestamps", return_timestamps);
    read_anymap_param(config_map, "initial_prompt", initial_prompt);
    read_anymap_param(config_map, "hotwords", hotwords);

    GenerationConfig::update_generation_config(config_map);
}

void WhisperGenerationConfig::validate() const {
    GenerationConfig::validate();

    if (is_multilingual && language.has_value()) {
        OPENVINO_ASSERT(lang_to_id.count(*language),
                        "'language' " + *language + " must be provided in generation_config.json 'lang_to_id' map.");
    }

    if (is_multilingual && task.has_value()) {
        OPENVINO_ASSERT(*task == "transcribe" || *task == "translate",
                        "'task' mast be 'transcribe' or 'translate'. Task provided: '",
                        *task,
                        "'.");
    }

    if (!is_multilingual) {
        OPENVINO_ASSERT(!language.has_value(), "Cannot specify 'language' for not multilingual model.");
        OPENVINO_ASSERT(!task.has_value(), "Cannot specify 'task' for not multilingual model.");
    }

    OPENVINO_ASSERT(num_return_sequences == 1,
                    "'num_return_sequences' must be 1. Provided: ",
                    num_return_sequences,
                    ".");

    OPENVINO_ASSERT(!is_assisting_generation(), "Assisted generation is not supported.");
}
}  // namespace genai
}  // namespace ov
