// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/generation_config.h"

#include "openvino/genai/generation_config.hpp"
#include "types_c.h"

ov_status_e ov_genai_generation_config_create(ov_genai_generation_config** config) {
    if (!config) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_generation_config> _config = std::make_unique<ov_genai_generation_config>();
        _config->object = std::make_shared<ov::genai::GenerationConfig>();
        *config = _config.release();

    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_create_from_json(const char* json_path, ov_genai_generation_config** config) {
    if (!config || !json_path) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_generation_config> _config = std::make_unique<ov_genai_generation_config>();
        _config->object = std::make_shared<ov::genai::GenerationConfig>(std::filesystem::path(json_path));
        *config = _config.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
void ov_genai_generation_config_free(ov_genai_generation_config* config) {
    if (config) {
        delete config;
    }
}

ov_status_e ov_genai_generation_config_set_max_new_tokens(ov_genai_generation_config* config, const size_t value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->max_new_tokens = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_max_length(ov_genai_generation_config* config, const size_t value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->max_length = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_ignore_eos(ov_genai_generation_config* config, const bool value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->ignore_eos = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_min_new_tokens(ov_genai_generation_config* config, const size_t value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->min_new_tokens = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_echo(ov_genai_generation_config* config, const bool value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->echo = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_logprobs(ov_genai_generation_config* config, const size_t value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->logprobs = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_generation_config_set_include_stop_str_in_output(ov_genai_generation_config* config,
                                                                      const bool value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->include_stop_str_in_output = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_stop_strings(ov_genai_generation_config* config,
                                                        const char** strings,
                                                        const size_t count) {
    if (!config || !(config->object) || !strings) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::set<std::string> stopStrings;
        for (size_t i = 0; i < count; i++) {
            if (strings[i])
                stopStrings.insert(strings[i]);
        }
        config->object->stop_strings = stopStrings;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_stop_token_ids(ov_genai_generation_config* config,
                                                          const int64_t* token_ids,
                                                          const size_t token_ids_num) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::set<int64_t> stop_token_ids;
        for (size_t i = 0; i < token_ids_num; i++) {
            stop_token_ids.insert(token_ids[i]);
        }
        config->object->stop_token_ids = stop_token_ids;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_num_beam_groups(ov_genai_generation_config* config, const size_t value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->num_beam_groups = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_num_beams(ov_genai_generation_config* config, const size_t value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->num_beams = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_diversity_penalty(ov_genai_generation_config* config, const float value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->diversity_penalty = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_length_penalty(ov_genai_generation_config* config, const float value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->length_penalty = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_num_return_sequences(ov_genai_generation_config* config,
                                                                const size_t value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->num_return_sequences = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_no_repeat_ngram_size(ov_genai_generation_config* config,
                                                                const size_t value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->no_repeat_ngram_size = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_generation_config_set_stop_criteria(ov_genai_generation_config* config, const StopCriteria value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->stop_criteria = static_cast<ov::genai::StopCriteria>(value);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_generation_config_set_temperature(ov_genai_generation_config* config, const float value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->temperature = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_top_p(ov_genai_generation_config* config, const float value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->top_p = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_top_k(ov_genai_generation_config* config, const size_t value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->top_k = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_do_sample(ov_genai_generation_config* config, const bool value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->do_sample = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_repetition_penalty(ov_genai_generation_config* config, const float value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->repetition_penalty = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_presence_penalty(ov_genai_generation_config* config, const float value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->presence_penalty = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_frequency_penalty(ov_genai_generation_config* config, const float value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->frequency_penalty = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_rng_seed(ov_genai_generation_config* config, const size_t value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->rng_seed = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_generation_config_set_assistant_confidence_threshold(ov_genai_generation_config* config,
                                                                          const float value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->assistant_confidence_threshold = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_num_assistant_tokens(ov_genai_generation_config* config,
                                                                const size_t value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->num_assistant_tokens = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_set_max_ngram_size(ov_genai_generation_config* config, const size_t value) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->max_ngram_size = value;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_generation_config_set_eos_token_id(ov_genai_generation_config* config, const int64_t id) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->eos_token_id = id;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_generation_config_get_max_new_tokens(const ov_genai_generation_config* config,
                                                          size_t* max_new_tokens) {
    if (!config || !(config->object) || !max_new_tokens) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *max_new_tokens = config->object->max_new_tokens;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_generation_config_validate(ov_genai_generation_config* config) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->validate();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
