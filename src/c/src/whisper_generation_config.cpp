// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/whisper_generation_config.h"

#include <filesystem>

#include "openvino/genai/whisper_generation_config.hpp"
#include "types_c.h"

ov_status_e ov_genai_whisper_generation_config_create(ov_genai_whisper_generation_config** config) {
    if (!config) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_whisper_generation_config> _config =
            std::make_unique<ov_genai_whisper_generation_config>();
        _config->object = std::make_shared<ov::genai::WhisperGenerationConfig>();
        if (!_config->object) {
            return ov_status_e::UNKNOW_EXCEPTION;
        }
        _config->object->max_length = 448;
        *config = _config.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_create_from_json(const char* json_path,
                                                                ov_genai_whisper_generation_config** config) {
    if (!config || !json_path) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_whisper_generation_config> _config =
            std::make_unique<ov_genai_whisper_generation_config>();
        _config->object = std::make_shared<ov::genai::WhisperGenerationConfig>(std::filesystem::path(json_path));
        *config = _config.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_whisper_generation_config_free(ov_genai_whisper_generation_config* config) {
    if (config) {
        delete config;
    }
}

ov_status_e ov_genai_whisper_generation_config_get_generation_config(const ov_genai_whisper_generation_config* config,
                                                                     ov_genai_generation_config** generation_config) {
    if (!config || !(config->object) || !generation_config) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_generation_config> _generation_config = std::make_unique<ov_genai_generation_config>();
        _generation_config->object = std::make_shared<ov::genai::GenerationConfig>(*(config->object));
        *generation_config = _generation_config.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_set_decoder_start_token_id(ov_genai_whisper_generation_config* config,
                                                                          int64_t token_id) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->decoder_start_token_id = token_id;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_decoder_start_token_id(
    const ov_genai_whisper_generation_config* config,
    int64_t* token_id) {
    if (!config || !(config->object) || !token_id) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *token_id = config->object->decoder_start_token_id;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_set_pad_token_id(ov_genai_whisper_generation_config* config,
                                                                int64_t token_id) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->pad_token_id = token_id;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_pad_token_id(const ov_genai_whisper_generation_config* config,
                                                                int64_t* token_id) {
    if (!config || !(config->object) || !token_id) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *token_id = config->object->pad_token_id;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_set_translate_token_id(ov_genai_whisper_generation_config* config,
                                                                      int64_t token_id) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->translate_token_id = token_id;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_translate_token_id(const ov_genai_whisper_generation_config* config,
                                                                      int64_t* token_id) {
    if (!config || !(config->object) || !token_id) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *token_id = config->object->translate_token_id;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_set_transcribe_token_id(ov_genai_whisper_generation_config* config,
                                                                       int64_t token_id) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->transcribe_token_id = token_id;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_transcribe_token_id(const ov_genai_whisper_generation_config* config,
                                                                       int64_t* token_id) {
    if (!config || !(config->object) || !token_id) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *token_id = config->object->transcribe_token_id;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_set_prev_sot_token_id(ov_genai_whisper_generation_config* config,
                                                                     int64_t token_id) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->prev_sot_token_id = token_id;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_prev_sot_token_id(const ov_genai_whisper_generation_config* config,
                                                                     int64_t* token_id) {
    if (!config || !(config->object) || !token_id) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *token_id = config->object->prev_sot_token_id;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_set_no_timestamps_token_id(ov_genai_whisper_generation_config* config,
                                                                          int64_t token_id) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->no_timestamps_token_id = token_id;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_no_timestamps_token_id(
    const ov_genai_whisper_generation_config* config,
    int64_t* token_id) {
    if (!config || !(config->object) || !token_id) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *token_id = config->object->no_timestamps_token_id;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_set_max_initial_timestamp_index(
    ov_genai_whisper_generation_config* config,
    size_t index) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->max_initial_timestamp_index = index;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_max_initial_timestamp_index(
    const ov_genai_whisper_generation_config* config,
    size_t* index) {
    if (!config || !(config->object) || !index) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *index = config->object->max_initial_timestamp_index;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_set_is_multilingual(ov_genai_whisper_generation_config* config,
                                                                   bool is_multilingual) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->is_multilingual = is_multilingual;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_is_multilingual(const ov_genai_whisper_generation_config* config,
                                                                   bool* is_multilingual) {
    if (!config || !(config->object) || !is_multilingual) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *is_multilingual = config->object->is_multilingual;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_set_language(ov_genai_whisper_generation_config* config,
                                                            const char* language) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        if (language) {
            config->object->language = std::string(language);
        } else {
            config->object->language = std::nullopt;
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_language(const ov_genai_whisper_generation_config* config,
                                                            char* language,
                                                            size_t* language_size) {
    if (!config || !(config->object) || !language_size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        if (!config->object->language.has_value()) {
            return ov_status_e::NOT_FOUND;
        }

        const std::string& str = config->object->language.value();
        if (!language) {
            *language_size = str.length() + 1;
        } else {
            if (*language_size < str.length() + 1) {
                return ov_status_e::OUT_OF_BOUNDS;
            }
            strncpy(language, str.c_str(), str.length() + 1);
            language[str.length()] = '\0';
            *language_size = str.length() + 1;
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_set_task(ov_genai_whisper_generation_config* config, const char* task) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        if (task) {
            config->object->task = std::string(task);
        } else {
            config->object->task = std::nullopt;
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_task(const ov_genai_whisper_generation_config* config,
                                                        char* task,
                                                        size_t* task_size) {
    if (!config || !(config->object) || !task_size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        if (!config->object->task.has_value()) {
            return ov_status_e::NOT_FOUND;
        }

        const std::string& str = config->object->task.value();
        if (!task) {
            *task_size = str.length() + 1;
        } else {
            if (*task_size < str.length() + 1) {
                return ov_status_e::OUT_OF_BOUNDS;
            }
            strncpy(task, str.c_str(), str.length() + 1);
            task[str.length()] = '\0';
            *task_size = str.length() + 1;
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_set_return_timestamps(ov_genai_whisper_generation_config* config,
                                                                     bool return_timestamps) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->return_timestamps = return_timestamps;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_return_timestamps(const ov_genai_whisper_generation_config* config,
                                                                     bool* return_timestamps) {
    if (!config || !(config->object) || !return_timestamps) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *return_timestamps = config->object->return_timestamps;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_set_initial_prompt(ov_genai_whisper_generation_config* config,
                                                                  const char* initial_prompt) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        if (initial_prompt) {
            config->object->initial_prompt = std::string(initial_prompt);
        } else {
            config->object->initial_prompt = std::nullopt;
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_initial_prompt(const ov_genai_whisper_generation_config* config,
                                                                  char* initial_prompt,
                                                                  size_t* prompt_size) {
    if (!config || !(config->object) || !prompt_size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        if (!config->object->initial_prompt.has_value()) {
            return ov_status_e::NOT_FOUND;
        }

        const std::string& str = config->object->initial_prompt.value();
        if (!initial_prompt) {
            *prompt_size = str.length() + 1;
        } else {
            if (*prompt_size < str.length() + 1) {
                return ov_status_e::OUT_OF_BOUNDS;
            }
            strncpy(initial_prompt, str.c_str(), str.length() + 1);
            initial_prompt[str.length()] = '\0';
            *prompt_size = str.length() + 1;
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_set_hotwords(ov_genai_whisper_generation_config* config,
                                                            const char* hotwords) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        if (hotwords) {
            config->object->hotwords = std::string(hotwords);
        } else {
            config->object->hotwords = std::nullopt;
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_hotwords(const ov_genai_whisper_generation_config* config,
                                                            char* hotwords,
                                                            size_t* hotwords_size) {
    if (!config || !(config->object) || !hotwords_size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        if (!config->object->hotwords.has_value()) {
            return ov_status_e::NOT_FOUND;
        }

        const std::string& str = config->object->hotwords.value();
        if (!hotwords) {
            *hotwords_size = str.length() + 1;
        } else {
            if (*hotwords_size < str.length() + 1) {
                return ov_status_e::OUT_OF_BOUNDS;
            }
            strncpy(hotwords, str.c_str(), str.length() + 1);
            hotwords[str.length()] = '\0';
            *hotwords_size = str.length() + 1;
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_set_begin_suppress_tokens(ov_genai_whisper_generation_config* config,
                                                                         const int64_t* tokens,
                                                                         size_t tokens_count) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->begin_suppress_tokens.clear();
        if (tokens && tokens_count > 0) {
            config->object->begin_suppress_tokens.assign(tokens, tokens + tokens_count);
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_begin_suppress_tokens_count(
    const ov_genai_whisper_generation_config* config,
    size_t* tokens_count) {
    if (!config || !(config->object) || !tokens_count) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *tokens_count = config->object->begin_suppress_tokens.size();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_begin_suppress_tokens(
    const ov_genai_whisper_generation_config* config,
    int64_t* tokens,
    size_t tokens_count) {
    if (!config || !(config->object) || !tokens) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        if (tokens_count < config->object->begin_suppress_tokens.size()) {
            return ov_status_e::OUT_OF_BOUNDS;
        }
        std::copy(config->object->begin_suppress_tokens.begin(), config->object->begin_suppress_tokens.end(), tokens);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_set_suppress_tokens(ov_genai_whisper_generation_config* config,
                                                                   const int64_t* tokens,
                                                                   size_t tokens_count) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->suppress_tokens.clear();
        if (tokens && tokens_count > 0) {
            config->object->suppress_tokens.assign(tokens, tokens + tokens_count);
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_suppress_tokens_count(
    const ov_genai_whisper_generation_config* config,
    size_t* tokens_count) {
    if (!config || !(config->object) || !tokens_count) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *tokens_count = config->object->suppress_tokens.size();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_get_suppress_tokens(const ov_genai_whisper_generation_config* config,
                                                                   int64_t* tokens,
                                                                   size_t tokens_count) {
    if (!config || !(config->object) || !tokens) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        if (tokens_count < config->object->suppress_tokens.size()) {
            return ov_status_e::OUT_OF_BOUNDS;
        }
        std::copy(config->object->suppress_tokens.begin(), config->object->suppress_tokens.end(), tokens);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_generation_config_validate(ov_genai_whisper_generation_config* config) {
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
