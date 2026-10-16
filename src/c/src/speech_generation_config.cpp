// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/speech_generation_config.h"

#include <cstring>
#include <filesystem>

#include "types_c.h"

namespace {
ov_status_e copy_string(const std::string& value, char* buffer, size_t* size) {
    if (!size)
        return ov_status_e::INVALID_C_PARAM;
    const size_t required = value.size() + 1;
    if (!buffer) {
        *size = required;
        return ov_status_e::OK;
    }
    if (*size < required)
        return ov_status_e::OUT_OF_BOUNDS;
    std::memcpy(buffer, value.c_str(), required);
    *size = required;
    return ov_status_e::OK;
}
}  // namespace

ov_status_e ov_genai_speech_generation_config_create(ov_genai_speech_generation_config** config) {
    if (!config)
        return ov_status_e::INVALID_C_PARAM;
    try {
        auto result = std::make_unique<ov_genai_speech_generation_config>();
        result->object = std::make_shared<ov::genai::SpeechGenerationConfig>();
        *config = result.release();
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_speech_generation_config_create_from_json(const char* json_path,
                                                               ov_genai_speech_generation_config** config) {
    if (!json_path || !config)
        return ov_status_e::INVALID_C_PARAM;
    try {
        auto result = std::make_unique<ov_genai_speech_generation_config>();
        result->object = std::make_shared<ov::genai::SpeechGenerationConfig>(std::filesystem::path(json_path));
        *config = result.release();
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

void ov_genai_speech_generation_config_free(ov_genai_speech_generation_config* config) {
    delete config;
}

ov_status_e ov_genai_speech_generation_config_validate(const ov_genai_speech_generation_config* config) {
    if (!config || !config->object)
        return ov_status_e::INVALID_C_PARAM;
    try {
        config->object->validate();
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

#define CONFIG_NUMBER_ACCESSORS(name, type)                                                                           \
    ov_status_e ov_genai_speech_generation_config_set_##name(ov_genai_speech_generation_config* config, type value) { \
        if (!config || !config->object)                                                                               \
            return ov_status_e::INVALID_C_PARAM;                                                                      \
        config->object->name = value;                                                                                 \
        return ov_status_e::OK;                                                                                       \
    }                                                                                                                 \
    ov_status_e ov_genai_speech_generation_config_get_##name(const ov_genai_speech_generation_config* config,         \
                                                             type* value) {                                           \
        if (!config || !config->object || !value)                                                                     \
            return ov_status_e::INVALID_C_PARAM;                                                                      \
        *value = config->object->name;                                                                                \
        return ov_status_e::OK;                                                                                       \
    }

CONFIG_NUMBER_ACCESSORS(minlenratio, float)
CONFIG_NUMBER_ACCESSORS(maxlenratio, float)
CONFIG_NUMBER_ACCESSORS(threshold, float)
CONFIG_NUMBER_ACCESSORS(speed, float)
CONFIG_NUMBER_ACCESSORS(max_phoneme_length, uint32_t)
#undef CONFIG_NUMBER_ACCESSORS

ov_status_e ov_genai_speech_generation_config_set_language(ov_genai_speech_generation_config* config,
                                                           const char* language) {
    if (!config || !config->object || !language)
        return ov_status_e::INVALID_C_PARAM;
    try {
        config->object->language = language;
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_speech_generation_config_get_language(const ov_genai_speech_generation_config* config,
                                                           char* buffer,
                                                           size_t* size) {
    if (!config || !config->object)
        return ov_status_e::INVALID_C_PARAM;
    return copy_string(config->object->language, buffer, size);
}

ov_status_e ov_genai_speech_generation_config_set_phonemize_fallback_model_dir(
    ov_genai_speech_generation_config* config,
    const char* path) {
    if (!config || !config->object)
        return ov_status_e::INVALID_C_PARAM;
    try {
        if (path)
            config->object->phonemize_fallback_model_dir = std::filesystem::path(path);
        else
            config->object->phonemize_fallback_model_dir = std::nullopt;
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_speech_generation_config_get_phonemize_fallback_model_dir(
    const ov_genai_speech_generation_config* config,
    char* buffer,
    size_t* size) {
    if (!config || !config->object)
        return ov_status_e::INVALID_C_PARAM;
    if (!config->object->phonemize_fallback_model_dir)
        return ov_status_e::NOT_FOUND;
    try {
        return copy_string(config->object->phonemize_fallback_model_dir->string(), buffer, size);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}
