// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/genai/c/speech_generation_config.h"

#include <filesystem>
#include <memory>

#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "types_c.h"

ov_status_e ov_genai_speech_generation_config_create(ov_genai_speech_generation_config** config) {
    if (!config) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_speech_generation_config> _config =
            std::make_unique<ov_genai_speech_generation_config>();
        _config->object = std::make_shared<ov::genai::SpeechGenerationConfig>();
        *config = _config.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_speech_generation_config_create_from_json(const char* json_path,
                                                               ov_genai_speech_generation_config** config) {
    if (!config || !json_path) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_speech_generation_config> _config =
            std::make_unique<ov_genai_speech_generation_config>();
        _config->object = std::make_shared<ov::genai::SpeechGenerationConfig>(std::filesystem::path(json_path));
        *config = _config.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_speech_generation_config_free(ov_genai_speech_generation_config* config) {
    if (config) {
        delete config;
    }
}

ov_status_e ov_genai_speech_generation_config_set_minlenratio(ov_genai_speech_generation_config* config,
                                                              float minlenratio) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->minlenratio = minlenratio;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_speech_generation_config_get_minlenratio(const ov_genai_speech_generation_config* config,
                                                              float* minlenratio) {
    if (!config || !(config->object) || !minlenratio) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *minlenratio = config->object->minlenratio;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_speech_generation_config_set_maxlenratio(ov_genai_speech_generation_config* config,
                                                              float maxlenratio) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->maxlenratio = maxlenratio;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_speech_generation_config_get_maxlenratio(const ov_genai_speech_generation_config* config,
                                                              float* maxlenratio) {
    if (!config || !(config->object) || !maxlenratio) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *maxlenratio = config->object->maxlenratio;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_speech_generation_config_set_threshold(ov_genai_speech_generation_config* config,
                                                            float threshold) {
    if (!config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->threshold = threshold;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_speech_generation_config_get_threshold(const ov_genai_speech_generation_config* config,
                                                            float* threshold) {
    if (!config || !(config->object) || !threshold) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *threshold = config->object->threshold;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
