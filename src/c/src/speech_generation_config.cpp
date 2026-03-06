// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/speech_generation_config.h"
#include "openvino/genai/speech_generation/speech_generation_config.hpp"

#include <filesystem>
#include <stdexcept>

struct speech_generation_config_t {
    ov::genai::SpeechGenerationConfig impl;
};

int speech_generation_config_create(speech_generation_config_handle_t* config) {
    if (!config) {
        return 1;  // Invalid argument
    }

    try {
        *config = new speech_generation_config_t{ov::genai::SpeechGenerationConfig()};
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int speech_generation_config_create_from_json(speech_generation_config_handle_t* config, const char* json_path) {
    if (!config || !json_path) {
        return 1;  // Invalid argument
    }

    try {
        *config = new speech_generation_config_t{
            ov::genai::SpeechGenerationConfig(std::filesystem::path(json_path))};
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int speech_generation_config_destroy(speech_generation_config_handle_t config) {
    if (!config) {
        return 1;  // Invalid argument
    }

    try {
        delete config;
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int speech_generation_config_set_minlenratio(speech_generation_config_handle_t config, float value) {
    if (!config) {
        return 1;  // Invalid argument
    }

    try {
        config->impl.minlenratio = value;
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int speech_generation_config_get_minlenratio(speech_generation_config_handle_t config, float* value) {
    if (!config || !value) {
        return 1;  // Invalid argument
    }

    try {
        *value = config->impl.minlenratio;
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int speech_generation_config_set_maxlenratio(speech_generation_config_handle_t config, float value) {
    if (!config) {
        return 1;  // Invalid argument
    }

    try {
        config->impl.maxlenratio = value;
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int speech_generation_config_get_maxlenratio(speech_generation_config_handle_t config, float* value) {
    if (!config || !value) {
        return 1;  // Invalid argument
    }

    try {
        *value = config->impl.maxlenratio;
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int speech_generation_config_set_threshold(speech_generation_config_handle_t config, float value) {
    if (!config) {
        return 1;  // Invalid argument
    }

    try {
        config->impl.threshold = value;
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int speech_generation_config_get_threshold(speech_generation_config_handle_t config, float* value) {
    if (!config || !value) {
        return 1;  // Invalid argument
    }

    try {
        *value = config->impl.threshold;
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}
