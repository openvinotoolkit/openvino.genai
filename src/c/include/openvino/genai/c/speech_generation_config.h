// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "openvino/c/ov_common.h"
#include "openvino/genai/c/visibility.h"

/** Configuration for SpeechT5 and Kokoro text-to-speech generation. */
typedef struct ov_genai_speech_generation_config_opaque ov_genai_speech_generation_config;

OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_create(ov_genai_speech_generation_config** config);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_create_from_json(const char* json_path, ov_genai_speech_generation_config** config);
OPENVINO_GENAI_C_EXPORTS void ov_genai_speech_generation_config_free(ov_genai_speech_generation_config* config);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_validate(const ov_genai_speech_generation_config* config);

OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_minlenratio(ov_genai_speech_generation_config* config, float value);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_minlenratio(const ov_genai_speech_generation_config* config, float* value);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_maxlenratio(ov_genai_speech_generation_config* config, float value);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_maxlenratio(const ov_genai_speech_generation_config* config, float* value);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_threshold(ov_genai_speech_generation_config* config, float value);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_threshold(const ov_genai_speech_generation_config* config, float* value);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_speed(ov_genai_speech_generation_config* config, float value);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_speed(const ov_genai_speech_generation_config* config, float* value);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_max_phoneme_length(ov_genai_speech_generation_config* config, uint32_t value);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_max_phoneme_length(const ov_genai_speech_generation_config* config,
                                                         uint32_t* value);

/** NULL clears the optional fallback model directory. */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_phonemize_fallback_model_dir(ov_genai_speech_generation_config* config,
                                                                   const char* path);
/** Query the required byte count with buffer=NULL; caller owns the buffer. Returns NOT_FOUND when unset. */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_phonemize_fallback_model_dir(const ov_genai_speech_generation_config* config,
                                                                   char* buffer,
                                                                   size_t* size);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_language(ov_genai_speech_generation_config* config, const char* language);
/** Query the required byte count with buffer=NULL; caller owns the buffer. */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_language(const ov_genai_speech_generation_config* config,
                                               char* buffer,
                                               size_t* size);
