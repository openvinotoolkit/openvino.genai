// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/speech_generation_config.h"

#include <cstring>
#include <filesystem>
#include <vector>

#include "openvino/runtime/tensor.hpp"
#include "types_c.h"

namespace {
struct shape_guard {
    ov_shape_t shape{};
    ~shape_guard() {
        ov_shape_free(&shape);
    }
};

struct tensor_guard {
    ov_tensor_t* tensor = nullptr;
    ~tensor_guard() {
        if (tensor)
            ov_tensor_free(tensor);
    }
};

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


ov_status_e copy_c_tensor_to_cpp(const ov_tensor_t* source,
                                 ov_element_type_e expected_type,
                                 const ov::element::Type& target_type,
                                 ov::Tensor& destination) {
    if (!source)
        return ov_status_e::INVALID_C_PARAM;
    ov_element_type_e source_type;
    auto status = ov_tensor_get_element_type(source, &source_type);
    if (status != ov_status_e::OK)
        return status;
    if (source_type != expected_type)
        return ov_status_e::INVALID_C_PARAM;

    shape_guard guard;
    status = ov_tensor_get_shape(source, &guard.shape);
    if (status != ov_status_e::OK)
        return status;

    ov::Shape shape;
    shape.reserve(guard.shape.rank);
    for (size_t i = 0; i < guard.shape.rank; ++i)
        shape.push_back(static_cast<size_t>(guard.shape.dims[i]));

    destination = ov::Tensor(target_type, shape);

    void* data = nullptr;
    status = ov_tensor_data(const_cast<ov_tensor_t*>(source), &data);
    if (status != ov_status_e::OK)
        return status;

    const size_t byte_size = destination.get_byte_size();
    if (!data && byte_size > 0)
        return ov_status_e::INVALID_C_PARAM;
    if (byte_size > 0)
        std::memcpy(destination.data(), data, byte_size);
    return ov_status_e::OK;
}

ov_status_e copy_cpp_tensor_to_c(const ov::Tensor& source,
                                 ov_element_type_e expected_type,
                                 const ov::element::Type& source_type,
                                 ov_tensor_t** destination) {
    if (!destination)
        return ov_status_e::INVALID_C_PARAM;
    if (!source)
        return ov_status_e::NOT_FOUND;
    if (source.get_element_type() != source_type)
        return ov_status_e::INVALID_C_PARAM;

    const auto& dims = source.get_shape();
    std::vector<int64_t> c_dims(dims.begin(), dims.end());

    shape_guard guard;
    auto status = ov_shape_create(c_dims.size(), c_dims.data(), &guard.shape);
    if (status != ov_status_e::OK)
        return status;

    tensor_guard output;
    status = ov_tensor_create(expected_type, guard.shape, &output.tensor);
    if (status != ov_status_e::OK)
        return status;

    ov::Tensor snapshot(source_type, source.get_shape());
    source.copy_to(snapshot);

    void* output_data = nullptr;
    status = ov_tensor_data(output.tensor, &output_data);
    if (status != ov_status_e::OK)
        return status;

    const size_t byte_size = snapshot.get_byte_size();
    if (!output_data && byte_size > 0)
        return ov_status_e::UNKNOW_EXCEPTION;
    if (byte_size > 0)
        std::memcpy(output_data, snapshot.data(), byte_size);

    *destination = output.tensor;
    output.tensor = nullptr;
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

ov_status_e ov_genai_speech_generation_config_get_generation_config(const ov_genai_speech_generation_config* config,
                                                                    ov_genai_generation_config** generation_config) {
    if (!config || !config->object || !generation_config)
        return ov_status_e::INVALID_C_PARAM;
    try {
        auto result = std::make_unique<ov_genai_generation_config>();
        result->object = std::make_shared<ov::genai::GenerationConfig>(*(config->object));
        *generation_config = result.release();
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_speech_generation_config_set_generation_config(ov_genai_speech_generation_config* config,
                                                                    const ov_genai_generation_config* generation_config) {
    if (!config || !config->object || !generation_config || !generation_config->object)
        return ov_status_e::INVALID_C_PARAM;
    try {
        *std::static_pointer_cast<ov::genai::GenerationConfig>(config->object) = *(generation_config->object);
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
CONFIG_NUMBER_ACCESSORS(non_streaming_mode, bool)
CONFIG_NUMBER_ACCESSORS(subtalker_dosample, bool)
CONFIG_NUMBER_ACCESSORS(subtalker_top_k, uint32_t)
CONFIG_NUMBER_ACCESSORS(subtalker_top_p, float)
CONFIG_NUMBER_ACCESSORS(subtalker_temperature, float)
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

ov_status_e ov_genai_speech_generation_config_set_speaker(ov_genai_speech_generation_config* config,
                                                          const char* speaker) {
    if (!config || !config->object || !speaker)
        return ov_status_e::INVALID_C_PARAM;
    try {
        config->object->speaker = speaker;
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_speech_generation_config_get_speaker(const ov_genai_speech_generation_config* config,
                                                          char* buffer,
                                                          size_t* size) {
    if (!config || !config->object)
        return ov_status_e::INVALID_C_PARAM;
    return copy_string(config->object->speaker, buffer, size);
}

ov_status_e ov_genai_speech_generation_config_set_instruct(ov_genai_speech_generation_config* config,
                                                           const char* instruct) {
    if (!config || !config->object || !instruct)
        return ov_status_e::INVALID_C_PARAM;
    try {
        config->object->instruct = instruct;
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_speech_generation_config_get_instruct(const ov_genai_speech_generation_config* config,
                                                           char* buffer,
                                                           size_t* size) {
    if (!config || !config->object)
        return ov_status_e::INVALID_C_PARAM;
    return copy_string(config->object->instruct, buffer, size);
}

ov_status_e ov_genai_speech_generation_config_set_ref_text(ov_genai_speech_generation_config* config,
                                                           const char* ref_text) {
    if (!config || !config->object || !ref_text)
        return ov_status_e::INVALID_C_PARAM;
    try {
        config->object->ref_text = ref_text;
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_speech_generation_config_get_ref_text(const ov_genai_speech_generation_config* config,
                                                           char* buffer,
                                                           size_t* size) {
    if (!config || !config->object)
        return ov_status_e::INVALID_C_PARAM;
    return copy_string(config->object->ref_text, buffer, size);
}

ov_status_e ov_genai_speech_generation_config_set_ref_audio(ov_genai_speech_generation_config* config,
                                                            const ov_tensor_t* ref_audio) {
    if (!config || !config->object || !ref_audio)
        return ov_status_e::INVALID_C_PARAM;
    try {
        ov::Tensor converted;
        const auto status = copy_c_tensor_to_cpp(ref_audio, ov_element_type_e::F32, ov::element::f32, converted);
        if (status != ov_status_e::OK)
            return status;
        config->object->ref_audio = std::move(converted);
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_speech_generation_config_get_ref_audio(const ov_genai_speech_generation_config* config,
                                                            ov_tensor_t** ref_audio) {
    if (!config || !config->object)
        return ov_status_e::INVALID_C_PARAM;
    try {
        return copy_cpp_tensor_to_c(config->object->ref_audio, ov_element_type_e::F32, ov::element::f32, ref_audio);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_speech_generation_config_set_ref_codec_ids(ov_genai_speech_generation_config* config,
                                                                const ov_tensor_t* ref_codec_ids) {
    if (!config || !config->object)
        return ov_status_e::INVALID_C_PARAM;
    try {
        if (!ref_codec_ids) {
            config->object->ref_codec_ids = ov::Tensor();
            return ov_status_e::OK;
        }
        ov::Tensor converted;
        const auto status = copy_c_tensor_to_cpp(ref_codec_ids,
                                                 ov_element_type_e::I64,
                                                 ov::element::i64,
                                                 converted);
        if (status != ov_status_e::OK)
            return status;
        config->object->ref_codec_ids = std::move(converted);
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_speech_generation_config_get_ref_codec_ids(const ov_genai_speech_generation_config* config,
                                                                ov_tensor_t** ref_codec_ids) {
    if (!config || !config->object)
        return ov_status_e::INVALID_C_PARAM;
    try {
        return copy_cpp_tensor_to_c(config->object->ref_codec_ids,
                                    ov_element_type_e::I64,
                                    ov::element::i64,
                                    ref_codec_ids);
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
