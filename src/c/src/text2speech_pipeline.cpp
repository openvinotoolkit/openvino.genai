// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/text2speech_pipeline.h"

#include <cstdarg>
#include <cstring>
#include <filesystem>
#include <vector>

#include <openvino/runtime/remote_tensor.hpp>
#include <openvino/runtime/tensor.hpp>

#include "types_c.h"

namespace {
struct shape_guard {
    ov_shape_t shape{};
    ~shape_guard() {
        ov_shape_free(&shape);
    }
};

ov_status_e convert_embedding(const ov_tensor_t* input, ov::Tensor& output) {
    if (!input)
        return ov_status_e::OK;
    ov_element_type_e type;
    auto status = ov_tensor_get_element_type(input, &type);
    if (status != ov_status_e::OK)
        return status;
    if (type != ov_element_type_e::F32)
        return ov_status_e::INVALID_C_PARAM;
    shape_guard guard;
    status = ov_tensor_get_shape(input, &guard.shape);
    if (status != ov_status_e::OK)
        return status;
    ov::Shape dims;
    for (int64_t i = 0; i < guard.shape.rank; ++i)
        dims.push_back(static_cast<size_t>(guard.shape.dims[i]));
    void* data = nullptr;
    status = ov_tensor_data(input, &data);
    if (status != ov_status_e::OK)
        return status;
    if (!data)
        return ov_status_e::INVALID_C_PARAM;
    output = ov::Tensor(ov::element::f32, dims, data);
    return ov_status_e::OK;
}

ov::AnyMap speech_properties(const ov::genai::SpeechGenerationConfig& config) {
    // Supply every speech setting so an explicit C config replaces any stored speech settings
    // for this request. The C++ generate method applies these to a request-local copy.
    ov::AnyMap properties{{"minlenratio", config.minlenratio},
                          {"maxlenratio", config.maxlenratio},
                          {"threshold", config.threshold},
                          {"speed", config.speed},
                          {"language", config.language},
                          {"max_phoneme_length", config.max_phoneme_length}};
    properties["phonemize_fallback_model_dir"] = config.phonemize_fallback_model_dir
                                                     ? ov::Any(*config.phonemize_fallback_model_dir)
                                                     : ov::Any{};
    return properties;
}
}  // namespace

ov_status_e ov_genai_text2speech_pipeline_create(const char* models_path,
                                                 const char* device,
                                                 size_t property_args_size,
                                                 ov_genai_text2speech_pipeline** pipeline,
                                                 ...) {
    if (!models_path || !device || !pipeline || property_args_size % 2)
        return ov_status_e::INVALID_C_PARAM;
    try {
        ov::AnyMap property;
        va_list args_ptr;
        va_start(args_ptr, pipeline);
        try {
            for (size_t i = 0; i < property_args_size / 2; ++i) {
                GET_PROPERTY_FROM_ARGS_LIST;
            }
        } catch (...) {
            va_end(args_ptr);
            throw;
        }
        va_end(args_ptr);
        auto result = std::make_unique<ov_genai_text2speech_pipeline>();
        result->object = std::make_shared<ov::genai::Text2SpeechPipeline>(std::filesystem::path(models_path),
                                                                          std::string(device),
                                                                          property);
        *pipeline = result.release();
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

void ov_genai_text2speech_pipeline_free(ov_genai_text2speech_pipeline* pipeline) {
    delete pipeline;
}

ov_status_e ov_genai_text2speech_pipeline_generate_batch(ov_genai_text2speech_pipeline* pipeline,
                                                         const char* const* texts,
                                                         size_t count,
                                                         const ov_tensor_t* speaker_embedding,
                                                         const ov_genai_speech_generation_config* config,
                                                         ov_genai_text2speech_decoded_results** results) {
    if (!pipeline || !pipeline->object || !texts || !count || !results)
        return ov_status_e::INVALID_C_PARAM;
    if (config && !config->object)
        return ov_status_e::INVALID_C_PARAM;
    try {
        std::vector<std::string> inputs;
        inputs.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            if (!texts[i])
                return ov_status_e::INVALID_C_PARAM;
            inputs.emplace_back(texts[i]);
        }
        ov::Tensor embedding;
        auto status = convert_embedding(speaker_embedding, embedding);
        if (status != ov_status_e::OK)
            return status;
        auto result = std::make_unique<ov_genai_text2speech_decoded_results>();
        if (config && config->object) {
            config->object->validate();
            const ov::AnyMap properties = speech_properties(*config->object);
            result->object = std::make_shared<ov::genai::Text2SpeechDecodedResults>(
                pipeline->object->generate(inputs, embedding, properties));
        } else {
            result->object = std::make_shared<ov::genai::Text2SpeechDecodedResults>(
                pipeline->object->generate(inputs, embedding));
        }
        *results = result.release();
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_text2speech_pipeline_generate(ov_genai_text2speech_pipeline* pipeline,
                                                   const char* text,
                                                   const ov_tensor_t* speaker_embedding,
                                                   const ov_genai_speech_generation_config* config,
                                                   ov_genai_text2speech_decoded_results** results) {
    if (!text)
        return ov_status_e::INVALID_C_PARAM;
    const char* texts[] = {text};
    return ov_genai_text2speech_pipeline_generate_batch(pipeline, texts, 1, speaker_embedding, config, results);
}

ov_status_e ov_genai_text2speech_pipeline_get_generation_config(const ov_genai_text2speech_pipeline* pipeline,
                                                                ov_genai_speech_generation_config** config) {
    if (!pipeline || !pipeline->object || !config)
        return ov_status_e::INVALID_C_PARAM;
    try {
        auto result = std::make_unique<ov_genai_speech_generation_config>();
        result->object = std::make_shared<ov::genai::SpeechGenerationConfig>(pipeline->object->get_generation_config());
        *config = result.release();
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_text2speech_pipeline_set_generation_config(ov_genai_text2speech_pipeline* pipeline,
                                                                const ov_genai_speech_generation_config* config) {
    if (!pipeline || !pipeline->object || !config || !config->object)
        return ov_status_e::INVALID_C_PARAM;
    try {
        pipeline->object->set_generation_config(*config->object);
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_text2speech_pipeline_get_speaker_embedding_shape(const ov_genai_text2speech_pipeline* pipeline,
                                                                      ov_shape_t* shape) {
    if (!pipeline || !pipeline->object || !shape)
        return ov_status_e::INVALID_C_PARAM;
    try {
        const auto dims = pipeline->object->get_speaker_embedding_shape();
        std::vector<int64_t> c_dims(dims.begin(), dims.end());
        return ov_shape_create(c_dims.size(), c_dims.data(), shape);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

void ov_genai_text2speech_decoded_results_free(ov_genai_text2speech_decoded_results* results) {
    delete results;
}

ov_status_e ov_genai_text2speech_decoded_results_get_speeches_count(const ov_genai_text2speech_decoded_results* results,
                                                                    size_t* count) {
    if (!results || !results->object || !count)
        return ov_status_e::INVALID_C_PARAM;
    *count = results->object->speeches.size();
    return ov_status_e::OK;
}

ov_status_e ov_genai_text2speech_decoded_results_get_output_sample_rate(
    const ov_genai_text2speech_decoded_results* results,
    uint32_t* sample_rate) {
    if (!results || !results->object || !sample_rate)
        return ov_status_e::INVALID_C_PARAM;
    *sample_rate = results->object->output_sample_rate;
    return ov_status_e::OK;
}

ov_status_e ov_genai_text2speech_decoded_results_get_speech_at(const ov_genai_text2speech_decoded_results* results,
                                                               size_t index,
                                                               ov_tensor_t** speech) {
    if (!results || !results->object || !speech)
        return ov_status_e::INVALID_C_PARAM;
    if (index >= results->object->speeches.size())
        return ov_status_e::OUT_OF_BOUNDS;
    struct tensor_guard {
        ov_tensor_t* tensor = nullptr;
        ~tensor_guard() {
            if (tensor)
                ov_tensor_free(tensor);
        }
    };
    try {
        const auto& source = results->object->speeches[index];
        if (source.get_element_type() != ov::element::f32)
            return ov_status_e::UNKNOW_EXCEPTION;
        // Snapshot the waveform into host memory first: the source may reference remote
        // device memory, where data() throws instead of transferring to host (R.1).
        ov::Tensor snapshot(ov::element::f32, source.get_shape());
        if (source.is<ov::RemoteTensor>()) {
            source.as<ov::RemoteTensor>().copy_to(snapshot);
        } else {
            source.copy_to(snapshot);
        }
        const auto dims = source.get_shape();
        std::vector<int64_t> c_dims(dims.begin(), dims.end());
        shape_guard guard;
        auto status = ov_shape_create(c_dims.size(), c_dims.data(), &guard.shape);
        if (status != ov_status_e::OK)
            return status;
        // Keep the allocated output tensor under RAII until ownership is transferred.
        tensor_guard output;
        status = ov_tensor_create(ov_element_type_e::F32, guard.shape, &output.tensor);
        if (status != ov_status_e::OK)
            return status;
        void* data = nullptr;
        status = ov_tensor_data(output.tensor, &data);
        if (status != ov_status_e::OK)
            return status;
        std::memcpy(data, snapshot.data(), snapshot.get_byte_size());
        *speech = output.tensor;
        output.tensor = nullptr;  // Ownership transferred to the caller.
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_text2speech_decoded_results_get_perf_metrics(const ov_genai_text2speech_decoded_results* results,
                                                                  ov_genai_perf_metrics** metrics) {
    if (!results || !results->object || !metrics)
        return ov_status_e::INVALID_C_PARAM;
    try {
        auto result = std::make_unique<ov_genai_perf_metrics>();
        result->object = std::make_shared<ov::genai::PerfMetrics>(results->object->perf_metrics);
        *metrics = result.release();
        return ov_status_e::OK;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

void ov_genai_text2speech_decoded_results_perf_metrics_free(ov_genai_perf_metrics* metrics) {
    delete metrics;
}

ov_status_e ov_genai_text2speech_decoded_results_get_num_generated_samples(
    const ov_genai_text2speech_decoded_results* results,
    size_t* count) {
    if (!results || !results->object || !count)
        return ov_status_e::INVALID_C_PARAM;
    *count = results->object->perf_metrics.num_generated_samples;
    return ov_status_e::OK;
}
