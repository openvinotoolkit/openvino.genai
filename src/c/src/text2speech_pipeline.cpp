// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/text2speech_pipeline.h"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "openvino/genai/speech_generation/speech_generation_config.hpp"

#include <cstring>
#include <memory>
#include <vector>
#include <string>

struct text2speech_pipeline_t {
    std::shared_ptr<ov::genai::Text2SpeechPipeline> impl;
};

int text2speech_pipeline_create(text2speech_pipeline_handle_t* pipeline,
                               const char* models_path,
                               const char* device) {
    if (!pipeline || !models_path || !device) {
        return 1;  // Invalid argument
    }

    try {
        auto ptr = std::make_shared<ov::genai::Text2SpeechPipeline>(std::filesystem::path(models_path), device, ov::AnyMap{});
        *pipeline = new text2speech_pipeline_t{ptr};
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int text2speech_pipeline_destroy(text2speech_pipeline_handle_t pipeline) {
    if (!pipeline) {
        return 1;  // Invalid argument
    }

    try {
        delete pipeline;
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

static text2speech_decoded_results_t* create_results(const ov::genai::Text2SpeechDecodedResults& res) {
    auto results = new text2speech_decoded_results_t();
    results->num_speeches = res.speeches.size();
    results->speeches = new speech_data_t[results->num_speeches];

    for (size_t i = 0; i < results->num_speeches; ++i) {
        const auto& tensor = res.speeches[i];
        const float* data = tensor.data<float>();
        size_t size = tensor.get_size();

        results->speeches[i].num_samples = size;
        results->speeches[i].sample_rate = 16000;  // Fixed sample rate
        results->speeches[i].samples = new float[size];
        std::memcpy(results->speeches[i].samples, data, size * sizeof(float));
    }

    // Copy performance metrics
    results->perf_metrics.num_generated_samples = res.perf_metrics.num_generated_samples;
    results->perf_metrics.base.generate_duration = res.perf_metrics.generate_duration;
    results->perf_metrics.base.throughput = res.perf_metrics.throughput;

    return results;
}

int text2speech_pipeline_generate(text2speech_pipeline_handle_t pipeline,
                                  const char* text,
                                  const float* speaker_embedding,
                                  size_t speaker_embedding_size,
                                  text2speech_decoded_results_t** results) {
    if (!pipeline || !text || !results) {
        return 1;  // Invalid argument
    }

    try {
        ov::Tensor speaker_tensor;
        if (speaker_embedding && speaker_embedding_size > 0) {
            ov::Shape shape{speaker_embedding_size};
            speaker_tensor = ov::Tensor(ov::element::f32, shape, const_cast<float*>(speaker_embedding));
        }

        std::vector<std::string> texts{std::string(text)};
        ov::genai::Text2SpeechDecodedResults res;
        if (speaker_embedding && speaker_embedding_size > 0) {
            res = pipeline->impl->generate(texts, speaker_tensor, pipeline->impl->get_generation_config());
        } else {
            res = pipeline->impl->generate(texts, ov::Tensor(), pipeline->impl->get_generation_config());
        }

        *results = create_results(res);
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int text2speech_pipeline_generate_batch(text2speech_pipeline_handle_t pipeline,
                                        const char** texts,
                                        size_t num_texts,
                                        const float* speaker_embedding,
                                        size_t speaker_embedding_size,
                                        text2speech_decoded_results_t** results) {
    if (!pipeline || !texts || num_texts == 0 || !results) {
        return 1;  // Invalid argument
    }

    try {
        std::vector<std::string> text_vec;
        for (size_t i = 0; i < num_texts; ++i) {
            text_vec.emplace_back(texts[i]);
        }

        ov::Tensor speaker_tensor;
        if (speaker_embedding && speaker_embedding_size > 0) {
            ov::Shape shape{speaker_embedding_size};
            speaker_tensor = ov::Tensor(ov::element::f32, shape, const_cast<float*>(speaker_embedding));
        }

        ov::genai::Text2SpeechDecodedResults res;
        if (speaker_embedding && speaker_embedding_size > 0) {
            res = pipeline->impl->generate(text_vec, speaker_tensor, pipeline->impl->get_generation_config());
        } else {
            res = pipeline->impl->generate(text_vec, ov::Tensor(), pipeline->impl->get_generation_config());
        }

        *results = create_results(res);
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int text2speech_pipeline_get_generation_config(text2speech_pipeline_handle_t pipeline,
                                               speech_generation_config_handle_t* config) {
    if (!pipeline || !config) {
        return 1;  // Invalid argument
    }

    try {
        auto gen_config = pipeline->impl->get_generation_config();
        *config = new speech_generation_config_t{gen_config};
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int text2speech_pipeline_set_generation_config(text2speech_pipeline_handle_t pipeline,
                                               speech_generation_config_handle_t config) {
    if (!pipeline || !config) {
        return 1;  // Invalid argument
    }

    try {
        pipeline->impl->set_generation_config(config->impl);
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int text2speech_decoded_results_destroy(text2speech_decoded_results_t* results) {
    if (!results) {
        return 1;  // Invalid argument
    }

    try {
        for (size_t i = 0; i < results->num_speeches; ++i) {
            delete[] results->speeches[i].samples;
        }
        delete[] results->speeches;
        delete results;
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}
