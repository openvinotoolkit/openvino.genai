// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/vlm_pipeline.h"

#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/c/ov_tensor.h"
#include "types_c.h"
#include <stdarg.h>

namespace {

struct ov_shape_guard {
    ov_shape_t shape{};
    ~ov_shape_guard() { ov_shape_free(&shape); }
};

ov_status_e convert_c_tensors_to_cpp(const ov_tensor_t** rgbs,
                                     size_t num_images,
                                     std::vector<ov::Tensor>& rgbs_cpp) {
    if (num_images > 0 && !rgbs) {
        return ov_status_e::INVALID_C_PARAM;
    }

    rgbs_cpp.clear();
    rgbs_cpp.reserve(num_images);
    for (size_t i = 0; i < num_images; ++i) {
        const ov_tensor* ct = rgbs[i];
        if (!ct) {
            return ov_status_e::INVALID_C_PARAM;
        }

        auto et = ov::element::Type_t::u8;

        ov_shape_guard guard;
        ov_status_e status = ov_tensor_get_shape(ct, &guard.shape);
        if (status != ov_status_e::OK) {
            return status;
        }
        std::vector<size_t> dims(guard.shape.rank);
        for (size_t d = 0; d < guard.shape.rank; ++d) {
            dims[d] = guard.shape.dims[d];
        }

        void* data_ptr = nullptr;
        status = ov_tensor_data(const_cast<ov_tensor*>(ct), &data_ptr);
        if (status != ov_status_e::OK) {
            return status;
        }
        if (!data_ptr) {
            return ov_status_e::INVALID_C_PARAM;
        }

        rgbs_cpp.emplace_back(ov::element::Type(et), ov::Shape(dims), data_ptr);
    }

    return ov_status_e::OK;
}

template <typename GenerateWithStreamer, typename GenerateWithoutStreamer>
ov_status_e generate_vlm_results(GenerateWithStreamer&& generate_with_streamer,
                                 GenerateWithoutStreamer&& generate_without_streamer,
                                 const streamer_callback* streamer,
                                 ov_genai_vlm_decoded_results** results) {
    try {
        std::unique_ptr<ov_genai_vlm_decoded_results> _results = std::make_unique<ov_genai_vlm_decoded_results>();
        _results->object = std::make_shared<ov::genai::VLMDecodedResults>();

        if (streamer) {
            auto callback = [streamer](std::string word) -> ov::genai::StreamingStatus {
                return static_cast<ov::genai::StreamingStatus>((streamer->callback_func)(word.c_str(), streamer->args));
            };
            *(_results->object) = generate_with_streamer(callback);
        } else {
            *(_results->object) = generate_without_streamer();
        }

        if (results) {
            *results = _results.release();
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }

    return ov_status_e::OK;
}

template <typename InputType>
ov_status_e generate_vlm_with_optional_images(ov::genai::VLMPipeline& pipeline,
                                              const InputType& input,
                                              const std::vector<ov::Tensor>& rgbs_cpp,
                                              size_t num_images,
                                              const ov_genai_generation_config* config,
                                              const streamer_callback* streamer,
                                              ov_genai_vlm_decoded_results** results) {
    if (num_images > 0) {
        return generate_vlm_results(
            [&](const auto& callback) {
                return (config && config->object)
                           ? pipeline.generate(input, rgbs_cpp, *(config->object), callback)
                           : pipeline.generate(input, rgbs_cpp, {}, callback);
            },
            [&]() {
                ov::AnyMap config_map = {ov::genai::images(rgbs_cpp)};
                if (config && config->object) {
                    config_map.insert(ov::genai::generation_config(*(config->object)));
                }
                return pipeline.generate(input, config_map);
            },
            streamer,
            results
        );
    }

    return generate_vlm_results(
        [&](const auto& callback) {
            ov::AnyMap config_map = {ov::genai::streamer(callback)};
            if (config && config->object) {
                config_map.insert(ov::genai::generation_config(*(config->object)));
            }
            return pipeline.generate(input, config_map);
        },
        [&]() {
            ov::AnyMap config_map = {};
            if (config && config->object) {
                config_map.insert(ov::genai::generation_config(*(config->object)));
            }
            return pipeline.generate(input, config_map);
        },
        streamer,
        results
    );
}

}  // namespace

ov_status_e ov_genai_vlm_decoded_results_create(ov_genai_vlm_decoded_results** results) {
    if (!results) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_vlm_decoded_results> _results = std::make_unique<ov_genai_vlm_decoded_results>();
        _results->object = std::make_shared<ov::genai::VLMDecodedResults>();
        *results = _results.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_vlm_decoded_results_free(ov_genai_vlm_decoded_results* results) {
    if (results) {
        delete results;
    }
}

ov_status_e ov_genai_vlm_decoded_results_get_perf_metrics(const ov_genai_vlm_decoded_results* results,
                                                          ov_genai_perf_metrics** metrics) {
    if (!results || !(results->object) || !metrics) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_perf_metrics> _metrics = std::make_unique<ov_genai_perf_metrics>();
        _metrics->object = std::make_shared<ov::genai::VLMPerfMetrics>(results->object->perf_metrics);
        *metrics = _metrics.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_vlm_decoded_results_perf_metrics_free(ov_genai_perf_metrics* metrics) {
    if (metrics) {
        delete metrics;
    }
}

ov_status_e ov_genai_vlm_decoded_results_get_string(const ov_genai_vlm_decoded_results* results,
                                                    char* output,
                                                    size_t* output_size) {
    if (!results || !(results->object) || !(output_size)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::string str = *(results->object);
        if (!output) {
            *output_size = str.length() + 1;
        } else {
            if (*output_size < str.length() + 1) {
                return ov_status_e::OUT_OF_BOUNDS;
            }
            strncpy(output, str.c_str(), str.length() + 1);
            output[str.length()] = '\0';
            *output_size = str.length() + 1;
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_vlm_pipeline_create(const char* models_path, const char* device, const size_t property_args_size, ov_genai_vlm_pipeline** pipe, ...) {
    if (!models_path || !device || !pipe || property_args_size % 2 != 0) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        ov::AnyMap property = {};
        va_list args_ptr;
        va_start(args_ptr, pipe);
        size_t property_size = property_args_size / 2;
        for (size_t i = 0; i < property_size; i++) {
            GET_PROPERTY_FROM_ARGS_LIST;
        }
        va_end(args_ptr);
        // Check Property MAX_PROMPT_LEN and MIN_RESPONSE_LEN for NPU
        // These two special properties, which only affect the genai level, should be manually converted to integers 
        // before being passed to the constructor of VLMPipeline
        if(std::string(device) == "NPU") {
            if(property.find("MAX_PROMPT_LEN") != property.end()) {
                std::string max_prompt_len = property["MAX_PROMPT_LEN"].as<std::string>();
                property.erase("MAX_PROMPT_LEN");
                property["MAX_PROMPT_LEN"] = std::stoi(max_prompt_len);
            }
            if(property.find("MIN_RESPONSE_LEN") != property.end()) {
                std::string min_response_len = property["MIN_RESPONSE_LEN"].as<std::string>();
                property.erase("MIN_RESPONSE_LEN");
                property["MIN_RESPONSE_LEN"] = std::stoi(min_response_len);
            }
        }
        std::unique_ptr<ov_genai_vlm_pipeline> _pipe = std::make_unique<ov_genai_vlm_pipeline>();
        _pipe->object =
            std::make_shared<ov::genai::VLMPipeline>(std::filesystem::path(models_path), std::string(device), property);
        *pipe = _pipe.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_vlm_pipeline_free(ov_genai_vlm_pipeline* pipe) {
    if (pipe) {
        delete pipe;
    }
}

ov_status_e ov_genai_vlm_pipeline_generate(ov_genai_vlm_pipeline* pipe,
                                           const char* text_inputs,
                                           const ov_tensor_t** rgbs,
                                           size_t num_images,
                                           const ov_genai_generation_config* config,
                                           const streamer_callback* streamer,
                                           ov_genai_vlm_decoded_results** results) {
    if (!pipe || !(pipe->object) || !text_inputs || !(streamer || results)) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::vector<ov::Tensor> rgbs_cpp;
        ov_status_e status = convert_c_tensors_to_cpp(rgbs, num_images, rgbs_cpp);
        if (status != ov_status_e::OK) {
            return status;
        }

        std::string input_str(text_inputs);
        return generate_vlm_with_optional_images(*(pipe->object),
                                                 input_str,
                                                 rgbs_cpp,
                                                 num_images,
                                                 config,
                                                 streamer,
                                                 results);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
} 

ov_status_e ov_genai_vlm_pipeline_generate_with_history(ov_genai_vlm_pipeline* pipe,
                                                        const ov_genai_chat_history* history,
                                                        const ov_tensor_t** rgbs,
                                                        size_t num_images,
                                                        const ov_genai_generation_config* config,
                                                        const streamer_callback* streamer,
                                                        ov_genai_vlm_decoded_results** results) {
    if (!pipe || !(pipe->object) || !history || !(history->object) || !(streamer || results)) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::vector<ov::Tensor> rgbs_cpp;
        ov_status_e status = convert_c_tensors_to_cpp(rgbs, num_images, rgbs_cpp);
        if (status != ov_status_e::OK) {
            return status;
        }

        return generate_vlm_with_optional_images(*(pipe->object),
                                                 *(history->object),
                                                 rgbs_cpp,
                                                 num_images,
                                                 config,
                                                 streamer,
                                                 results);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_vlm_pipeline_start_chat(ov_genai_vlm_pipeline* pipe) {
    if (!pipe || !(pipe->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        pipe->object->start_chat();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_vlm_pipeline_finish_chat(ov_genai_vlm_pipeline* pipe) {
    if (!pipe || !(pipe->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        pipe->object->finish_chat();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_vlm_pipeline_get_generation_config(const ov_genai_vlm_pipeline* pipe,
                                                        ov_genai_generation_config** config) {
    if (!pipe || !(pipe->object) || !config) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_generation_config> _config = std::make_unique<ov_genai_generation_config>();
        _config->object = std::make_shared<ov::genai::GenerationConfig>(pipe->object->get_generation_config());
        *config = _config.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_vlm_pipeline_set_generation_config(ov_genai_vlm_pipeline* pipe,
                                                        ov_genai_generation_config* config) {
    if (!pipe || !(pipe->object) || !config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        pipe->object->set_generation_config(*(config->object));
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
