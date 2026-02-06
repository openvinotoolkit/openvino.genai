// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>

#include "openvino/runtime/tensor.hpp"
#include "utils.hpp"

namespace ov::genai::module {

#ifndef USE_FULL_MODEL
#    define USE_FULL_MODEL 0
#endif

class CSplittedModelInfer {
private:
    CSplittedModelInfer() = delete;
    CSplittedModelInfer(const std::string& model_path,
                        const std::string& device,
                        const bool& dynamic_load_model_weights = true,
                        const ov::AnyMap& properties = {});

    bool m_dynamic_load_model_weights;
    bool m_is_gpu = false;
    ov::AnyMap m_properties;

    void get_splitted_model_paths(const std::string& model_path, const std::string& device);
    void load_model(const std::string& model_path, const ov::AnyMap& properties, const std::string& device);

    std::vector<std::string> m_splitted_model_paths;
    std::string m_preprocess_model_path;
    std::string m_postprocess_model_path;

#if USE_FULL_MODEL
    ov::CompiledModel m_full_compiled_model;
    ov::InferRequest m_full_infer_request;
#else
    std::vector<ov::CompiledModel> m_compiled_models;
    std::vector<ov::InferRequest> m_infer_requests;
    ov::CompiledModel m_preprocess_compiled_model;
    ov::InferRequest m_preprocess_infer_request;
    ov::CompiledModel m_postprocess_compiled_model;
    ov::InferRequest m_postprocess_infer_request;
    ov::RemoteContext m_context;
#endif

    ov::Tensor convert_to_remote_tensor(const ov::Tensor& tensor);
public:
    ~CSplittedModelInfer();
    using PTR = std::shared_ptr<CSplittedModelInfer>;
    static PTR create(const std::string& model_path,
                      const std::string& device,
                      const bool& dynamic_load_model_weights = true,
                      const ov::AnyMap& properties = {}) {
        return std::shared_ptr<CSplittedModelInfer>(new CSplittedModelInfer(model_path, device, dynamic_load_model_weights, properties));
    }

    void infer(const ov::AnyMap& inputs);

    ov::Tensor get_output_tensor(const size_t& index = 0);

    void set_output_tensor(size_t idx, const ov::Tensor& tensor);
};
}  // namespace ov::genai::module