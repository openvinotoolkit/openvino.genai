// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

#include "openvino/openvino.hpp"

// Forward declare llama.cpp and ggml structs to avoid bleeding C-headers 
// into the public OpenVINO GenAI namespace.
struct llama_model;
struct llama_context;
struct ggml_cgraph;

namespace ov {
namespace genai {

/// @brief A dynamic GGUF reader utilizing llama.cpp and ov::frontend::ggml
class GGUFReaderV2 {
public:
    GGUFReaderV2();
    
    // Disable copy constructors to prevent double-freeing the llama_context
    GGUFReaderV2(const GGUFReaderV2&) = delete;
    GGUFReaderV2& operator=(const GGUFReaderV2&) = delete;

    ~GGUFReaderV2();

    /// @brief Loads a GGUF file, generates the GGML graph, and translates it to ov::Model
    /// @param model_path Path to the .gguf file
    /// @return A fully constructed OpenVINO model
    std::shared_ptr<ov::Model> read(const std::string& model_path);

private:
    void init_llama_context(const std::string& model_path);
    ggml_cgraph* build_computation_graph();
    std::shared_ptr<ov::Model> translate_to_ov(ggml_cgraph* graph);

    llama_model* m_model = nullptr;
    llama_context* m_ctx = nullptr;
};

} // namespace genai
} // namespace ov