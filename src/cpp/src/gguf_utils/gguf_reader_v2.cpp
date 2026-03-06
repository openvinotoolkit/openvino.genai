// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_reader_v2.hpp"
#include <openvino/openvino.hpp>

#ifdef HAS_LLAMA_CPP
#include "llama.h"
#include "ggml.h"
#endif

#include <iostream>
#include <mutex>
#include <stdexcept>

namespace ov {
namespace genai {

GGUFReaderV2::GGUFReaderV2() {
#ifdef HAS_LLAMA_CPP
    std::cout << "HAS_LLAMA_CPP defined\n";

    // Init backend only once globally — safe for multiple instances
    static std::once_flag backend_init_flag;
    std::call_once(backend_init_flag, []() {
        llama_backend_init();
    });
#else
    std::cout << "HAS_LLAMA_CPP NOT defined\n";

    throw std::runtime_error("GenAI was built without llama.cpp support!");
#endif
}

GGUFReaderV2::~GGUFReaderV2() {
#ifdef HAS_LLAMA_CPP
    if (m_ctx)   { llama_free(m_ctx);        m_ctx   = nullptr; }
    if (m_model) { llama_free_model(m_model); m_model = nullptr; }
#endif
}

void GGUFReaderV2::init_llama_context(const std::string& model_path) {
#ifdef HAS_LLAMA_CPP
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU only for graph capture

    m_model = llama_load_model_from_file(model_path.c_str(), model_params);
    OPENVINO_ASSERT(m_model != nullptr, 
        "[GGUFReaderV2] Failed to load GGUF file: ", model_path);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = 512;
    ctx_params.n_batch = 512;

    m_ctx = llama_init_from_model(m_model, ctx_params);
    OPENVINO_ASSERT(m_ctx != nullptr,
        "[GGUFReaderV2] Failed to create llama context");

    std::cout << "[GGUFReaderV2] Model loaded: " << model_path << "\n";
#endif
}

ggml_cgraph* GGUFReaderV2::build_computation_graph() {
    // TODO: implement graph building via llama.cpp internal API
    // This is the next step — need to find correct public API
    return nullptr;
}

std::shared_ptr<ov::Model> GGUFReaderV2::translate_to_ov(ggml_cgraph* graph) {
    // TODO: implement GgmlOvDecoder → InputModel → FrontEnd::convert
    return std::make_shared<ov::Model>(ov::ResultVector{}, ov::ParameterVector{});
}

std::shared_ptr<ov::Model> GGUFReaderV2::read(const std::string& model_path) {
#ifndef HAS_LLAMA_CPP
    throw std::runtime_error("Cannot read GGUF: llama.cpp support disabled.");
#else
    // Step 1 — Load model and create context
    init_llama_context(model_path);

    // Step 2 — Build GGML computation graph (TODO)
    ggml_cgraph* graph = build_computation_graph();

    // Step 3 — Translate to OpenVINO model (TODO)
    return translate_to_ov(graph);
#endif
}

} // namespace genai
} // namespace ov
