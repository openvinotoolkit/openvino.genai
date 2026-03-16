// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_reader_v2.hpp"
#include <openvino/openvino.hpp>

#ifdef HAS_LLAMA_CPP
#include "llama.h"
#include "ggml.h"

#include "ggml-openvino/ggml-decoder.h"
#include "ggml-openvino/openvino/input_model.h"
#include "ggml-openvino/openvino/frontend.h"
#include "ggml-openvino/ggml-openvino-extra.h"
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
    model_params.n_gpu_layers = 9999;  // CPU only for graph capture

    m_model = llama_load_model_from_file(model_path.c_str(), model_params);
    OPENVINO_ASSERT(m_model != nullptr, 
        "[GGUFReaderV2] Failed to load GGUF file: ", model_path);

    llama_context_params ctx_params = llama_context_default_params();

    uint32_t trace_seq_len = 1;


    ctx_params.n_ctx   = trace_seq_len;
    ctx_params.n_batch = trace_seq_len;
    ctx_params.n_ubatch = trace_seq_len;

    m_ctx = llama_init_from_model(m_model, ctx_params);
    OPENVINO_ASSERT(m_ctx != nullptr,
        "[GGUFReaderV2] Failed to create llama context");

    std::cout << "[GGUFReaderV2] Model loaded: " << model_path << "\n";
#endif
}

ggml_cgraph* GGUFReaderV2::build_computation_graph() {
#ifdef HAS_LLAMA_CPP
    std::cout << "[GGUFReaderV2] Forcing graph build with dummy decode...\n";

    // 1. Get the BOS token for the loaded model
    const llama_vocab* vocab = llama_model_get_vocab(m_model);
    llama_token bos_token = llama_token_bos(vocab);
    if (bos_token == -1) {
        bos_token = 0; // Fallback if the model lacks a BOS token
    }

    // 2. Create a minimal batch of size 1
    llama_batch batch = llama_batch_get_one(&bos_token, 1);

    // 3. TODO: Enable capture mode in your custom backend
    // You will need to add these API calls to Ravi's ggml-openvino backend
    ggml_backend_ov_set_capture_mode(true);

    // 4. Trigger the decode to force ggml_cgraph construction
    int res = llama_decode(m_ctx, batch);
    OPENVINO_ASSERT(res == 0, "[GGUFReaderV2] Error: Dummy llama_decode failed!");

    // 5. TODO: Extract the intercepted graph
    ggml_cgraph* extracted_graph = ggml_backend_ov_get_captured_graph();

    // 6. TODO: Disable capture mode
    ggml_backend_ov_set_capture_mode(false);

    std::cout << "[GGUFReaderV2] Graph successfully extracted!\n";
    return extracted_graph;
#else
    return nullptr;
#endif
}

std::shared_ptr<ov::Model> GGUFReaderV2::translate_to_ov(ggml_cgraph* graph) {
#ifdef HAS_LLAMA_CPP
    OPENVINO_ASSERT(graph != nullptr, "[GGUFReaderV2] Cannot translate a null graph.");

    std::cout << "[GGUFReaderV2] Starting GGML to OpenVINO translation...\n";

    // 1. Setup translation parameters
    // We default to dynamic shapes (is_static = false) and no state initially 
    // to match standard reader behavior.
    bool is_static = true; 
    bool stateful = false;  

    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(graph, is_static);

    // 2. Extract weights and initialize the decoder
    auto model_weights = GgmlOvDecoder::create_weight_nodes(graph);
    auto ggml_decoder = std::make_shared<GgmlOvDecoder>(
        graph, m_params, c_params, model_weights, is_static, stateful
    );

    // 3. Wrap in InputModel and Convert
    auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder);
    std::shared_ptr<ov::Model> ov_model = ov::frontend::ggml::FrontEnd::convert(input_model);

    // 4. Cleanup to free memory
    ggml_decoder->clear_model_weights();

    std::cout << "[GGUFReaderV2] Translation complete!\n";
    return ov_model;
#else
    return nullptr;
#endif
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

float* GGUFReaderV2::get_native_logits() const {
#ifdef HAS_LLAMA_CPP
    if (m_ctx) {
        return llama_get_logits(m_ctx);
    }
#endif
    return nullptr;
}

} // namespace genai
} // namespace ov
