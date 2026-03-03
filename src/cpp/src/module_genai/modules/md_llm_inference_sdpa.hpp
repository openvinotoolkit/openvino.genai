// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"

#include "openvino/genai/tokenizer.hpp"

#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/weights/quantization_config.hpp"

namespace ov {
namespace genai {
namespace module {

/// @brief LLM inference module using SDPA (stateful) backend.
///
/// Supports two modes driven by the available inputs:
///   - **Text mode**: receives "input_ids" (OVTensor)
///     → builds attention_mask internally, runs stateful prefill + decode,
///     outputs "generated_text".
///   - **VL mode**: additionally receives "visual_embeds" (OVTensor),
///     "visual_pos_mask" (OVTensor), and "grid_thw" (OVTensor) → computes 3D MRoPE
///     position_ids via build_plan, scatters visual embeddings, then runs stateful
///     prefill + decode, outputs "generated_text".
///
/// Unlike LLMInferenceModule (PA / ContinuousBatching), this module compiles the
/// text model directly via ov::Core and drives a single InferRequest with
/// explicit KV-cache state management – i.e. the SDPA attention path.
class LLMInferenceSDPAModule : public IBaseModule {
    DeclareModuleConstructor(LLMInferenceSDPAModule);

private:
    bool initialize();

    // --- Helpers reused from md_qwen3_5_modeling logic ---
    std::string quant_suffix() const;
    static bool has_ir_pair(const std::filesystem::path& xml,
                            const std::filesystem::path& bin);
    static bool has_model_input(const std::shared_ptr<ov::Model>& m,
                                const std::string& name);
    static ov::Tensor make_zeros(ov::element::Type t, ov::Shape shape);
    static ov::Tensor make_beam_idx(size_t batch);
    static int64_t argmax_last(const ov::Tensor& logits);

    // --- Text-only decode (no vision inputs) ---
    std::string run_text_decode(const ov::Tensor& input_ids,
                                const ov::Tensor& attention_mask,
                                const ov::Tensor& position_ids,
                                const ov::Tensor& rope_deltas);

    // --- VL decode (with visual embeddings) ---
    std::string run_vl_decode(const ov::Tensor& input_ids,
                              const ov::Tensor& attention_mask,
                              const ov::Tensor& position_ids,
                              const ov::Tensor& rope_deltas,
                              const ov::Tensor& visual_embeds,
                              const ov::Tensor& visual_pos_mask);

    // Compiled text model + infer request
    ov::Core m_core;
    std::optional<ov::CompiledModel> m_compiled_text;
    bool m_text_uses_vl_ir = false;

    // Stop token tracking
    std::set<int64_t> m_stop_ids;

    // Model config
    ov::genai::modeling::models::Qwen3_5Config m_model_config;

    // Tokenizer (for text mode and decoding)
    std::unique_ptr<ov::genai::Tokenizer> m_tokenizer;

    // Max tokens to generate (default 256, overridden by params)
    size_t m_max_new_tokens = 256;
};

REGISTER_MODULE_CONFIG(LLMInferenceSDPAModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
