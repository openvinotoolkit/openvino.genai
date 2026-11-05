// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "decoder.hpp"

#include "utils.hpp"

namespace {

constexpr size_t TEXT_TOKEN_BUDGET = 128;
constexpr size_t PROMPT_LEN_ALIGNMENT = 64;

void reshape_encoder_hidden_states(const std::shared_ptr<ov::Model>& model) {
    if (!ov::genai::utils::has_input(model, "encoder_hidden_states")) {
        return;
    }
    auto shape = model->input("encoder_hidden_states").get_partial_shape();
    OPENVINO_ASSERT(shape.rank().is_static() && shape.rank().get_length() >= 2,
                    "Unexpected rank for 'encoder_hidden_states' input.");
    shape[0] = 1;
    shape[1] = ov::Dimension::dynamic();
    model->reshape({{"encoder_hidden_states", shape}});
}

}  // namespace

namespace ov::genai {

Qwen3ASRDecoderNPU::Qwen3ASRDecoderNPU(const std::filesystem::path& models_path,
                                       const ov::AnyMap& properties,
                                       size_t max_encoder_len) {
    OPENVINO_ASSERT(max_encoder_len > 0, "NPUW_QWEN3_ASR_MAX_ENCODER_LEN must be > 0, got ", max_encoder_len);
    auto model = read_model(models_path, properties);
    const auto kv_pos = utils::get_kv_axes_pos(model);
    reshape_encoder_hidden_states(model);

    auto npu_properties = properties;
    npu_properties["NPUW_QWEN3_ASR_MAX_ENCODER_LEN"] = static_cast<int64_t>(max_encoder_len);
    if (npu_properties.count("MAX_PROMPT_LEN") == 0) {
        const size_t max_prompt_len =
            ((max_encoder_len + TEXT_TOKEN_BUDGET + PROMPT_LEN_ALIGNMENT - 1) / PROMPT_LEN_ALIGNMENT) *
            PROMPT_LEN_ALIGNMENT;
        npu_properties["MAX_PROMPT_LEN"] = static_cast<int64_t>(max_prompt_len);
    }
    utils::set_config_default(npu_properties, "NPUW_QWEN3_ASR", std::string{"YES"});
    utils::set_config_default(npu_properties, "NPUW_LLM_PREFILL_HINT", std::string{"STATIC"});
    auto [compiled_model, kv_desc] = utils::compile_decoder_for_npu(model, npu_properties, kv_pos);
    utils::print_compiled_model_properties(compiled_model, "qwen3-asr decoder model");
    m_request = compiled_model.create_infer_request();
}

}  // namespace ov::genai
