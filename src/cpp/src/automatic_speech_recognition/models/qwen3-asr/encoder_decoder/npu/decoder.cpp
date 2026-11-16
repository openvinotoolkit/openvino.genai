// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "decoder.hpp"

#include <charconv>
#include <limits>

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

int64_t read_encoder_capacity(const ov::Any& value) {
    if (value.is<int64_t>()) {
        return value.as<int64_t>();
    }
    if (value.is<int>()) {
        return value.as<int>();
    }
    if (value.is<std::string>()) {
        const auto text = value.as<std::string>();
        int64_t result = 0;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
        OPENVINO_ASSERT(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(),
                        "NPUW_QWEN3_ASR_MAX_ENCODER_LEN must be an integer, got '",
                        text,
                        "'.");
        return result;
    }
    OPENVINO_THROW("NPUW_QWEN3_ASR_MAX_ENCODER_LEN must be an int, int64 or integer string.");
}

}  // namespace

namespace ov::genai {

ASRGenerationConfig prepare_qwen3_asr_npu_generation(size_t prompt_len,
                                                     const ASRGenerationConfig& config,
                                                     size_t max_prompt_len,
                                                     size_t kv_cache_capacity) {
    OPENVINO_ASSERT(prompt_len > 0 && prompt_len <= max_prompt_len,
                    "Qwen3-ASR NPU prompt length ",
                    prompt_len,
                    " must be between 1 and MAX_PROMPT_LEN ",
                    max_prompt_len,
                    ".");
    // Prefill predicts the first token; each further token needs one decoder inference.
    OPENVINO_ASSERT(prompt_len <= kv_cache_capacity + 1,
                    "Qwen3-ASR NPU prompt exceeds the compiled KV-cache capacity.");
    const size_t available_new_tokens = kv_cache_capacity + 2 - prompt_len;
    OPENVINO_ASSERT(config.min_new_tokens <= available_new_tokens,
                    "Qwen3-ASR NPU min_new_tokens is ",
                    config.min_new_tokens,
                    ", but the compiled KV-cache can accommodate only ",
                    available_new_tokens,
                    " after a prompt of ",
                    prompt_len,
                    " tokens. Reduce min_new_tokens or increase MAX_PROMPT_LEN or MIN_RESPONSE_LEN.");
    auto result = config;
    if (config.get_max_new_tokens(prompt_len) > available_new_tokens) {
        result.max_new_tokens = available_new_tokens;
    }
    return result;
}

ov::AnyMap prepare_qwen3_asr_npu_properties(const ov::AnyMap& properties, size_t max_encoder_len) {
    OPENVINO_ASSERT(
        max_encoder_len > 0 && max_encoder_len <= static_cast<size_t>(std::numeric_limits<int64_t>::max() -
                                                                      TEXT_TOKEN_BUDGET - PROMPT_LEN_ALIGNMENT + 1),
        "NPUW_QWEN3_ASR_MAX_ENCODER_LEN must be > 0 and fit int64, got ",
        max_encoder_len);
    auto result = properties;
    const auto encoder_len_it = result.find("NPUW_QWEN3_ASR_MAX_ENCODER_LEN");
    if (encoder_len_it != result.end()) {
        const auto configured_len = read_encoder_capacity(encoder_len_it->second);
        OPENVINO_ASSERT(configured_len == static_cast<int64_t>(max_encoder_len),
                        "NPUW_QWEN3_ASR_MAX_ENCODER_LEN is ",
                        configured_len,
                        ", but the audio pipeline requires ",
                        max_encoder_len,
                        ".");
    }
    result["NPUW_QWEN3_ASR_MAX_ENCODER_LEN"] = static_cast<int64_t>(max_encoder_len);

    const auto blob = result.find("BLOB_PATH");
    const auto export_blob = result.find("EXPORT_BLOB");
    const bool importing = blob != result.end() && !blob->second.as<std::string>().empty() &&
                           (export_blob == result.end() || !export_blob->second.as<bool>());
    if (!importing && result.count("MAX_PROMPT_LEN") == 0) {
        const size_t max_prompt_len =
            ((max_encoder_len + TEXT_TOKEN_BUDGET + PROMPT_LEN_ALIGNMENT - 1) / PROMPT_LEN_ALIGNMENT) *
            PROMPT_LEN_ALIGNMENT;
        result["MAX_PROMPT_LEN"] = static_cast<int64_t>(max_prompt_len);
    }
    utils::set_config_default(result, "NPUW_QWEN3_ASR", std::string{"YES"});
    utils::set_config_default(result, "NPUW_LLM_PREFILL_HINT", std::string{"STATIC"});
    return result;
}

Qwen3ASRDecoderNPU::Qwen3ASRDecoderNPU(const std::filesystem::path& models_path,
                                       const ov::AnyMap& properties,
                                       size_t max_encoder_len) {
    auto npu_properties = prepare_qwen3_asr_npu_properties(properties, max_encoder_len);
    auto model = read_model(models_path, properties);
    const auto kv_pos = utils::get_kv_axes_pos(model);
    reshape_encoder_hidden_states(model);

    auto [compiled_model, kv_desc] = utils::compile_decoder_for_npu(model, npu_properties, kv_pos);
    m_max_prompt_len = compiled_model.get_property("NPUW_LLM_MAX_PROMPT_LEN").as<uint32_t>();
    m_kv_cache_capacity = utils::get_npu_kv_cache_capacity(compiled_model);
    utils::print_compiled_model_properties(compiled_model, "qwen3-asr decoder model");
    m_request = compiled_model.create_infer_request();
}

ASRGenerationConfig Qwen3ASRDecoderNPU::prepare_generation_config(size_t prompt_len,
                                                                  const ASRGenerationConfig& config) const {
    return prepare_qwen3_asr_npu_generation(prompt_len, config, m_max_prompt_len, m_kv_cache_capacity);
}

}  // namespace ov::genai
