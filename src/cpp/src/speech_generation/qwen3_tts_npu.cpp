// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_tts_npu.hpp"

#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>

#include <openvino/core/model.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/runtime/compiled_model.hpp>

#include "logger.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

namespace {

constexpr uint32_t DEFAULT_MAX_PROMPT_LEN = 1024u;
constexpr uint32_t DEFAULT_MIN_RESPONSE_LEN = 128u;

}  // namespace

ov::InferRequest compile_talker_for_npu(ov::Core& core,
                                        const std::filesystem::path& model_path,
                                        const ov::AnyMap& properties) {
    auto model = core.read_model(model_path);

    // Apply TTS-specific static-shape sizing unless the caller already set it.
    // compile_decoder_for_npu consumes MAX_PROMPT_LEN / MIN_RESPONSE_LEN from
    // these properties (falling back to LLM-oriented defaults otherwise).
    ov::AnyMap talker_properties = properties;
    talker_properties.emplace("MAX_PROMPT_LEN", static_cast<int64_t>(DEFAULT_MAX_PROMPT_LEN));
    talker_properties.emplace("MIN_RESPONSE_LEN", static_cast<int64_t>(DEFAULT_MIN_RESPONSE_LEN));

    // Derive the KV-cache batch/seq axes from the stateful model, then hand off
    // to the shared NPUW decoder path. compile_decoder_for_npu enables
    // NPU_USE_NPUW / NPUW_LLM, applies MAX_PROMPT_LEN / MIN_RESPONSE_LEN, and
    // relies on the NPU plugin to do stateful->stateless + reshape-to-static.
    const auto kv_pos = ov::genai::utils::get_kv_axes_pos(model);

    ov::CompiledModel compiled;
    ov::genai::utils::KVDesc kv_desc;
    std::tie(compiled, kv_desc) = ov::genai::utils::compile_decoder_for_npu(model, talker_properties, kv_pos);

    ov::genai::utils::print_compiled_model_properties(compiled, "qwen3_tts talker (NPU)");
    return compiled.create_infer_request();
}

}  // namespace genai
}  // namespace ov
