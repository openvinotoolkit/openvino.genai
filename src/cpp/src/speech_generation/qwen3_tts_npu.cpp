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

// TTS-specific NPU static-shape defaults for the talker. The generic NPUW LLM
// path defaults to MAX_PROMPT_LEN=1024 / MIN_RESPONSE_LEN=128, which are tuned
// for chat models with long prompts. The Qwen3-TTS talker works on short
// sequences (text tokens + codec frames at 12 Hz), so those defaults would
// allocate a much larger static KV cache and compile a bigger graph for no
// benefit. babelvox ships 256 for the equivalent knobs (max_talker_seq /
// max_kv_len), so we follow that.
//
// MAX_PROMPT_LEN  -> static prefill length (prompt the talker ingests in one pass).
// MIN_RESPONSE_LEN -> static decode budget; total static KV cache is the sum.
//                    256 frames at 12 Hz is ~21 s of audio, a reasonable starting
//                    point until we have data on typical generation lengths.
//
// Both are only applied when the caller has not specified them, so user-provided
// values (e.g. via the NPU device-properties map) still win.
constexpr uint32_t DEFAULT_MAX_PROMPT_LEN = 256u;
constexpr uint32_t DEFAULT_MIN_RESPONSE_LEN = 256u;

// The talker carries a Qwen3-TTS specific rank-3 `position_ids` ([3, batch,
// seq]) for its MROPE layout, unlike the rank-2 `position_ids` ([batch, seq])
// assumed by typical text decoders. The NPUW "LLM" path reshapes position_ids
// based on the sequence axis, so flag the rank-3 case explicitly: if NPUW ever
// rejects it, this gives a precise, actionable diagnostic instead of an opaque
// plugin error.
void warn_if_unconventional_position_ids(const std::shared_ptr<const ov::Model>& model) {
    for (const auto& input : model->inputs()) {
        const auto& names = input.get_names();
        if (names.find("position_ids") == names.end()) {
            continue;
        }
        const auto rank = input.get_partial_shape().rank();
        if (rank.is_static() && rank.get_length() != 2) {
            GENAI_WARN(
                "Qwen3-TTS talker exposes rank-%lld position_ids; the NPUW LLM path is tuned for rank-2 "
                "position_ids. If NPU compilation fails on reshape, this is the most likely cause.",
                static_cast<long long>(rank.get_length()));
        }
        return;
    }
}

// The exported talker IR carries the two outputs the pipeline actually consumes
// (`logits` and the final `hidden_states`) plus ~29 intermediate per-layer
// hidden-state outputs (`hidden_states.1`, `hidden_states.45`, ...). Those
// intermediates are export/trace artifacts that are never read at inference
// time, and they break NPUW's CutLMHead pass: it greedily matches a MatMul ->
// Result chain on one of the intermediate outputs instead of the real LM head,
// then fails with "Model references undeclared parameters".
//
// Pruning the talker down to just {logits, hidden_states} removes the ambiguity
// so the genuine logits MatMul is the unique LM-head cut point. This is an
// NPU-only graph fixup; the CPU/GPU paths keep the model untouched.
void prune_intermediate_hidden_states(const std::shared_ptr<ov::Model>& model) {
    static const std::unordered_set<std::string> keep_names{"logits", "hidden_states"};

    std::vector<std::shared_ptr<ov::op::v0::Result>> results_to_remove;
    for (const auto& result : model->get_results()) {
        const auto& names = result->get_output_tensor(0).get_names();
        const bool keep = std::any_of(names.begin(), names.end(), [](const std::string& name) {
            return keep_names.count(name) != 0;
        });
        if (!keep) {
            results_to_remove.push_back(result);
        }
    }

    for (const auto& result : results_to_remove) {
        model->remove_result(result);
    }

    if (!results_to_remove.empty()) {
        GENAI_DEBUG("Qwen3-TTS talker (NPU): pruned %zu intermediate hidden-state output(s); kept %zu.",
                    results_to_remove.size(),
                    model->get_results().size());
    }
}

}  // namespace

ov::InferRequest compile_talker_for_npu(ov::Core& core,
                                        const std::filesystem::path& model_path,
                                        const ov::AnyMap& properties) {
    std::cout << "talker model path: " << model_path << std::endl;
    auto model = core.read_model(model_path);

    warn_if_unconventional_position_ids(model);

    // Drop the unused intermediate per-layer hidden-state outputs that otherwise
    // confuse NPUW's LM-head detection.
    prune_intermediate_hidden_states(model);

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
