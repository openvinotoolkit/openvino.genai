// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include <openvino/core/any.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/infer_request.hpp>

namespace ov {
namespace genai {

/// @brief NPU-specific compilation for the Qwen3-TTS talker (the 28-layer
///        stateful decoder transformer).
///
/// The talker is a standard stateful LLM-style decoder (inputs_embeds /
/// attention_mask / position_ids + KV-cache state), so it is routed through the
/// same NPUW "LLM" path used by the LLM and VLM pipelines
/// (utils::compile_decoder_for_npu). That path enables NPU_USE_NPUW / NPUW_LLM,
/// derives the KV batch/seq axes, and lets the NPU plugin perform the
/// stateful->stateless conversion and reshape-to-static internally — so the
/// exact same IR exported for CPU/GPU is reused here, with no separate static
/// export.
///
/// @param core        Shared OpenVINO core (ov::genai::utils::singleton_core()).
/// @param model_path  Path to the talker IR (.xml).
/// @param properties  Compile-time properties for the NPU device. May contain
///                    MAX_PROMPT_LEN / MIN_RESPONSE_LEN to size the static KV
///                    cache; sensible defaults are applied otherwise.
/// @return An infer request created from the NPU-compiled talker.
ov::InferRequest compile_talker_for_npu(ov::Core& core,
                                        const std::filesystem::path& model_path,
                                        const ov::AnyMap& properties);

}  // namespace genai
}  // namespace ov
