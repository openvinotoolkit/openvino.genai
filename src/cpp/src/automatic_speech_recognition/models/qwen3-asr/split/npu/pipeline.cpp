// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline.hpp"

#include "openvino/runtime/properties.hpp"
#include "utils.hpp"

namespace {

ov::AnyMap configure_npu_input_models(const ov::AnyMap& properties) {
    for (const auto& role : {"audio_encoder", "text_embeddings"}) {
        const auto model_properties = ov::genai::utils::get_model_properties(properties, role, "AUTO");
        const auto priority = model_properties.find(ov::device::priorities.name());
        OPENVINO_ASSERT(priority == model_properties.end() || priority->second.as<std::string>() == "NPU",
                        "Qwen3-ASR on NPU requires MULTI_DEVICE_PRIORITIES=NPU for ",
                        role,
                        ".");
    }
    ov::AnyMap result = properties;
    auto devices = ov::genai::utils::pop_or_default<ov::AnyMap>(result, ov::device::properties.name(), {});
    auto automatic = ov::genai::utils::pop_or_default<ov::AnyMap>(devices, "AUTO", {});
    automatic[ov::device::priorities.name()] = std::string{"NPU"};
    devices["AUTO"] = automatic;
    result[ov::device::properties.name()] = devices;

    return result;
}

}  // namespace

namespace ov::genai {

Qwen3ASRNPUVLMPipelineAdapter::Qwen3ASRNPUVLMPipelineAdapter(const std::filesystem::path& models_path,
                                                             const ov::AnyMap& properties)
    : Qwen3ASRVLMPipelineAdapter(models_path, "NPU", configure_npu_input_models(properties)) {
    m_max_asr_input_seconds = 60;
}

}  // namespace ov::genai
