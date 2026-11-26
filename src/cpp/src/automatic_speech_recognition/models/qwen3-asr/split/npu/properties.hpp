// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <set>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils.hpp"

namespace ov::genai {

inline std::string qwen3_asr_input_device(const std::string& device,
                                          const ov::AnyMap& properties,
                                          const std::string& model_role) {
    if (device != "AUTO") {
        return device;
    }
    const auto auto_properties = utils::get_model_properties(properties, model_role, "AUTO");
    const auto priority = auto_properties.find(ov::device::priorities.name());
    if (priority != auto_properties.end() && priority->second.as<std::string>() == "NPU") {
        return "NPU";
    }
    return device;
}

inline ov::AnyMap qwen3_asr_input_model_properties(const ov::AnyMap& properties,
                                                   const std::string& model_role,
                                                   const std::string& device) {
    static const std::set<std::string> decoder_options{
        "MAX_PROMPT_LEN",
        "MIN_RESPONSE_LEN",
        "PREFILL_CONFIG",
        "PREFILL_HINT",
        "GENERATE_CONFIG",
        "GENERATE_HINT",
        "SHARED_HEAD_CONFIG",
        "BLOB_PATH",
        "EXPORT_BLOB",
    };
    auto result = utils::get_model_properties(properties, model_role, "NPU");
    const auto blob = result.find("BLOB_PATH");
    const bool decoder_blob = blob != result.end() && !blob->second.as<std::string>().empty();
    if (device == "AUTO") {
        result.erase(ov::device::priorities.name());
    }
    for (auto it = result.begin(); it != result.end();) {
        const auto option = it->first.rfind("++", 0) == 0 ? it->first.substr(2) : it->first;
        if (option == "NPUW_LLM" || option.rfind("NPUW_LLM_", 0) == 0 || decoder_options.count(option) ||
            (option == "WEIGHTS_PATH" && decoder_blob)) {
            it = result.erase(it);
        } else {
            ++it;
        }
    }
    return result;
}

inline ov::AnyMap qwen3_asr_npu_input_model_properties(const ov::AnyMap& properties,
                                                       const std::string& model_role,
                                                       const std::string& device) {
    auto result = qwen3_asr_input_model_properties(properties, model_role, device);
    utils::set_config_default(result, "NPU_USE_NPUW", std::string{"YES"});
    utils::set_config_default(result, "NPUW_DEVICES", std::string{"NPU"});
    // Keep the static input graphs intact unless the caller selects another partitioning policy.
    utils::set_config_default(result, "NPUW_ONLINE_PIPELINE", std::string{"NONE"});
    return result;
}

}  // namespace ov::genai
