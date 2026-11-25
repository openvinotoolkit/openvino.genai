// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <set>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::genai {

inline std::string qwen3_asr_input_device(const std::string& device, const ov::AnyMap& properties) {
    if (device != "AUTO") {
        return device;
    }
    const auto devices = properties.find(ov::device::properties.name());
    if (devices == properties.end()) {
        return device;
    }
    const auto& device_map = devices->second.as<ov::AnyMap>();
    const auto auto_device = device_map.find("AUTO");
    if (auto_device == device_map.end()) {
        return device;
    }
    const auto& auto_properties = auto_device->second.as<ov::AnyMap>();
    const auto priority = auto_properties.find(ov::device::priorities.name());
    if (priority != auto_properties.end() && priority->second.as<std::string>() == "NPU") {
        return "NPU";
    }
    return device;
}

inline ov::AnyMap qwen3_asr_input_model_properties(const ov::AnyMap& properties) {
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
    auto result = properties;
    const auto blob = properties.find("BLOB_PATH");
    const bool decoder_blob = blob != properties.end() && !blob->second.as<std::string>().empty();
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

}  // namespace ov::genai
