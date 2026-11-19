// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <set>
#include <string>

#include "openvino/core/any.hpp"

namespace ov::genai {

inline ov::AnyMap qwen3_asr_encoder_properties(const ov::AnyMap& properties) {
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
        if (option == "NPUW_LLM" || option.rfind("NPUW_LLM_", 0) == 0 || option.rfind("NPUW_QWEN3_ASR", 0) == 0 ||
            decoder_options.count(option) || (option == "WEIGHTS_PATH" && decoder_blob)) {
            it = result.erase(it);
        } else {
            ++it;
        }
    }
    return result;
}

}  // namespace ov::genai
