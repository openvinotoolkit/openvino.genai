// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/pipeline_base.hpp"

#include <set>
#include <string>

#include "openvino/genai/automatic_speech_recognition/generation_config.hpp"

namespace {

const std::set<std::string> allowed_asr_ctor_properties = {
    ov::genai::word_timestamps.name(),
};

}  // namespace

namespace ov::genai {

void erase_allowed_asr_ctor_properties(ov::AnyMap& properties) {
    for (const std::string& property_name : allowed_asr_ctor_properties) {
        properties.erase(property_name);
    }
}

}  // namespace ov::genai
