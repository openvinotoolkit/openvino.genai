// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "model_desc.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
ov::genai::ModelDesc get_draft_model_from_config(const ov::AnyMap& config) {
    ov::genai::ModelDesc draft_model;
    if (config.find(utils::DRAFT_MODEL_ARG_NAME) != config.end()) {
        draft_model = config.at(utils::DRAFT_MODEL_ARG_NAME).as<ov::genai::ModelDesc>();
    }
    return draft_model;
}

ov::genai::ModelDesc extract_draft_model_from_config(ov::AnyMap& config) {
    ov::genai::ModelDesc draft_model;
    if (config.find(ov::genai::utils::DRAFT_MODEL_ARG_NAME) != config.end()) {
        draft_model = config.at(ov::genai::utils::DRAFT_MODEL_ARG_NAME).as<ov::genai::ModelDesc>();
        config.erase(ov::genai::utils::DRAFT_MODEL_ARG_NAME);
    }
    return draft_model;
}
}  // namespace genai
}  // namespace ov