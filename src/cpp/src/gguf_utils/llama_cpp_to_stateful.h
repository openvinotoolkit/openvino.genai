// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::genai::gguf_utils {

class LlamaCppToStateful : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::genai::gguf_utils::LlamaCppToStateful")

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::genai::gguf_utils
