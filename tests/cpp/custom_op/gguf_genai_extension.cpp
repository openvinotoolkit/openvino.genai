// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/core/extension.hpp>
#include <openvino/frontend/gguf/extension/genai.hpp>
#include <vector>

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({
    std::make_shared<ov::frontend::gguf::GenAIExtension>(),
}));
