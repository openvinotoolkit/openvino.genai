// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/core/extension.hpp>
#include <openvino/frontend/extension/conversion.hpp>
#include <vector>

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({
    std::make_shared<ov::frontend::ConversionExtension>("GGML_OP_MUL_MAT",
                                                        [](const ov::frontend::NodeContext&) -> ov::OutputVector {
                                                            OPENVINO_THROW("GGUF frontend extension was invoked");
                                                        }),
}));
