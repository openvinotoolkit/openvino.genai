// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/core/extension.hpp>
#include <openvino/frontend/extension/conversion.hpp>

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({
    std::make_shared<ov::frontend::ConversionExtension>("GGML_OP_CONV_2D",
                                                        [](const ov::frontend::NodeContext&) -> ov::OutputVector {
                                                            OPENVINO_THROW(
                                                                "GGUF projector frontend extension was invoked");
                                                        }),
}));
