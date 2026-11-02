// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>
#include <openvino/frontend/extension/conversion.hpp>
#include <vector>

#include "custom_add.hpp"

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({
    // Register operation
    std::make_shared<ov::OpExtension<TemplateExtension::MyAdd>>(),
    std::make_shared<ov::frontend::OpExtension<TemplateExtension::MyAdd>>(),
    std::make_shared<ov::frontend::ConversionExtension>("GGML_OP_MUL_MAT",
                                                        [](const ov::frontend::NodeContext&) -> ov::OutputVector {
                                                            OPENVINO_THROW("GGUF frontend extension was invoked");
                                                        }),
}));
