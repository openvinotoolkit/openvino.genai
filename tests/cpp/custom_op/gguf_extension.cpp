// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/core/extension.hpp>
#include <openvino/frontend/extension/decoder_transformation.hpp>
#include <openvino/pass/validate.hpp>
#include <vector>

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({
    std::make_shared<ov::frontend::DecoderTransformationExtension>(ov::pass::Validate()),
}));
