// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

namespace ov {
namespace genai {

struct WhisperInitializedModels {
    ov::InferRequest encoder;
    ov::CompiledModel decoder_compiled;
    ov::InferRequest decoder;
    ov::CompiledModel decoder_with_past_compiled;
    ov::InferRequest decoder_with_past;
};
}  // namespace genai
}  // namespace ov
