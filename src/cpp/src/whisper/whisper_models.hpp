// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

namespace ov {
namespace genai {

struct WhisperInitializedModels {
    ov::InferRequest encoder;
    ov::InferRequest decoder;
    ov::InferRequest decoder_with_past;
};
}  // namespace genai
}  // namespace ov
