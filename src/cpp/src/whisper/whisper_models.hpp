// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>
#include <vector>

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160

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
