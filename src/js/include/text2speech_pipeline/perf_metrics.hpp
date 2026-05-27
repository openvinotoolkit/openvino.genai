// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "include/base/perf_metrics.hpp"
#include "openvino/genai/speech_generation/speech_generation_perf_metrics.hpp"

class Text2SpeechPerfMetricsWrapper
    : public BasePerfMetricsWrapper<Text2SpeechPerfMetricsWrapper, ov::genai::SpeechGenerationPerfMetrics> {
public:
    Text2SpeechPerfMetricsWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);
    static Napi::Object wrap(Napi::Env env, const ov::genai::SpeechGenerationPerfMetrics& metrics);

    Napi::Value get_num_generated_samples(const Napi::CallbackInfo& info);
    Napi::Value get_raw_metrics(const Napi::CallbackInfo& info);
};
