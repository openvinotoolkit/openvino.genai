// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "include/base/perf_metrics.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

class WhisperPerfMetricsWrapper
    : public BasePerfMetricsWrapper<WhisperPerfMetricsWrapper, ov::genai::WhisperPerfMetrics> {
public:
    WhisperPerfMetricsWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);
    static Napi::Object wrap(Napi::Env env, const ov::genai::WhisperPerfMetrics& metrics);

    Napi::Value get_features_extraction_duration(const Napi::CallbackInfo& info);
    Napi::Value get_word_level_timestamps_processing_duration(const Napi::CallbackInfo& info);
    Napi::Value get_raw_metrics(const Napi::CallbackInfo& info);
    Napi::Value get_whisper_raw_metrics(const Napi::CallbackInfo& info);
};
