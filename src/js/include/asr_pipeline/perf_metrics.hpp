// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "include/base/perf_metrics.hpp"
#include "openvino/genai/automatic_speech_recognition/perf_metrics.hpp"

class ASRPerfMetricsWrapper : public BasePerfMetricsWrapper<ASRPerfMetricsWrapper, ov::genai::ASRPerfMetrics> {
public:
    ASRPerfMetricsWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);
    static Napi::Object wrap(Napi::Env env, const ov::genai::ASRPerfMetrics& metrics);

    Napi::Value get_features_extraction_duration(const Napi::CallbackInfo& info);
    Napi::Value get_word_level_timestamps_processing_duration(const Napi::CallbackInfo& info);
    Napi::Value get_encode_inference_duration(const Napi::CallbackInfo& info);
    Napi::Value get_decode_inference_duration(const Napi::CallbackInfo& info);
    Napi::Value get_raw_metrics(const Napi::CallbackInfo& info);
    Napi::Value get_asr_raw_metrics(const Napi::CallbackInfo& info);
};
