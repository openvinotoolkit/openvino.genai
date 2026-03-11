// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "include/base/perf_metrics.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"

class VLMPerfMetricsWrapper : public BasePerfMetricsWrapper<VLMPerfMetricsWrapper, ov::genai::VLMPerfMetrics> {
public:
    VLMPerfMetricsWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);
    static Napi::Object wrap(Napi::Env env, const ov::genai::VLMPerfMetrics& metrics);

    Napi::Value get_prepare_embeddings_duration(const Napi::CallbackInfo& info);
    Napi::Value get_raw_metrics(const Napi::CallbackInfo& info);
    Napi::Value get_vlm_raw_metrics(const Napi::CallbackInfo& info);
};
