// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "include/base/perf_metrics.hpp"
#include "openvino/genai/perf_metrics.hpp"

class PerfMetricsWrapper : public BasePerfMetricsWrapper<PerfMetricsWrapper> {
public:
    PerfMetricsWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);
    static Napi::Object wrap(Napi::Env env, const ov::genai::PerfMetrics& metrics);

    Napi::Value get_raw_metrics(const Napi::CallbackInfo& info);
};
