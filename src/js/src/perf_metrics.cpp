// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/perf_metrics.hpp"

#include "include/addon.hpp"

PerfMetricsWrapper::PerfMetricsWrapper(const Napi::CallbackInfo& info)
    : BasePerfMetricsWrapper<PerfMetricsWrapper>(info) {}

Napi::Function PerfMetricsWrapper::get_class(Napi::Env env) {
    auto properties = BasePerfMetricsWrapper<PerfMetricsWrapper>::get_class_properties();
    return DefineClass(env, "PerfMetrics", properties);
}

Napi::Object PerfMetricsWrapper::wrap(Napi::Env env, const ov::genai::PerfMetrics& metrics) {
    const auto& prototype = env.GetInstanceData<AddonData>()->perf_metrics;
    OPENVINO_ASSERT(prototype, "Invalid pointer to prototype.");
    auto obj = prototype.New({});
    const auto m_ptr = Napi::ObjectWrap<PerfMetricsWrapper>::Unwrap(obj);
    m_ptr->_metrics = metrics;
    return obj;
}

Napi::Value PerfMetricsWrapper::get_raw_metrics(const Napi::CallbackInfo& info) {
    return BasePerfMetricsWrapper<PerfMetricsWrapper>::get_raw_metrics(info);
}
