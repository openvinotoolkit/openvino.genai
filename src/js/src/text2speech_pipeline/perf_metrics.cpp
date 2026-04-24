// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text2speech_pipeline/perf_metrics.hpp"

#include "include/addon.hpp"
#include "include/helper.hpp"

Text2SpeechPerfMetricsWrapper::Text2SpeechPerfMetricsWrapper(const Napi::CallbackInfo& info)
    : BasePerfMetricsWrapper<Text2SpeechPerfMetricsWrapper, ov::genai::SpeechGenerationPerfMetrics>(info) {}

Napi::Function Text2SpeechPerfMetricsWrapper::get_class(Napi::Env env) {
    auto properties =
        BasePerfMetricsWrapper<Text2SpeechPerfMetricsWrapper, ov::genai::SpeechGenerationPerfMetrics>::
            get_class_properties();
    properties.push_back(
        InstanceMethod("getNumGeneratedSamples", &Text2SpeechPerfMetricsWrapper::get_num_generated_samples));
    return DefineClass(env, "Text2SpeechPerfMetrics", properties);
}

Napi::Object Text2SpeechPerfMetricsWrapper::wrap(Napi::Env env,
                                                  const ov::genai::SpeechGenerationPerfMetrics& metrics) {
    const auto& prototype = env.GetInstanceData<AddonData>()->text2speech_perf_metrics;
    OPENVINO_ASSERT(prototype, "Invalid pointer to prototype.");
    auto obj = prototype.New({});
    const auto m_ptr = Napi::ObjectWrap<Text2SpeechPerfMetricsWrapper>::Unwrap(obj);
    m_ptr->_metrics = metrics;
    return obj;
}

Napi::Value Text2SpeechPerfMetricsWrapper::get_num_generated_samples(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getNumGeneratedSamples()");
    return cpp_to_js<size_t, Napi::Value>(info.Env(), _metrics.num_generated_samples);
}

Napi::Value Text2SpeechPerfMetricsWrapper::get_raw_metrics(const Napi::CallbackInfo& info) {
    return BasePerfMetricsWrapper<Text2SpeechPerfMetricsWrapper,
                                  ov::genai::SpeechGenerationPerfMetrics>::get_raw_metrics(info);
}
