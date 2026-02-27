// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/vlm_pipeline/perf_metrics.hpp"

#include "include/addon.hpp"
#include "include/helper.hpp"

using ov::genai::common_bindings::utils::get_ms;

VLMPerfMetricsWrapper::VLMPerfMetricsWrapper(const Napi::CallbackInfo& info)
    : BasePerfMetricsWrapper<VLMPerfMetricsWrapper, ov::genai::VLMPerfMetrics>(info) {}

Napi::Function VLMPerfMetricsWrapper::get_class(Napi::Env env) {
    auto properties = BasePerfMetricsWrapper<VLMPerfMetricsWrapper, ov::genai::VLMPerfMetrics>::get_class_properties();
    properties.push_back(
        InstanceMethod("getPrepareEmbeddingsDuration", &VLMPerfMetricsWrapper::get_prepare_embeddings_duration));
    properties.push_back(InstanceAccessor<&VLMPerfMetricsWrapper::get_vlm_raw_metrics>("vlmRawMetrics"));
    return DefineClass(env, "VLMPerfMetrics", properties);
}

Napi::Object VLMPerfMetricsWrapper::wrap(Napi::Env env, const ov::genai::VLMPerfMetrics& metrics) {
    const auto& prototype = env.GetInstanceData<AddonData>()->vlm_perf_metrics;
    OPENVINO_ASSERT(prototype, "Invalid pointer to prototype.");
    auto obj = prototype.New({});
    const auto m_ptr = Napi::ObjectWrap<VLMPerfMetricsWrapper>::Unwrap(obj);
    m_ptr->_metrics = metrics;
    return obj;
}

Napi::Value VLMPerfMetricsWrapper::get_prepare_embeddings_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getPrepareEmbeddingsDuration()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_prepare_embeddings_duration());
}

Napi::Value VLMPerfMetricsWrapper::get_raw_metrics(const Napi::CallbackInfo& info) {
    return BasePerfMetricsWrapper<VLMPerfMetricsWrapper, ov::genai::VLMPerfMetrics>::get_raw_metrics(info);
}

Napi::Value VLMPerfMetricsWrapper::get_vlm_raw_metrics(const Napi::CallbackInfo& info) {
    Napi::Object obj = Napi::Object::New(info.Env());
    obj.Set("prepareEmbeddingsDurations",
            cpp_to_js<std::vector<float>, Napi::Value>(
                info.Env(),
                get_ms(_metrics.vlm_raw_metrics, &ov::genai::VLMRawPerfMetrics::prepare_embeddings_durations)));

    return obj;
}
