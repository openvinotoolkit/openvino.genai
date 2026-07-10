// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/asr_pipeline/perf_metrics.hpp"

#include "include/addon.hpp"
#include "include/helper.hpp"

using ov::genai::common_bindings::utils::get_ms;

ASRPerfMetricsWrapper::ASRPerfMetricsWrapper(const Napi::CallbackInfo& info)
    : BasePerfMetricsWrapper<ASRPerfMetricsWrapper, ov::genai::ASRPerfMetrics>(info) {}

Napi::Function ASRPerfMetricsWrapper::get_class(Napi::Env env) {
    auto properties = BasePerfMetricsWrapper<ASRPerfMetricsWrapper, ov::genai::ASRPerfMetrics>::get_class_properties();
    properties.push_back(
        InstanceMethod("getFeaturesExtractionDuration", &ASRPerfMetricsWrapper::get_features_extraction_duration));
    properties.push_back(InstanceMethod("getWordLevelTimestampsProcessingDuration",
                                        &ASRPerfMetricsWrapper::get_word_level_timestamps_processing_duration));
    properties.push_back(
        InstanceMethod("getEncodeInferenceDuration", &ASRPerfMetricsWrapper::get_encode_inference_duration));
    properties.push_back(
        InstanceMethod("getDecodeInferenceDuration", &ASRPerfMetricsWrapper::get_decode_inference_duration));
    properties.push_back(InstanceAccessor("asrRawMetrics", &ASRPerfMetricsWrapper::get_asr_raw_metrics, nullptr));
    return DefineClass(env, "ASRPerfMetrics", properties);
}

Napi::Object ASRPerfMetricsWrapper::wrap(Napi::Env env, const ov::genai::ASRPerfMetrics& metrics) {
    const auto& prototype = env.GetInstanceData<AddonData>()->asr_perf_metrics;
    OPENVINO_ASSERT(prototype, "Invalid pointer to prototype.");
    auto obj = prototype.New({});
    const auto m_ptr = Napi::ObjectWrap<ASRPerfMetricsWrapper>::Unwrap(obj);
    m_ptr->_metrics = metrics;
    return obj;
}

Napi::Value ASRPerfMetricsWrapper::get_features_extraction_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getFeaturesExtractionDuration()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_features_extraction_duration());
}

Napi::Value ASRPerfMetricsWrapper::get_word_level_timestamps_processing_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getWordLevelTimestampsProcessingDuration()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_word_level_timestamps_processing_duration());
}

Napi::Value ASRPerfMetricsWrapper::get_encode_inference_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getEncodeInferenceDuration()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_encode_inference_duration());
}

Napi::Value ASRPerfMetricsWrapper::get_decode_inference_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getDecodeInferenceDuration()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_decode_inference_duration());
}

Napi::Value ASRPerfMetricsWrapper::get_raw_metrics(const Napi::CallbackInfo& info) {
    return BasePerfMetricsWrapper<ASRPerfMetricsWrapper, ov::genai::ASRPerfMetrics>::get_raw_metrics(info);
}

Napi::Value ASRPerfMetricsWrapper::get_asr_raw_metrics(const Napi::CallbackInfo& info) {
    Napi::Object obj = Napi::Object::New(info.Env());
    obj.Set("featuresExtractionDurations",
            cpp_to_js<std::vector<float>, Napi::Value>(
                info.Env(),
                get_ms(_metrics.asr_raw_metrics, &ov::genai::ASRRawPerfMetrics::features_extraction_durations)));
    obj.Set("wordLevelTimestampsProcessingDurations",
            cpp_to_js<std::vector<float>, Napi::Value>(
                info.Env(),
                get_ms(_metrics.asr_raw_metrics,
                       &ov::genai::ASRRawPerfMetrics::word_level_timestamps_processing_durations)));
    obj.Set("encodeInferenceDurations",
            cpp_to_js<std::vector<float>, Napi::Value>(
                info.Env(),
                get_ms(_metrics.asr_raw_metrics, &ov::genai::ASRRawPerfMetrics::encode_inference_durations)));
    obj.Set("decodeInferenceDurations",
            cpp_to_js<std::vector<float>, Napi::Value>(
                info.Env(),
                get_ms(_metrics.asr_raw_metrics, &ov::genai::ASRRawPerfMetrics::decode_inference_durations)));
    return obj;
}
