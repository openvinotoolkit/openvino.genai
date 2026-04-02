// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/whisper_pipeline/perf_metrics.hpp"

#include "include/addon.hpp"
#include "include/helper.hpp"

using ov::genai::common_bindings::utils::get_ms;

WhisperPerfMetricsWrapper::WhisperPerfMetricsWrapper(const Napi::CallbackInfo& info)
    : BasePerfMetricsWrapper<WhisperPerfMetricsWrapper, ov::genai::WhisperPerfMetrics>(info) {}

Napi::Function WhisperPerfMetricsWrapper::get_class(Napi::Env env) {
    auto properties =
        BasePerfMetricsWrapper<WhisperPerfMetricsWrapper, ov::genai::WhisperPerfMetrics>::get_class_properties();
    properties.push_back(
        InstanceMethod("getFeaturesExtractionDuration", &WhisperPerfMetricsWrapper::get_features_extraction_duration)
    );
    properties.push_back(InstanceMethod(
        "getWordLevelTimestampsProcessingDuration",
        &WhisperPerfMetricsWrapper::get_word_level_timestamps_processing_duration
    ));
    properties.push_back(InstanceAccessor<&WhisperPerfMetricsWrapper::get_whisper_raw_metrics>("whisperRawMetrics"));
    return DefineClass(env, "WhisperPerfMetrics", properties);
}

Napi::Object WhisperPerfMetricsWrapper::wrap(Napi::Env env, const ov::genai::WhisperPerfMetrics& metrics) {
    const auto& prototype = env.GetInstanceData<AddonData>()->whisper_perf_metrics;
    OPENVINO_ASSERT(prototype, "Invalid pointer to prototype.");
    auto obj = prototype.New({});
    const auto m_ptr = Napi::ObjectWrap<WhisperPerfMetricsWrapper>::Unwrap(obj);
    m_ptr->_metrics = metrics;
    return obj;
}

Napi::Value WhisperPerfMetricsWrapper::get_features_extraction_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getFeaturesExtractionDuration()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_features_extraction_duration());
}

Napi::Value WhisperPerfMetricsWrapper::get_word_level_timestamps_processing_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getWordLevelTimestampsProcessingDuration()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_word_level_timestamps_processing_duration());
}

Napi::Value WhisperPerfMetricsWrapper::get_raw_metrics(const Napi::CallbackInfo& info) {
    return BasePerfMetricsWrapper<WhisperPerfMetricsWrapper, ov::genai::WhisperPerfMetrics>::get_raw_metrics(info);
}

Napi::Value WhisperPerfMetricsWrapper::get_whisper_raw_metrics(const Napi::CallbackInfo& info) {
    Napi::Object obj = Napi::Object::New(info.Env());
    obj.Set(
        "featuresExtractionDurations",
        cpp_to_js<std::vector<float>, Napi::Value>(
            info.Env(),
            get_ms(_metrics.whisper_raw_metrics, &ov::genai::WhisperRawPerfMetrics::features_extraction_durations)
        )
    );
    obj.Set(
        "wordLevelTimestampsProcessingDurations",
        cpp_to_js<std::vector<float>, Napi::Value>(
            info.Env(),
            get_ms(
                _metrics.whisper_raw_metrics,
                &ov::genai::WhisperRawPerfMetrics::word_level_timestamps_processing_durations
            )
        )
    );
    return obj;
}
