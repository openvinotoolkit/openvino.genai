// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "bindings_utils.hpp"
#include "include/helper.hpp"
#include "openvino/genai/perf_metrics.hpp"

using ov::genai::common_bindings::utils::get_ms;
using ov::genai::common_bindings::utils::timestamp_to_ms;

namespace perf_utils {

inline Napi::Object create_mean_std_pair(Napi::Env env, const ov::genai::MeanStdPair& pair) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("mean", Napi::Number::New(env, pair.mean));
    obj.Set("std", Napi::Number::New(env, pair.std));
    return obj;
}

inline Napi::Object create_summary_stats(Napi::Env env, const ov::genai::SummaryStats& stats) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("mean", Napi::Number::New(env, stats.mean));
    obj.Set("std", Napi::Number::New(env, stats.std));
    obj.Set("min", Napi::Number::New(env, stats.min));
    obj.Set("max", Napi::Number::New(env, stats.max));
    return obj;
}

}  // namespace perf_utils

/**
 * @brief Base template class for PerfMetrics wrappers.
 *
 * This class provides common functionality for wrapping ov::genai::PerfMetrics
 * and derived classes (like VLMPerfMetrics) in Node.js addon.
 *
 * @tparam T The derived wrapper class (CRTP pattern).
 * @tparam MetricsType The type of metrics to store (default: ov::genai::PerfMetrics).
 */
template <class T, class MetricsType = ov::genai::PerfMetrics>
class BasePerfMetricsWrapper : public Napi::ObjectWrap<T> {
public:
    using PropertyDescriptor = typename Napi::ObjectWrap<T>::PropertyDescriptor;

    BasePerfMetricsWrapper(const Napi::CallbackInfo& info);
    virtual ~BasePerfMetricsWrapper() {}

    /**
     * @brief Returns a vector of base class property descriptors.
     *
     * Derived classes can use this to get all base methods and add their own.
     */
    static std::vector<PropertyDescriptor> get_class_properties();

    Napi::Value get_load_time(const Napi::CallbackInfo& info);
    Napi::Value get_num_generated_tokens(const Napi::CallbackInfo& info);
    Napi::Value get_num_input_tokens(const Napi::CallbackInfo& info);
    Napi::Value get_ttft(const Napi::CallbackInfo& info);
    Napi::Value get_tpot(const Napi::CallbackInfo& info);
    Napi::Value get_ipot(const Napi::CallbackInfo& info);
    Napi::Value get_throughput(const Napi::CallbackInfo& info);

    Napi::Value get_inference_duration(const Napi::CallbackInfo& info);
    Napi::Value get_generate_duration(const Napi::CallbackInfo& info);
    Napi::Value get_tokenization_duration(const Napi::CallbackInfo& info);
    Napi::Value get_detokenization_duration(const Napi::CallbackInfo& info);

    Napi::Value get_grammar_compiler_init_times(const Napi::CallbackInfo& info);
    Napi::Value get_grammar_compile_time(const Napi::CallbackInfo& info);

    /**
     * @brief Base implementation of get_raw_metrics.
     *
     * Derived classes MUST override this method to use it with InstanceAccessor.
     * Example:
     *
     * Napi::Value get_raw_metrics(const Napi::CallbackInfo& info) {
     *     return BasePerfMetricsWrapper<DerivedClass, MetricsType>::get_raw_metrics(info);
     * }
     */
    Napi::Value get_raw_metrics(const Napi::CallbackInfo& info);
    Napi::Value add(const Napi::CallbackInfo& info);
    MetricsType& get_value();

protected:
    MetricsType _metrics;
};

// Template implementations

template <class T, class MetricsType>
BasePerfMetricsWrapper<T, MetricsType>::BasePerfMetricsWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<T>(info),
      _metrics{} {}

template <class T, class MetricsType>
std::vector<typename BasePerfMetricsWrapper<T, MetricsType>::PropertyDescriptor>
BasePerfMetricsWrapper<T, MetricsType>::get_class_properties() {
    return {
        T::InstanceMethod("getLoadTime", &T::get_load_time),
        T::InstanceMethod("getNumGeneratedTokens", &T::get_num_generated_tokens),
        T::InstanceMethod("getNumInputTokens", &T::get_num_input_tokens),
        T::InstanceMethod("getTTFT", &T::get_ttft),
        T::InstanceMethod("getTPOT", &T::get_tpot),
        T::InstanceMethod("getIPOT", &T::get_ipot),
        T::InstanceMethod("getThroughput", &T::get_throughput),
        T::InstanceMethod("getInferenceDuration", &T::get_inference_duration),
        T::InstanceMethod("getGenerateDuration", &T::get_generate_duration),
        T::InstanceMethod("getTokenizationDuration", &T::get_tokenization_duration),
        T::InstanceMethod("getDetokenizationDuration", &T::get_detokenization_duration),
        T::InstanceMethod("getGrammarCompilerInitTimes", &T::get_grammar_compiler_init_times),
        T::InstanceMethod("getGrammarCompileTime", &T::get_grammar_compile_time),
        T::template InstanceAccessor<&T::get_raw_metrics>("rawMetrics"),
        T::InstanceMethod("add", &T::add),
    };
}

template <class T, class MetricsType>
Napi::Value BasePerfMetricsWrapper<T, MetricsType>::get_load_time(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getLoadTime()");
    return Napi::Number::New(info.Env(), _metrics.get_load_time());
}

template <class T, class MetricsType>
Napi::Value BasePerfMetricsWrapper<T, MetricsType>::get_num_generated_tokens(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getNumGeneratedTokens()");
    return Napi::Number::New(info.Env(), _metrics.get_num_generated_tokens());
}

template <class T, class MetricsType>
Napi::Value BasePerfMetricsWrapper<T, MetricsType>::get_num_input_tokens(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getNumInputTokens()");
    return Napi::Number::New(info.Env(), _metrics.get_num_input_tokens());
}

template <class T, class MetricsType>
Napi::Value BasePerfMetricsWrapper<T, MetricsType>::get_ttft(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getTTFT()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_ttft());
}

template <class T, class MetricsType>
Napi::Value BasePerfMetricsWrapper<T, MetricsType>::get_tpot(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getTPOT()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_tpot());
}

template <class T, class MetricsType>
Napi::Value BasePerfMetricsWrapper<T, MetricsType>::get_ipot(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getIPOT()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_ipot());
}

template <class T, class MetricsType>
Napi::Value BasePerfMetricsWrapper<T, MetricsType>::get_throughput(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getThroughput()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_throughput());
}

template <class T, class MetricsType>
Napi::Value BasePerfMetricsWrapper<T, MetricsType>::get_inference_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getInferenceDuration()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_inference_duration());
}

template <class T, class MetricsType>
Napi::Value BasePerfMetricsWrapper<T, MetricsType>::get_generate_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getGenerateDuration()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_generate_duration());
}

template <class T, class MetricsType>
Napi::Value BasePerfMetricsWrapper<T, MetricsType>::get_tokenization_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getTokenizationDuration()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_tokenization_duration());
}

template <class T, class MetricsType>
Napi::Value BasePerfMetricsWrapper<T, MetricsType>::get_detokenization_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getDetokenizationDuration()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_detokenization_duration());
}

template <class T, class MetricsType>
Napi::Value BasePerfMetricsWrapper<T, MetricsType>::get_grammar_compiler_init_times(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getGrammarCompilerInitTimes()");
    return cpp_map_to_js_object(info.Env(), _metrics.get_grammar_compiler_init_times());
}

template <class T, class MetricsType>
Napi::Value BasePerfMetricsWrapper<T, MetricsType>::get_grammar_compile_time(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getGrammarCompileTime()");
    return perf_utils::create_summary_stats(info.Env(), _metrics.get_grammar_compile_time());
}

template <class T, class MetricsType>
Napi::Value BasePerfMetricsWrapper<T, MetricsType>::get_raw_metrics(const Napi::CallbackInfo& info) {
    Napi::Object obj = Napi::Object::New(info.Env());
    obj.Set("generateDurations",
            cpp_to_js<std::vector<float>, Napi::Value>(
                info.Env(),
                get_ms(_metrics.raw_metrics, &ov::genai::RawPerfMetrics::generate_durations)));
    obj.Set("tokenizationDurations",
            cpp_to_js<std::vector<float>, Napi::Value>(
                info.Env(),
                get_ms(_metrics.raw_metrics, &ov::genai::RawPerfMetrics::tokenization_durations)));
    obj.Set("detokenizationDurations",
            cpp_to_js<std::vector<float>, Napi::Value>(
                info.Env(),
                get_ms(_metrics.raw_metrics, &ov::genai::RawPerfMetrics::detokenization_durations)));

    obj.Set("timesToFirstToken",
            cpp_to_js<std::vector<float>, Napi::Value>(
                info.Env(),
                get_ms(_metrics.raw_metrics, &ov::genai::RawPerfMetrics::m_times_to_first_token)));
    obj.Set("newTokenTimes",
            cpp_to_js<std::vector<double>, Napi::Value>(
                info.Env(),
                timestamp_to_ms(_metrics.raw_metrics, &ov::genai::RawPerfMetrics::m_new_token_times)));
    obj.Set("tokenInferDurations",
            cpp_to_js<std::vector<float>, Napi::Value>(
                info.Env(),
                get_ms(_metrics.raw_metrics, &ov::genai::RawPerfMetrics::m_token_infer_durations)));
    obj.Set("batchSizes", cpp_to_js<std::vector<size_t>, Napi::Value>(info.Env(), _metrics.raw_metrics.m_batch_sizes));
    obj.Set("durations",
            cpp_to_js<std::vector<float>, Napi::Value>(
                info.Env(),
                get_ms(_metrics.raw_metrics, &ov::genai::RawPerfMetrics::m_durations)));
    obj.Set("inferenceDurations",
            cpp_to_js<std::vector<float>, Napi::Value>(
                info.Env(),
                get_ms(_metrics.raw_metrics, &ov::genai::RawPerfMetrics::m_inference_durations)));

    obj.Set("grammarCompileTimes",
            cpp_to_js<std::vector<float>, Napi::Value>(
                info.Env(),
                get_ms(_metrics.raw_metrics, &ov::genai::RawPerfMetrics::m_grammar_compile_times)));

    return obj;
}

template <class T, class MetricsType>
Napi::Value BasePerfMetricsWrapper<T, MetricsType>::add(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 1, "add()");
    const auto env = info.Env();
    try {
        _metrics += unwrap<MetricsType>(env, info[0]);
    } catch (const std::exception& ex) {
        Napi::TypeError::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return info.This();
}

template <class T, class MetricsType>
MetricsType& BasePerfMetricsWrapper<T, MetricsType>::get_value() {
    return _metrics;
}
