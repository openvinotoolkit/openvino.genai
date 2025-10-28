#include "include/perf_metrics.hpp"

#include "bindings_utils.hpp"
#include "include/addon.hpp"
#include "include/helper.hpp"

using ov::genai::common_bindings::utils::get_ms;
using ov::genai::common_bindings::utils::timestamp_to_ms;

PerfMetricsWrapper::PerfMetricsWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<PerfMetricsWrapper>(info),
      _metrics{} {};

Napi::Function PerfMetricsWrapper::get_class(Napi::Env env) {
    return DefineClass(
        env,
        "PerfMetrics",
        {
            InstanceMethod("getLoadTime", &PerfMetricsWrapper::get_load_time),
            InstanceMethod("getNumGeneratedTokens", &PerfMetricsWrapper::get_num_generated_tokens),
            InstanceMethod("getNumInputTokens", &PerfMetricsWrapper::get_num_input_tokens),
            InstanceMethod("getTTFT", &PerfMetricsWrapper::get_ttft),
            InstanceMethod("getTPOT", &PerfMetricsWrapper::get_tpot),
            InstanceMethod("getIPOT", &PerfMetricsWrapper::get_ipot),
            InstanceMethod("getThroughput", &PerfMetricsWrapper::get_throughput),
            InstanceMethod("getInferenceDuration", &PerfMetricsWrapper::get_inference_duration),
            InstanceMethod("getGenerateDuration", &PerfMetricsWrapper::get_generate_duration),
            InstanceMethod("getTokenizationDuration", &PerfMetricsWrapper::get_tokenization_duration),
            InstanceMethod("getDetokenizationDuration", &PerfMetricsWrapper::get_detokenization_duration),
            InstanceMethod("getGrammarCompilerInitTimes", &PerfMetricsWrapper::get_grammar_compiler_init_times),
            InstanceMethod("getGrammarCompileTime", &PerfMetricsWrapper::get_grammar_compile_time),
            InstanceAccessor<&PerfMetricsWrapper::get_raw_metrics>("rawMetrics"),
            InstanceMethod("add", &PerfMetricsWrapper::add),
        });
}

Napi::Object PerfMetricsWrapper::wrap(Napi::Env env, const ov::genai::PerfMetrics& metrics) {
    const auto& prototype = env.GetInstanceData<AddonData>()->perf_metrics;
    OPENVINO_ASSERT(prototype, "Invalid pointer to prototype.");
    auto obj = prototype.New({});
    const auto m_ptr = Napi::ObjectWrap<PerfMetricsWrapper>::Unwrap(obj);
    m_ptr->_metrics = metrics;
    return obj;
}

Napi::Value PerfMetricsWrapper::get_load_time(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getLoadTime()");
    return Napi::Number::New(info.Env(), _metrics.get_load_time());
}

Napi::Value PerfMetricsWrapper::get_num_generated_tokens(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getNumGeneratedTokens()");
    return Napi::Number::New(info.Env(), _metrics.get_num_generated_tokens());
}

Napi::Value PerfMetricsWrapper::get_num_input_tokens(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getNumInputTokens()");
    return Napi::Number::New(info.Env(), _metrics.get_num_input_tokens());
}

Napi::Object create_mean_std_pair(Napi::Env env, const ov::genai::MeanStdPair& pair) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("mean", Napi::Number::New(env, pair.mean));
    obj.Set("std", Napi::Number::New(env, pair.std));
    return obj;
}

Napi::Object create_summary_stats(Napi::Env env, const ov::genai::SummaryStats& stats) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("mean", Napi::Number::New(env, stats.mean));
    obj.Set("std", Napi::Number::New(env, stats.std));
    obj.Set("min", Napi::Number::New(env, stats.min));
    obj.Set("max", Napi::Number::New(env, stats.max));
    return obj;
}

Napi::Value PerfMetricsWrapper::get_ttft(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getTTFT()");
    return create_mean_std_pair(info.Env(), _metrics.get_ttft());
}

Napi::Value PerfMetricsWrapper::get_tpot(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getTPOT()");
    return create_mean_std_pair(info.Env(), _metrics.get_tpot());
}

Napi::Value PerfMetricsWrapper::get_ipot(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getIPOT()");
    return create_mean_std_pair(info.Env(), _metrics.get_ipot());
}

Napi::Value PerfMetricsWrapper::get_throughput(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getThroughput()");
    return create_mean_std_pair(info.Env(), _metrics.get_throughput());
}

Napi::Value PerfMetricsWrapper::get_inference_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getInferenceDuration()");
    return create_mean_std_pair(info.Env(), _metrics.get_inference_duration());
}

Napi::Value PerfMetricsWrapper::get_generate_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getGenerateDuration()");
    return create_mean_std_pair(info.Env(), _metrics.get_generate_duration());
}

Napi::Value PerfMetricsWrapper::get_tokenization_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getTokenizationDuration()");
    return create_mean_std_pair(info.Env(), _metrics.get_tokenization_duration());
}

Napi::Value PerfMetricsWrapper::get_grammar_compiler_init_times(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getGrammarCompilerInitTimes()");
    return cpp_map_to_js_object(info.Env(), _metrics.get_grammar_compiler_init_times());
}

Napi::Value PerfMetricsWrapper::get_grammar_compile_time(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getGrammarCompileTime()");
    return create_summary_stats(info.Env(), _metrics.get_grammar_compile_time());
};

Napi::Value PerfMetricsWrapper::get_detokenization_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getDetokenizationDuration()");
    return create_mean_std_pair(info.Env(), _metrics.get_detokenization_duration());
}

Napi::Value PerfMetricsWrapper::get_raw_metrics(const Napi::CallbackInfo& info) {
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

Napi::Value PerfMetricsWrapper::add(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 1, "add()");
    const auto env = info.Env();
    try {
        _metrics += unwrap<ov::genai::PerfMetrics>(env, info[0]);
    } catch (const std::exception& ex) {
        Napi::TypeError::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return info.This();
}

ov::genai::PerfMetrics& PerfMetricsWrapper::get_value() {
    return _metrics;
}
