#include "include/js_perf_metrics.hpp"

#include "include/helper.hpp"
#include "include/addon.hpp"

PerfMetricsWrapper::PerfMetricsWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<PerfMetricsWrapper>(info),
      _metrics{} {};

Napi::Function PerfMetricsWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "PerfMetrics",
                       {
                           InstanceMethod("getLoadTime", &PerfMetricsWrapper::get_load_time),
                           InstanceMethod("toString", &PerfMetricsWrapper::to_string),
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

Napi::Value PerfMetricsWrapper::to_string(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), "PerfMetrics.toString(). Not implemented.");
}
