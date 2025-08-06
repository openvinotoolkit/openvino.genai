#include "include/js_perf_metrics.hpp"

PerfMetricsWrapper::PerfMetricsWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<PerfMetricsWrapper>(info) {};

Napi::Function PerfMetricsWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "PerfMetrics",
                       {
                           InstanceMethod("toString", &PerfMetricsWrapper::to_string),
                       });
}

Napi::Value PerfMetricsWrapper::to_string(const Napi::CallbackInfo& info) {
    printf("PerfMetricsWrapper::to_string called\n");
    return Napi::String::New(info.Env(), "PerfMetrics.toString(). Not implemented.");
}
