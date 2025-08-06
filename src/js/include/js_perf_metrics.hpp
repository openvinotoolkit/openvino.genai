#pragma once

#include <napi.h>

#include "openvino/genai/perf_metrics.hpp"

class PerfMetricsWrapper : public Napi::ObjectWrap<PerfMetricsWrapper> {
public:
    PerfMetricsWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);
    Napi::Value to_string(const Napi::CallbackInfo& info);
};
