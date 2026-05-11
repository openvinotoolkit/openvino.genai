// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text2image_pipeline/perf_metrics.hpp"

#include "include/addon.hpp"
#include "include/base/perf_metrics.hpp"
#include "include/helper.hpp"

Text2ImagePerfMetricsWrapper::Text2ImagePerfMetricsWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Text2ImagePerfMetricsWrapper>(info) {}

Napi::Function Text2ImagePerfMetricsWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "Text2ImagePerfMetrics",
                       {
                           InstanceMethod("getLoadTime", &Text2ImagePerfMetricsWrapper::get_load_time),
                           InstanceMethod("getGenerateDuration", &Text2ImagePerfMetricsWrapper::get_generate_duration),
                           InstanceMethod("getIterationDuration", &Text2ImagePerfMetricsWrapper::get_iteration_duration),
                           InstanceMethod("getUnetInferDuration", &Text2ImagePerfMetricsWrapper::get_unet_infer_duration),
                           InstanceMethod("getTransformerInferDuration", &Text2ImagePerfMetricsWrapper::get_transformer_infer_duration),
                           InstanceMethod("getVaeEncoderInferDuration", &Text2ImagePerfMetricsWrapper::get_vae_encoder_infer_duration),
                           InstanceMethod("getVaeDecoderInferDuration", &Text2ImagePerfMetricsWrapper::get_vae_decoder_infer_duration),
                           InstanceMethod("getTextEncoderInferDuration", &Text2ImagePerfMetricsWrapper::get_text_encoder_infer_duration),
                           InstanceMethod("getInferenceDuration", &Text2ImagePerfMetricsWrapper::get_inference_duration),
                           InstanceAccessor<&Text2ImagePerfMetricsWrapper::get_raw_metrics>("rawMetrics"),
                       });
}

Napi::Object Text2ImagePerfMetricsWrapper::wrap(Napi::Env env, const ov::genai::ImageGenerationPerfMetrics& metrics) {
    const auto& prototype = env.GetInstanceData<AddonData>()->text2image_perf_metrics;
    OPENVINO_ASSERT(prototype, "Invalid pointer to prototype.");
    auto obj = prototype.New({});
    const auto wrapper = Napi::ObjectWrap<Text2ImagePerfMetricsWrapper>::Unwrap(obj);
    wrapper->_metrics = metrics;
    return obj;
}

ov::genai::ImageGenerationPerfMetrics& Text2ImagePerfMetricsWrapper::get_value() {
    return _metrics;
}

Napi::Value Text2ImagePerfMetricsWrapper::get_load_time(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getLoadTime()");
    return Napi::Number::New(info.Env(), _metrics.get_load_time());
}

Napi::Value Text2ImagePerfMetricsWrapper::get_generate_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getGenerateDuration()");
    return Napi::Number::New(info.Env(), _metrics.get_generate_duration());
}

Napi::Value Text2ImagePerfMetricsWrapper::get_iteration_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getIterationDuration()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_iteration_duration());
}

Napi::Value Text2ImagePerfMetricsWrapper::get_unet_infer_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getUnetInferDuration()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_unet_infer_duration());
}

Napi::Value Text2ImagePerfMetricsWrapper::get_transformer_infer_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getTransformerInferDuration()");
    return perf_utils::create_mean_std_pair(info.Env(), _metrics.get_transformer_infer_duration());
}

Napi::Value Text2ImagePerfMetricsWrapper::get_vae_encoder_infer_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getVaeEncoderInferDuration()");
    return Napi::Number::New(info.Env(), _metrics.get_vae_encoder_infer_duration());
}

Napi::Value Text2ImagePerfMetricsWrapper::get_vae_decoder_infer_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getVaeDecoderInferDuration()");
    return Napi::Number::New(info.Env(), _metrics.get_vae_decoder_infer_duration());
}

Napi::Value Text2ImagePerfMetricsWrapper::get_text_encoder_infer_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getTextEncoderInferDuration()");
    return cpp_map_to_js_object(info.Env(), _metrics.get_text_encoder_infer_duration());
}

Napi::Value Text2ImagePerfMetricsWrapper::get_inference_duration(const Napi::CallbackInfo& info) {
    VALIDATE_ARGS_COUNT(info, 0, "getInferenceDuration()");
    return Napi::Number::New(info.Env(), _metrics.get_inference_duration());
}

Napi::Value Text2ImagePerfMetricsWrapper::get_raw_metrics(const Napi::CallbackInfo& info) {
    return cpp_to_js<ov::genai::RawImageGenerationPerfMetrics, Napi::Value>(info.Env(), _metrics.raw_metrics);
}
