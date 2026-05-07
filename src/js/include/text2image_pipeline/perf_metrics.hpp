// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/genai/image_generation/image_generation_perf_metrics.hpp"

class Text2ImagePerfMetricsWrapper : public Napi::ObjectWrap<Text2ImagePerfMetricsWrapper> {
public:
    Text2ImagePerfMetricsWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);
    static Napi::Object wrap(Napi::Env env, const ov::genai::ImageGenerationPerfMetrics& metrics);

    ov::genai::ImageGenerationPerfMetrics& get_value();

    Napi::Value get_load_time(const Napi::CallbackInfo& info);
    Napi::Value get_generate_duration(const Napi::CallbackInfo& info);
    Napi::Value get_iteration_duration(const Napi::CallbackInfo& info);
    Napi::Value get_unet_infer_duration(const Napi::CallbackInfo& info);
    Napi::Value get_transformer_infer_duration(const Napi::CallbackInfo& info);
    Napi::Value get_vae_encoder_infer_duration(const Napi::CallbackInfo& info);
    Napi::Value get_vae_decoder_infer_duration(const Napi::CallbackInfo& info);
    Napi::Value get_text_encoder_infer_duration(const Napi::CallbackInfo& info);
    Napi::Value get_inference_duration(const Napi::CallbackInfo& info);
    Napi::Value get_raw_metrics(const Napi::CallbackInfo& info);

private:
    ov::genai::ImageGenerationPerfMetrics _metrics;
};
