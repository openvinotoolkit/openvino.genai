// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "include/base/async_worker.hpp"

template <typename Pipeline>
Napi::Value decode_image(const Napi::CallbackInfo& info,
                         const std::shared_ptr<Pipeline>& pipeline,
                         const std::shared_ptr<PipelineState>& state,
                         const std::string& name) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        OPENVINO_ASSERT(pipeline, name, " is not initialized");
        auto operation = std::make_shared<PipelineOperationGuard>(
            state, PipelineOperation::DECODE, "decode() cannot run while another generate() or decode() is in progress");
        VALIDATE_ARGS_COUNT(info, 2, "decode()");
        auto latent = js_to_cpp<ov::Tensor>(info.Env(), info[0]);
        OPENVINO_ASSERT(info[1].IsFunction(), "decode callback is not a function");
        queue_pipeline_worker(info,
                              info[1].As<Napi::Function>(),
                              {},
                              [pipeline, latent = std::move(latent), operation = std::move(operation)]() mutable {
                                  // Resume generation after native decode, independently of JavaScript completion.
                                  const auto decode_operation = std::move(operation);
                                  // ov::Tensor for the image pipelines, ov::genai::VideoGenerationResult for
                                  // Text2VideoPipeline.
                                  auto result = pipeline->decode(latent);
                                  return [result = std::move(result)](Napi::Env env) {
                                      return cpp_to_js<std::decay_t<decltype(result)>, Napi::Value>(env, result);
                                  };
                              });
        return info.Env().Undefined();
    });
}
