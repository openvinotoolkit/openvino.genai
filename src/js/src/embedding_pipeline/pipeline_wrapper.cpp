// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/embedding_pipeline/pipeline_wrapper.hpp"

#include "include/base/async_worker.hpp"

EmbeddingPipelineWrapper::EmbeddingPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<EmbeddingPipelineWrapper>(info) {}

Napi::Function EmbeddingPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "EmbeddingPipeline",
                       {InstanceMethod("init", &EmbeddingPipelineWrapper::init),
                        InstanceMethod("embed", &EmbeddingPipelineWrapper::embed)});
}

Napi::Value EmbeddingPipelineWrapper::init(const Napi::CallbackInfo& info) {
    return initialize_pipeline(info, pipe, state);
}

Napi::Value EmbeddingPipelineWrapper::embed(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "EmbeddingPipeline is not initialized");
        auto operation = std::make_shared<PipelineOperationGuard>(
            state, PipelineOperation::ACCESS, "Another embedding is already in progress");
        VALIDATE_ARGS_COUNT(info, 6, "embed()");
        auto text = js_to_cpp<ov::genai::StringInputs>(env, info[0]);
        auto images = js_to_cpp<std::vector<ov::Tensor>>(env, info[1]);
        auto videos = js_to_cpp<std::vector<ov::Tensor>>(env, info[2]);
        auto videos_metadata = js_to_cpp<std::vector<ov::genai::VideoMetadata>>(env, info[3]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[4]);
        OPENVINO_ASSERT(info[5].IsFunction(), "embed callback is not a function");
        queue_pipeline_worker(info,
                              info[5].As<Napi::Function>(),
                              std::move(operation),
                              [pipe = pipe,
                               text = std::move(text),
                               images = std::move(images),
                               videos = std::move(videos),
                               videos_metadata = std::move(videos_metadata),
                               properties = std::move(properties)] {
                                  auto result = pipe->embed(text, images, videos, videos_metadata, properties);
                                  return [result = std::move(result)](Napi::Env env) -> Napi::Value {
                                      return cpp_to_js<ov::genai::EmbedResult, Napi::Value>(env, result);
                                  };
                              });
        return env.Undefined();
    });
}
