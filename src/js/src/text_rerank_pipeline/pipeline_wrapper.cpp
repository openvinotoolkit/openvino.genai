// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text_rerank_pipeline/pipeline_wrapper.hpp"

#include "include/base/async_worker.hpp"

TextRerankPipelineWrapper::TextRerankPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<TextRerankPipelineWrapper>(info) {}

Napi::Function TextRerankPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "TextRerankPipeline",
                       {InstanceMethod("init", &TextRerankPipelineWrapper::init),
                        InstanceMethod("rerank", &TextRerankPipelineWrapper::rerank)});
}

Napi::Value TextRerankPipelineWrapper::init(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(!pipe, "Pipeline is already initialized");
        auto operation = std::make_shared<PipelineOperationGuard>(
            state, PipelineOperation::INITIALIZE, "Pipeline is already initializing");
        VALIDATE_ARGS_COUNT(info, 5, "init()");
        auto model_path = js_to_cpp<std::filesystem::path>(env, info[0]);
        auto device = js_to_cpp<std::string>(env, info[1]);
        auto config = js_to_cpp<ov::AnyMap>(env, info[2]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[3]);
        OPENVINO_ASSERT(info[4].IsFunction(), "init callback is not a function");
        queue_pipeline_init(info,
                            info[4].As<Napi::Function>(),
                            std::move(operation),
                            pipe,
                            [model_path = std::move(model_path),
                             device = std::move(device),
                             config = std::move(config),
                             properties = std::move(properties)] {
                                return std::make_shared<ov::genai::TextRerankPipeline>(
                                    model_path, device, ov::genai::TextRerankPipeline::Config(config), properties);
                            });
        return env.Undefined();
    });
}

Napi::Value TextRerankPipelineWrapper::rerank(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "TextRerankPipeline is not initialized");
        auto operation = std::make_shared<PipelineOperationGuard>(
            state, PipelineOperation::ACCESS, "Another reranking is already in progress");
        VALIDATE_ARGS_COUNT(info, 3, "rerank()");
        auto query = js_to_cpp<std::string>(env, info[0]);
        auto documents = js_to_cpp<std::vector<std::string>>(env, info[1]);
        OPENVINO_ASSERT(info[2].IsFunction(), "rerank callback is not a function");
        queue_pipeline_worker(info,
                              info[2].As<Napi::Function>(),
                              std::move(operation),
                              [pipe = pipe, query = std::move(query), documents = std::move(documents)] {
                                  auto results = pipe->rerank(query, documents);
                                  return [results = std::move(results)](Napi::Env env) -> Napi::Value {
                                      return cpp_to_js<std::vector<std::pair<size_t, float>>, Napi::Value>(env, results);
                                  };
                              });
        return env.Undefined();
    });
}
