// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text_embedding_pipeline/pipeline_wrapper.hpp"

#include "include/base/async_worker.hpp"

TextEmbeddingPipelineWrapper::TextEmbeddingPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<TextEmbeddingPipelineWrapper>(info) {}

Napi::Function TextEmbeddingPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "TextEmbeddingPipeline",
                       {InstanceMethod("init", &TextEmbeddingPipelineWrapper::init),
                        InstanceMethod("embedQuerySync", &TextEmbeddingPipelineWrapper::embed_query),
                        InstanceMethod("embedDocumentsSync", &TextEmbeddingPipelineWrapper::embed_documents),
                        InstanceMethod("embedQuery", &TextEmbeddingPipelineWrapper::embed_query_async),
                        InstanceMethod("embedDocuments", &TextEmbeddingPipelineWrapper::embed_documents_async)});
}

Napi::Value TextEmbeddingPipelineWrapper::init(const Napi::CallbackInfo& info) {
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
                                return std::make_shared<ov::genai::TextEmbeddingPipeline>(
                                    model_path, device, ov::genai::TextEmbeddingPipeline::Config(config), properties);
                            });
        return env.Undefined();
    });
}

Napi::Value TextEmbeddingPipelineWrapper::embed_query(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        OPENVINO_ASSERT(pipe, "TextEmbeddingPipeline is not initialized");
        const PipelineOperationGuard operation(state, PipelineOperation::ACCESS, "Another embedding is already in progress");
        VALIDATE_ARGS_COUNT(info, 1, "embedQuerySync()");
        const auto result = pipe->embed_query(js_to_cpp<std::string>(info.Env(), info[0]));
        return cpp_to_js<ov::genai::EmbeddingResult, Napi::Value>(info.Env(), result);
    });
}

Napi::Value TextEmbeddingPipelineWrapper::embed_documents(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        OPENVINO_ASSERT(pipe, "TextEmbeddingPipeline is not initialized");
        const PipelineOperationGuard operation(state, PipelineOperation::ACCESS, "Another embedding is already in progress");
        VALIDATE_ARGS_COUNT(info, 1, "embedDocumentsSync()");
        const auto documents = js_to_cpp<std::vector<std::string>>(info.Env(), info[0]);
        const auto results = pipe->embed_documents(documents);
        return cpp_to_js<ov::genai::EmbeddingResults, Napi::Value>(info.Env(), results);
    });
}

Napi::Value TextEmbeddingPipelineWrapper::embed_query_async(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "TextEmbeddingPipeline is not initialized");
        auto operation =
            std::make_shared<PipelineOperationGuard>(state, PipelineOperation::ACCESS, "Another embedding is already in progress");
        VALIDATE_ARGS_COUNT(info, 2, "embedQuery()");
        auto text = js_to_cpp<std::string>(env, info[0]);
        OPENVINO_ASSERT(info[1].IsFunction(), "embedQuery callback is not a function");
        queue_pipeline_worker(info, info[1].As<Napi::Function>(), std::move(operation), [pipe = pipe, text = std::move(text)] {
            auto result = pipe->embed_query(text);
            return [result = std::move(result)](Napi::Env env) -> Napi::Value {
                return cpp_to_js<ov::genai::EmbeddingResult, Napi::Value>(env, result);
            };
        });
        return env.Undefined();
    });
}

Napi::Value TextEmbeddingPipelineWrapper::embed_documents_async(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "TextEmbeddingPipeline is not initialized");
        auto operation =
            std::make_shared<PipelineOperationGuard>(state, PipelineOperation::ACCESS, "Another embedding is already in progress");
        VALIDATE_ARGS_COUNT(info, 2, "embedDocuments()");
        auto documents = js_to_cpp<std::vector<std::string>>(env, info[0]);
        OPENVINO_ASSERT(info[1].IsFunction(), "embedDocuments callback is not a function");
        queue_pipeline_worker(info,
                              info[1].As<Napi::Function>(),
                              std::move(operation),
                              [pipe = pipe, documents = std::move(documents)] {
                                  auto results = pipe->embed_documents(documents);
                                  return [results = std::move(results)](Napi::Env env) -> Napi::Value {
                                      return cpp_to_js<ov::genai::EmbeddingResults, Napi::Value>(env, results);
                                  };
                              });
        return env.Undefined();
    });
}
