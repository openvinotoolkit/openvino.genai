// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text_rerank_pipeline/pipeline_wrapper.hpp"

#include "include/helper.hpp"
#include "include/text_rerank_pipeline/init_worker.hpp"
#include "include/text_rerank_pipeline/rerank_worker.hpp"

TextRerankPipelineWrapper::TextRerankPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<TextRerankPipelineWrapper>(info) {}

Napi::Function TextRerankPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "TextRerankPipeline",
                       {
                           InstanceMethod("init", &TextRerankPipelineWrapper::init),
                           InstanceMethod("rerank", &TextRerankPipelineWrapper::rerank),
                       });
}

Napi::Value TextRerankPipelineWrapper::init(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(!this->pipe, "Pipeline is already initialized");
        OPENVINO_ASSERT(!*this->is_initializing, "Pipeline is already initializing");
        *this->is_initializing = true;

        VALIDATE_ARGS_COUNT(info, 5, "init()");
        auto model_path = js_to_cpp<std::string>(env, info[0]);
        auto device = js_to_cpp<std::string>(env, info[1]);
        auto config = js_to_cpp<ov::AnyMap>(env, info[2]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[3]);
        OPENVINO_ASSERT(info[4].IsFunction(), "init callback is not a function");
        Napi::Function callback = info[4].As<Napi::Function>();

        auto async_worker = new RerankInitWorker(callback,
                                                 this->pipe,
                                                 this->is_initializing,
                                                 std::move(model_path),
                                                 std::move(device),
                                                 std::move(config),
                                                 std::move(properties));
        async_worker->Queue();
    } catch (const std::exception& ex) {
        *this->is_initializing = false;
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}

Napi::Value TextRerankPipelineWrapper::rerank(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "TextRerankPipeline is not initialized");
        OPENVINO_ASSERT(!*this->is_reranking, "Another reranking is already in progress");
        *this->is_reranking = true;

        VALIDATE_ARGS_COUNT(info, 3, "rerank()");
        auto query = js_to_cpp<std::string>(env, info[0]);
        auto documents = js_to_cpp<std::vector<std::string>>(env, info[1]);
        OPENVINO_ASSERT(info[2].IsFunction(), "rerank callback is not a function");
        auto callback = info[2].As<Napi::Function>();

        auto async_worker =
            new RerankWorker(callback, this->pipe, this->is_reranking, std::move(query), std::move(documents));
        async_worker->Queue();
    } catch (const std::exception& ex) {
        *this->is_reranking = false;
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}
