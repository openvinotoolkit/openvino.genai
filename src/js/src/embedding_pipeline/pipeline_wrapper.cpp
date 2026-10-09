// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <thread>

#include "include/helper.hpp"
#include "include/embedding_pipeline/pipeline_wrapper.hpp"
#include "include/embedding_pipeline/init_worker.hpp"
#include "include/embedding_pipeline/embed_worker.hpp"

EmbeddingPipelineWrapper::EmbeddingPipelineWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<EmbeddingPipelineWrapper>(info) {};

EmbeddingPipelineWrapper::~EmbeddingPipelineWrapper() {
    if (!this->pipe) {
        return;
    }
    // init() builds the pipeline (and its OpenVINO CPU executor pools) on a libuv worker thread;
    // releasing it on the main V8 GC thread races their teardown and crashes on Windows/macOS.
    std::thread destroyer([pipe = std::move(this->pipe)]() mutable {
        pipe.reset();
    });
    destroyer.join();
}

Napi::Function EmbeddingPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(
        env,
        "EmbeddingPipeline",
        {
            InstanceMethod("init", &EmbeddingPipelineWrapper::init),
            InstanceMethod("embed", &EmbeddingPipelineWrapper::embed),
        }
    );
}

Napi::Value EmbeddingPipelineWrapper::init(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto model_path = js_to_cpp<std::filesystem::path>(env, info[0]);
    auto device = js_to_cpp<std::string>(env, info[1]);
    auto properties = info[2].As<Napi::Object>();
    auto callback = info[3].As<Napi::Function>();

    auto* asyncWorker = new EmbeddingPipelineInitWorker(callback, this->pipe, std::move(model_path), std::move(device), properties);
    asyncWorker->Queue();

    return info.Env().Undefined();
}

Napi::Value EmbeddingPipelineWrapper::embed(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto text = js_to_cpp<ov::genai::StringInputs>(env, info[0]);
    auto images = js_to_cpp<std::vector<ov::Tensor>>(env, info[1]);
    auto videos = js_to_cpp<std::vector<ov::Tensor>>(env, info[2]);
    auto videos_metadata = js_to_cpp<std::vector<ov::genai::VideoMetadata>>(env, info[3]);
    auto properties = js_to_cpp<ov::AnyMap>(env, info[4]);
    auto callback = info[5].As<Napi::Function>();

    auto* asyncWorker = new EmbeddingPipelineEmbedWorker(callback,
                                                         this->pipe,
                                                         std::move(text),
                                                         std::move(images),
                                                         std::move(videos),
                                                         std::move(videos_metadata),
                                                         std::move(properties));
    asyncWorker->Queue();

    return info.Env().Undefined();
}
