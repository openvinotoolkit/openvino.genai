// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/inpainting_pipeline/pipeline_wrapper.hpp"

#include <filesystem>
#include <thread>

#include "include/helper.hpp"
#include "include/image_decode_worker.hpp"
#include "include/image_generation_inference_thread.hpp"
#include "include/inpainting_pipeline/init_worker.hpp"

InpaintingPipelineWrapper::InpaintingPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<InpaintingPipelineWrapper>(info) {}

Napi::Function InpaintingPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "InpaintingPipeline",
                       {
                           InstanceMethod("init", &InpaintingPipelineWrapper::init),
                           InstanceMethod("generate", &InpaintingPipelineWrapper::generate),
                           InstanceMethod("decode", &InpaintingPipelineWrapper::decode),
                           InstanceMethod("getPerformanceMetrics",
                                          &InpaintingPipelineWrapper::get_performance_metrics),
                           InstanceMethod("getGenerationConfig", &InpaintingPipelineWrapper::get_generation_config),
                           InstanceMethod("setGenerationConfig", &InpaintingPipelineWrapper::set_generation_config),
                       });
}

Napi::Value InpaintingPipelineWrapper::init(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(!this->pipe, "Pipeline is already initialized");
        OPENVINO_ASSERT(!this->is_initializing->load(), "Pipeline is already initializing");
        this->is_initializing->store(true);

        VALIDATE_ARGS_COUNT(info, 4, "init()");
        auto model_path = js_to_cpp<std::filesystem::path>(env, info[0]);
        auto device = js_to_cpp<std::string>(env, info[1]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[2]);
        OPENVINO_ASSERT(info[3].IsFunction(), "init callback is not a function");
        Napi::Function callback = info[3].As<Napi::Function>();

        auto async_worker = new InpaintingInitWorker(callback,
                                                     this->pipe,
                                                     this->is_initializing,
                                                     std::move(model_path),
                                                     std::move(device),
                                                     std::move(properties));
        async_worker->Queue();
    } catch (const std::exception& ex) {
        this->is_initializing->store(false);
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}

Napi::Value InpaintingPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "InpaintingPipeline is not initialized");
        OPENVINO_ASSERT(!this->is_busy->load() && !this->is_generating->load(),
                        "generate() cannot run while another generate() or decode() is in progress");
        this->is_busy->store(true);
        this->is_generating->store(true);

        // generate(prompt, image, mask, properties, streamer, doneCallback)
        VALIDATE_ARGS_COUNT(info, 6, "generate()");
        OPENVINO_ASSERT(info[0].IsString(), "InpaintingPipeline.generate() expects a prompt string.");
        OPENVINO_ASSERT(info[4].IsUndefined() || info[4].IsNull() || info[4].IsFunction(),
                        "InpaintingPipeline.generate() expects streamer to be a function, null or undefined.");
        OPENVINO_ASSERT(info[5].IsFunction(), "generate done callback is not a function");

        auto prompt = js_to_cpp<std::string>(env, info[0]);
        auto image = js_to_cpp<ov::Tensor>(env, info[1]);
        const auto& image_shape = image.get_shape();
        OPENVINO_ASSERT(image.get_element_type() == ov::element::u8,
                        "InpaintingPipeline.generate() expects image tensor with u8 element type, got ",
                        image.get_element_type(),
                        ".");
        OPENVINO_ASSERT(image_shape.size() == 4 && image_shape[0] == 1 && image_shape[3] == 3,
                        "InpaintingPipeline.generate() expects image tensor with batched NHWC shape [1, H, W, 3], got ",
                        image_shape,
                        ".");
        auto mask_image = js_to_cpp<ov::Tensor>(env, info[2]);
        const auto& mask_shape = mask_image.get_shape();
        OPENVINO_ASSERT(mask_image.get_element_type() == ov::element::u8,
                        "InpaintingPipeline.generate() expects mask tensor with u8 element type, got ",
                        mask_image.get_element_type(),
                        ".");
        OPENVINO_ASSERT(mask_shape.size() == 4 && mask_shape[0] == 1 && mask_shape[3] == 3,
                        "InpaintingPipeline.generate() expects mask tensor with batched NHWC shape [1, H, W, 3], got ",
                        mask_shape,
                        ".");
        auto generation_properties = js_to_cpp<ov::AnyMap>(env, info[3]);
        Napi::Function callback = info[5].As<Napi::Function>();

        auto pipe = this->pipe;
        auto is_busy = this->is_busy;
        auto* context = new InferenceThreadContext(this->is_generating, "inpainting_perform_inference_thread");
        context->streamer_exception_header = "Step callback exceptions occurred:";
        context->on_finished = [is_busy] {
            is_busy->store(false);
        };
        context->run_generate = [context, pipe, is_busy, prompt = std::move(prompt), image = std::move(image),
                                 mask_image = std::move(mask_image),
                                 properties = std::move(generation_properties)]() mutable -> JsResultProducer {
            if (context->streamer_tsfn.has_value()) {
                properties["callback"] = make_image_generation_step_callback(context, is_busy);
            }
            ov::Tensor result = pipe->generate(prompt, image, mask_image, properties);
            return [result = std::move(result)](Napi::Env env) -> Napi::Value {
                return cpp_to_js<ov::Tensor, Napi::Value>(env, result);
            };
        };

        // streamer at index 4 can be a function or null/undefined
        if (info[4].IsFunction()) {
            context->streamer_tsfn =
                Napi::ThreadSafeFunction::New(env, info[4].As<Napi::Function>(), "Inpainting_streamer", 0, 1);
        }

        context->callback_tsfn =
            Napi::ThreadSafeFunction::New(env, callback, "Inpainting_generate_callback", 0, 1, [context](Napi::Env) {
                if (context->native_thread.joinable()) {
                    context->native_thread.join();
                }
                delete context;
            });

        context->native_thread = std::thread(perform_generate_thread, context);
    } catch (const std::exception& ex) {
        this->is_busy->store(false);
        this->is_generating->store(false);
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}

Napi::Value InpaintingPipelineWrapper::decode(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "InpaintingPipeline is not initialized");
        OPENVINO_ASSERT(!this->is_busy->load(),
                        "decode() cannot run while another generate() or decode() is in progress");
        VALIDATE_ARGS_COUNT(info, 2, "decode()");
        auto latent = js_to_cpp<ov::Tensor>(env, info[0]);
        OPENVINO_ASSERT(info[1].IsFunction(), "decode callback is not a function");
        Napi::Function callback = info[1].As<Napi::Function>();

        this->is_busy->store(true);
        auto* async_worker =
            new ImageDecodeWorker<ov::genai::InpaintingPipeline>(callback, this->pipe, std::move(latent), this->is_busy);
        async_worker->Queue();
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value InpaintingPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "InpaintingPipeline is not initialized");
        return cpp_to_js<ov::genai::ImageGenerationConfig, Napi::Value>(env, this->pipe->get_generation_config());
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

Napi::Value InpaintingPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "InpaintingPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 1, "setGenerationConfig()");
        OPENVINO_ASSERT(!info[0].IsUndefined() && !info[0].IsNull(), "Generation config cannot be undefined or null");
        auto config = js_to_cpp<ov::genai::ImageGenerationConfig>(env, info[0]);
        this->pipe->set_generation_config(config);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value InpaintingPipelineWrapper::get_performance_metrics(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "InpaintingPipeline is not initialized");
        return cpp_to_js<ov::genai::ImageGenerationPerfMetrics, Napi::Value>(
            env,
            this->pipe->get_performance_metrics());
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}
