// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text2image_pipeline/pipeline_wrapper.hpp"

#include "include/image_decode_worker.hpp"
#include "include/image_generation_inference_thread.hpp"

Text2ImagePipelineWrapper::Text2ImagePipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Text2ImagePipelineWrapper>(info) {}

Napi::Function Text2ImagePipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "Text2ImagePipeline",
                       {InstanceMethod("init", &Text2ImagePipelineWrapper::init),
                        InstanceMethod("generate", &Text2ImagePipelineWrapper::generate),
                        InstanceMethod("decode", &Text2ImagePipelineWrapper::decode),
                        InstanceMethod("getPerformanceMetrics", &Text2ImagePipelineWrapper::get_performance_metrics),
                        InstanceMethod("getGenerationConfig", &Text2ImagePipelineWrapper::get_generation_config),
                        InstanceMethod("setGenerationConfig", &Text2ImagePipelineWrapper::set_generation_config)});
}

Napi::Value Text2ImagePipelineWrapper::init(const Napi::CallbackInfo& info) {
    return initialize_pipeline(info, pipe, state);
}

Napi::Value Text2ImagePipelineWrapper::generate(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "Text2ImagePipeline is not initialized");
        auto operation = std::make_shared<PipelineOperationGuard>(
            state,
            PipelineOperation::GENERATE,
            "generate() cannot run while another generate() or decode() is in progress");
        VALIDATE_ARGS_COUNT(info, 4, "generate()");
        OPENVINO_ASSERT(info[0].IsString(), "Text2ImagePipeline.generate() expects a prompt string.");
        auto prompt = js_to_cpp<std::string>(env, info[0]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[1]);
        OPENVINO_ASSERT(info[2].IsUndefined() || info[2].IsNull() || info[2].IsFunction(),
                        "Text2ImagePipeline.generate() expects streamer to be a function, null or undefined.");
        OPENVINO_ASSERT(info[3].IsFunction(), "generate done callback is not a function");
        auto context = std::make_unique<InferenceThreadContext>(std::move(operation), "text2image_perform_inference_thread");
        if (info[2].IsFunction()) {
            context->streamer =
                std::make_unique<ThreadSafeCallback>(env, info[2].As<Napi::Function>(), "Text2Image_streamer");
        }
        context->run_generate = [context = context.get(),
                                 pipe = pipe,
                                 state = state,
                                 prompt = std::move(prompt),
                                 properties = std::move(properties)]() mutable -> JsResultProducer {
            if (context->streamer) {
                properties["callback"] = make_image_generation_step_callback(context, state);
            }
            auto result = pipe->generate(prompt, properties);
            return [result = std::move(result)](Napi::Env env) {
                return cpp_to_js<ov::Tensor, Napi::Value>(env, result);
            };
        };
        start_inference(info, info[3].As<Napi::Function>(), std::move(context));
        return env.Undefined();
    });
}

Napi::Value Text2ImagePipelineWrapper::decode(const Napi::CallbackInfo& info) {
    return decode_image(info, pipe, state, "Text2ImagePipeline");
}

Napi::Value Text2ImagePipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    return get_pipeline_generation_config(info, pipe, state, "Text2ImagePipeline");
}

Napi::Value Text2ImagePipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    return set_pipeline_generation_config(info, pipe, state, "Text2ImagePipeline");
}

Napi::Value Text2ImagePipelineWrapper::get_performance_metrics(const Napi::CallbackInfo& info) {
    return get_pipeline_performance_metrics(info, pipe, state, "Text2ImagePipeline");
}
