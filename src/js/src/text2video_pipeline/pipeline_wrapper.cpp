// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text2video_pipeline/pipeline_wrapper.hpp"

#include "include/image_generation/decode_worker.hpp"
#include "include/image_generation/inference_thread.hpp"

Text2VideoPipelineWrapper::Text2VideoPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Text2VideoPipelineWrapper>(info) {}

Napi::Function Text2VideoPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "Text2VideoPipeline",
                       {InstanceMethod("init", &Text2VideoPipelineWrapper::init),
                        InstanceMethod("generate", &Text2VideoPipelineWrapper::generate),
                        InstanceMethod("decode", &Text2VideoPipelineWrapper::decode),
                        InstanceMethod("getGenerationConfig", &Text2VideoPipelineWrapper::get_generation_config),
                        InstanceMethod("setGenerationConfig", &Text2VideoPipelineWrapper::set_generation_config)});
}

Napi::Value Text2VideoPipelineWrapper::init(const Napi::CallbackInfo& info) {
    return initialize_pipeline(info, pipe, state);
}

Napi::Value Text2VideoPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "Text2VideoPipeline is not initialized");
        auto operation = std::make_shared<PipelineOperationGuard>(
            state,
            PipelineOperation::GENERATE,
            "generate() cannot run while another generate() or decode() is in progress");
        VALIDATE_ARGS_COUNT(info, 4, "generate()");
        OPENVINO_ASSERT(info[0].IsString(), "Text2VideoPipeline.generate() expects a prompt string.");
        auto prompt = js_to_cpp<std::string>(env, info[0]);
        // Apply the options on top of the current config, so they are validated the same way as in
        // setGenerationConfig() and only affect this call
        auto config = pipe->get_generation_config();
        update_video_generation_config(env, info[1], config);
        auto properties = to_generation_properties(config);
        OPENVINO_ASSERT(info[2].IsUndefined() || info[2].IsNull() || info[2].IsFunction(),
                        "Text2VideoPipeline.generate() expects callback to be a function, null or undefined.");
        OPENVINO_ASSERT(info[3].IsFunction(), "generate done callback is not a function");
        auto context = std::make_unique<InferenceThreadContext>(std::move(operation), "text2video_inference_thread");
        if (info[2].IsFunction()) {
            context->streamer =
                std::make_unique<ThreadSafeCallback>(env, info[2].As<Napi::Function>(), "Text2Video_callback");
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
                return cpp_to_js<ov::genai::VideoGenerationResult, Napi::Value>(env, result);
            };
        };
        start_inference(info, info[3].As<Napi::Function>(), std::move(context));
        return env.Undefined();
    });
}

Napi::Value Text2VideoPipelineWrapper::decode(const Napi::CallbackInfo& info) {
    return decode_image(info, pipe, state, "Text2VideoPipeline");
}

Napi::Value Text2VideoPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    return get_pipeline_generation_config(info, pipe, state, "Text2VideoPipeline");
}

Napi::Value Text2VideoPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    return set_pipeline_generation_config(info, pipe, state, "Text2VideoPipeline");
}
