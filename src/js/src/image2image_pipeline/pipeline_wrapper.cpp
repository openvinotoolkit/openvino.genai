// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/image2image_pipeline/pipeline_wrapper.hpp"

#include "include/image_decode_worker.hpp"
#include "include/image_generation_inference_thread.hpp"

Image2ImagePipelineWrapper::Image2ImagePipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Image2ImagePipelineWrapper>(info) {}

Napi::Function Image2ImagePipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "Image2ImagePipeline",
                       {InstanceMethod("init", &Image2ImagePipelineWrapper::init),
                        InstanceMethod("generate", &Image2ImagePipelineWrapper::generate),
                        InstanceMethod("decode", &Image2ImagePipelineWrapper::decode),
                        InstanceMethod("getPerformanceMetrics", &Image2ImagePipelineWrapper::get_performance_metrics),
                        InstanceMethod("getGenerationConfig", &Image2ImagePipelineWrapper::get_generation_config),
                        InstanceMethod("setGenerationConfig", &Image2ImagePipelineWrapper::set_generation_config)});
}

Napi::Value Image2ImagePipelineWrapper::init(const Napi::CallbackInfo& info) {
    return initialize_pipeline(info, pipe, state);
}

Napi::Value Image2ImagePipelineWrapper::generate(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "Image2ImagePipeline is not initialized");
        auto operation = std::make_shared<PipelineOperationGuard>(
            state,
            PipelineOperation::GENERATE,
            "generate() cannot run while another generate() or decode() is in progress");
        VALIDATE_ARGS_COUNT(info, 5, "generate()");
        OPENVINO_ASSERT(info[0].IsString(), "Image2ImagePipeline.generate() expects a prompt string.");
        auto prompt = js_to_cpp<std::string>(env, info[0]);
        auto image = js_to_cpp<ov::Tensor>(env, info[1]);
        const auto& shape = image.get_shape();
        OPENVINO_ASSERT(image.get_element_type() == ov::element::u8,
                        "Image2ImagePipeline.generate() expects image tensor with u8 element type, got ",
                        image.get_element_type());
        OPENVINO_ASSERT(shape.size() == 4 && shape[0] == 1 && shape[3] == 3,
                        "Image2ImagePipeline.generate() expects image tensor with batched NHWC shape [1, H, W, 3], got ",
                        shape);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[2]);
        OPENVINO_ASSERT(info[3].IsUndefined() || info[3].IsNull() || info[3].IsFunction(),
                        "Image2ImagePipeline.generate() expects streamer to be a function, null or undefined.");
        OPENVINO_ASSERT(info[4].IsFunction(), "generate done callback is not a function");
        auto context = std::make_unique<InferenceThreadContext>(std::move(operation), "image2image_inference_thread");
        if (info[3].IsFunction()) {
            context->streamer =
                std::make_unique<ThreadSafeCallback>(env, info[3].As<Napi::Function>(), "Image2Image_streamer");
        }
        context->run_generate = [context = context.get(),
                                 pipe = pipe,
                                 state = state,
                                 prompt = std::move(prompt),
                                 image = std::move(image),
                                 properties = std::move(properties)]() mutable -> JsResultProducer {
            if (context->streamer) {
                properties["callback"] = make_image_generation_step_callback(context, state);
            }
            auto result = pipe->generate(prompt, image, properties);
            return [result = std::move(result)](Napi::Env env) {
                return cpp_to_js<ov::Tensor, Napi::Value>(env, result);
            };
        };
        start_inference(info, info[4].As<Napi::Function>(), std::move(context));
        return env.Undefined();
    });
}

Napi::Value Image2ImagePipelineWrapper::decode(const Napi::CallbackInfo& info) {
    return decode_image(info, pipe, state, "Image2ImagePipeline");
}

Napi::Value Image2ImagePipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    return get_pipeline_generation_config(info, pipe, state, "Image2ImagePipeline");
}

Napi::Value Image2ImagePipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    return set_pipeline_generation_config(info, pipe, state, "Image2ImagePipeline");
}

Napi::Value Image2ImagePipelineWrapper::get_performance_metrics(const Napi::CallbackInfo& info) {
    return get_pipeline_performance_metrics(info, pipe, state, "Image2ImagePipeline");
}
