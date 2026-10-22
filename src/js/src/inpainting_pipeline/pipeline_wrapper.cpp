// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/inpainting_pipeline/pipeline_wrapper.hpp"

#include "include/image_decode_worker.hpp"
#include "include/image_generation_inference_thread.hpp"

InpaintingPipelineWrapper::InpaintingPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<InpaintingPipelineWrapper>(info) {}

Napi::Function InpaintingPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "InpaintingPipeline",
                       {InstanceMethod("init", &InpaintingPipelineWrapper::init),
                        InstanceMethod("generate", &InpaintingPipelineWrapper::generate),
                        InstanceMethod("decode", &InpaintingPipelineWrapper::decode),
                        InstanceMethod("getPerformanceMetrics", &InpaintingPipelineWrapper::get_performance_metrics),
                        InstanceMethod("getGenerationConfig", &InpaintingPipelineWrapper::get_generation_config),
                        InstanceMethod("setGenerationConfig", &InpaintingPipelineWrapper::set_generation_config)});
}

Napi::Value InpaintingPipelineWrapper::init(const Napi::CallbackInfo& info) {
    return initialize_pipeline(info, pipe, state);
}

Napi::Value InpaintingPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "InpaintingPipeline is not initialized");
        auto operation = std::make_shared<PipelineOperationGuard>(
            state,
            PipelineOperation::GENERATE,
            "generate() cannot run while another generate() or decode() is in progress");
        VALIDATE_ARGS_COUNT(info, 6, "generate()");
        OPENVINO_ASSERT(info[0].IsString(), "InpaintingPipeline.generate() expects a prompt string.");
        auto prompt = js_to_cpp<std::string>(env, info[0]);
        auto image = js_to_cpp<ov::Tensor>(env, info[1]);
        const auto& shape = image.get_shape();
        OPENVINO_ASSERT(image.get_element_type() == ov::element::u8,
                        "InpaintingPipeline.generate() expects image tensor with u8 element type, got ",
                        image.get_element_type());
        OPENVINO_ASSERT(shape.size() == 4 && shape[0] == 1 && shape[3] == 3,
                        "InpaintingPipeline.generate() expects image tensor with batched NHWC shape [1, H, W, 3], got ",
                        shape);
        auto mask = js_to_cpp<ov::Tensor>(env, info[2]);
        const auto& mask_shape = mask.get_shape();
        OPENVINO_ASSERT(mask.get_element_type() == ov::element::u8,
                        "InpaintingPipeline.generate() expects mask tensor with u8 element type, got ",
                        mask.get_element_type());
        OPENVINO_ASSERT(mask_shape.size() == 4 && mask_shape[0] == 1 && mask_shape[3] == 3,
                        "InpaintingPipeline.generate() expects mask tensor with batched NHWC shape [1, H, W, 3], got ",
                        mask_shape);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[3]);
        OPENVINO_ASSERT(info[4].IsUndefined() || info[4].IsNull() || info[4].IsFunction(),
                        "InpaintingPipeline.generate() expects streamer to be a function, null or undefined.");
        OPENVINO_ASSERT(info[5].IsFunction(), "generate done callback is not a function");
        auto context = std::make_unique<InferenceThreadContext>(std::move(operation), "inpainting_perform_inference_thread");
        if (info[4].IsFunction()) {
            context->streamer =
                std::make_unique<ThreadSafeCallback>(env, info[4].As<Napi::Function>(), "Inpainting_streamer");
        }
        context->run_generate = [context = context.get(),
                                 pipe = pipe,
                                 state = state,
                                 prompt = std::move(prompt),
                                 image = std::move(image),
                                 mask = std::move(mask),
                                 properties = std::move(properties)]() mutable -> JsResultProducer {
            if (context->streamer) {
                properties["callback"] = make_image_generation_step_callback(context, state);
            }
            auto result = pipe->generate(prompt, image, mask, properties);
            return [result = std::move(result)](Napi::Env env) {
                return cpp_to_js<ov::Tensor, Napi::Value>(env, result);
            };
        };
        start_inference(info, info[5].As<Napi::Function>(), std::move(context));
        return env.Undefined();
    });
}

Napi::Value InpaintingPipelineWrapper::decode(const Napi::CallbackInfo& info) {
    return decode_image(info, pipe, state, "InpaintingPipeline");
}

Napi::Value InpaintingPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    return get_pipeline_generation_config(info, pipe, state, "InpaintingPipeline");
}

Napi::Value InpaintingPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    return set_pipeline_generation_config(info, pipe, state, "InpaintingPipeline");
}

Napi::Value InpaintingPipelineWrapper::get_performance_metrics(const Napi::CallbackInfo& info) {
    return get_pipeline_performance_metrics(info, pipe, state, "InpaintingPipeline");
}
