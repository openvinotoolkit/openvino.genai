// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/addon.hpp"

#include <napi.h>

#include <thread>

#include "include/chat_history.hpp"
#include "include/llm_pipeline/llm_pipeline_wrapper.hpp"
#include "include/parser.hpp"
#include "include/perf_metrics.hpp"
#include "include/vlm_pipeline/vlm_pipeline_wrapper.hpp"
#include "include/vlm_pipeline/perf_metrics.hpp"
#include "include/text_embedding_pipeline/pipeline_wrapper.hpp"
#include "include/text_rerank_pipeline/pipeline_wrapper.hpp"
#include "include/tokenizer.hpp"

void init_class(Napi::Env env,
                Napi::Object exports,
                std::string class_name,
                Prototype func,
                Napi::FunctionReference& reference) {
    const auto& prototype = func(env);

    reference = Napi::Persistent(prototype);
    exports.Set(class_name, prototype);
}

void set_ov_addon(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "setOpenvinoAddon expects one argument").ThrowAsJavaScriptException();
        return;
    }
    if (info[0].IsUndefined() || info[0].IsNull() || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Passed addon must be an object").ThrowAsJavaScriptException();
        return;
    }

    auto addon_data = env.GetInstanceData<AddonData>();
    if (!addon_data) {
        Napi::TypeError::New(env, "Addon data is not initialized").ThrowAsJavaScriptException();
        return;
    }

    auto ov_addon = info[0].As<Napi::Object>();
    addon_data->openvino_addon = Napi::Persistent(ov_addon);
}

// Define the addon initialization function
Napi::Object init_module(Napi::Env env, Napi::Object exports) {
    auto addon_data = new AddonData();
    env.SetInstanceData<AddonData>(addon_data);

    init_class(env, exports, "LLMPipeline", &LLMPipelineWrapper::get_class, addon_data->core);
    init_class(env, exports, "VLMPipeline", &VLMPipelineWrapper::get_class, addon_data->vlm_pipeline);
    init_class(env, exports, "TextEmbeddingPipeline", &TextEmbeddingPipelineWrapper::get_class, addon_data->core);
    init_class(env,
               exports,
               "TextRerankPipeline",
               &TextRerankPipelineWrapper::get_class,
               addon_data->text_rerank_pipeline);
    init_class(env, exports, "Tokenizer", &TokenizerWrapper::get_class, addon_data->tokenizer);
    init_class(env, exports, "PerfMetrics", &PerfMetricsWrapper::get_class, addon_data->perf_metrics);
    init_class(env, exports, "VLMPerfMetrics", &VLMPerfMetricsWrapper::get_class, addon_data->vlm_perf_metrics);
    init_class(env, exports, "ChatHistory", &ChatHistoryWrap::get_class, addon_data->chat_history);
    init_class(env, exports, "ReasoningParser", &ReasoningParserWrapper::get_class, addon_data->reasoning_parser);
    init_class(env,
               exports,
               "DeepSeekR1ReasoningParser",
               &DeepSeekR1ReasoningParserWrapper::get_class,
               addon_data->deepseek_r1_reasoning_parser);
    init_class(env,
               exports,
               "Phi4ReasoningParser",
               &Phi4ReasoningParserWrapper::get_class,
               addon_data->phi4_reasoning_parser);
    init_class(env,
               exports,
               "Llama3PythonicToolParser",
               &Llama3PythonicToolParserWrapper::get_class,
               addon_data->llama3_pythonic_tool_parser);
    init_class(env,
               exports,
               "Llama3JsonToolParser",
               &Llama3JsonToolParserWrapper::get_class,
               addon_data->llama3_json_tool_parser);

    // Expose a helper to set the openvino-node addon from JS (useful for ESM)
    exports.Set("setOpenvinoAddon", Napi::Function::New(env, set_ov_addon));

    return exports;
}

// Register the addon with Node.js
NODE_API_MODULE(openvino-genai-node, init_module)
