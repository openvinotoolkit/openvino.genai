#include <napi.h>
#include <thread>

#include "include/addon.hpp"

#include "include/perf_metrics.hpp"
#include "include/llm_pipeline/llm_pipeline_wrapper.hpp"
#include "include/text_embedding_pipeline/pipeline_wrapper.hpp"
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

// Define the addon initialization function
Napi::Object init_module(Napi::Env env, Napi::Object exports) {
    auto addon_data = new AddonData();
    env.SetInstanceData<AddonData>(addon_data);

    init_class(env, exports, "LLMPipeline", &LLMPipelineWrapper::get_class, addon_data->core);
    init_class(env, exports, "TextEmbeddingPipeline", &TextEmbeddingPipelineWrapper::get_class, addon_data->core);
    init_class(env, exports, "Tokenizer", &TokenizerWrapper::get_class, addon_data->tokenizer);
    init_class(env, exports, "PerfMetrics", &PerfMetricsWrapper::get_class, addon_data->perf_metrics);

    return exports;
}

// Register the addon with Node.js
NODE_API_MODULE(openvino-genai-node, init_module)
