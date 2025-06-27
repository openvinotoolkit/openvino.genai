#pragma once

#include <napi.h>
#include "openvino/genai/rag/text_embedding_pipeline.hpp"

class TextEmbeddingPipelineWrapper : public Napi::ObjectWrap<TextEmbeddingPipelineWrapper> {
    public:
        TextEmbeddingPipelineWrapper(const Napi::CallbackInfo& info);
        static Napi::Function get_class(Napi::Env env);
        Napi::Value init(const Napi::CallbackInfo& info);
        Napi::Value embed_documents(const Napi::CallbackInfo& info);
        Napi::Value embed_documents_async(const Napi::CallbackInfo& info);
        Napi::Value embed_query(const Napi::CallbackInfo& info);
        Napi::Value embed_query_async(const Napi::CallbackInfo& info);
    private:
        std::shared_ptr<ov::genai::TextEmbeddingPipeline> pipe = nullptr;
};
