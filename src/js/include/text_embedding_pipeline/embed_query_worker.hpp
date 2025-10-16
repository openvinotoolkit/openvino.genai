#pragma once

#include <napi.h>
#include "openvino/genai/rag/text_embedding_pipeline.hpp"

using namespace Napi;

class EmbedQueryWorker : public AsyncWorker {
    public:
        EmbedQueryWorker(
            Function& callback,
            std::shared_ptr<ov::genai::TextEmbeddingPipeline>& pipe,
            String text);
        virtual ~EmbedQueryWorker(){}

        void Execute() override;
        void OnOK() override;
    private:
        std::shared_ptr<ov::genai::TextEmbeddingPipeline>& pipe;
        std::string text;
        ov::genai::EmbeddingResult embed_result;
};
