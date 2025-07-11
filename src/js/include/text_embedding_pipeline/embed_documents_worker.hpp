#pragma once

#include <napi.h>
#include "openvino/genai/rag/text_embedding_pipeline.hpp"

using namespace Napi;

class EmbedDocumentsWorker : public AsyncWorker {
    public:
        EmbedDocumentsWorker(
            Function& callback,
            std::shared_ptr<ov::genai::TextEmbeddingPipeline>& pipe,
            Array documents
        );
        virtual ~EmbedDocumentsWorker(){}

        void Execute() override;
        void OnOK() override;
    private:
        std::shared_ptr<ov::genai::TextEmbeddingPipeline>& pipe;
        std::vector<std::string> documents;
        ov::genai::EmbeddingResults embed_results;
};
