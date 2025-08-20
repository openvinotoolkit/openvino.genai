#pragma once

#include <napi.h>
#include "openvino/genai/rag/text_embedding_pipeline.hpp"

using namespace Napi;

class EmbeddingInitWorker : public AsyncWorker {
    public:
        EmbeddingInitWorker(
            Function& callback,
            std::shared_ptr<ov::genai::TextEmbeddingPipeline>& pipe,
            const std::string model_path,
            std::string device,
            Object config,
            Object properties
        );
        virtual ~EmbeddingInitWorker(){}
        void Execute() override;
        void OnOK() override;
    private:
        std::shared_ptr<ov::genai::TextEmbeddingPipeline>& pipe;
        std::string model_path;
        std::string device;
        ov::AnyMap config;
        ov::AnyMap properties;
};
