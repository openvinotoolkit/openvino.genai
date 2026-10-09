// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/helper.hpp"
#include "include/embedding_pipeline/embed_worker.hpp"

EmbeddingPipelineEmbedWorker::EmbeddingPipelineEmbedWorker(
    Function& callback,
    std::shared_ptr<ov::genai::EmbeddingPipeline>& pipe,
    ov::genai::StringInputs text,
    std::vector<ov::Tensor> images,
    std::vector<ov::Tensor> videos,
    std::vector<ov::genai::VideoMetadata> videos_metadata,
    ov::AnyMap properties
) : AsyncWorker(callback),
    pipe(pipe),
    text(std::move(text)),
    images(std::move(images)),
    videos(std::move(videos)),
    videos_metadata(std::move(videos_metadata)),
    properties(std::move(properties)) {};

void EmbeddingPipelineEmbedWorker::Execute() {
    this->embed_result = this->pipe->embed(this->text, this->images, this->videos, this->videos_metadata, this->properties);
};

void EmbeddingPipelineEmbedWorker::OnOK() {
    Callback().Call({
        Env().Null(),                                                              // Error result
        cpp_to_js<ov::genai::EmbedResult, Napi::Value>(Env(), this->embed_result)  // Ok result
    });
};
