// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/embedding_pipeline/init_worker.hpp"
#include "include/helper.hpp"

EmbeddingPipelineInitWorker::EmbeddingPipelineInitWorker(
    Function& callback,
    std::shared_ptr<ov::genai::EmbeddingPipeline>& pipe,
    std::filesystem::path model_path,
    std::string device,
    Object properties
) : AsyncWorker(callback),
    pipe(pipe),
    model_path(std::move(model_path)),
    device(std::move(device)),
    properties(js_to_cpp<ov::AnyMap>(Env(), properties)) {};

void EmbeddingPipelineInitWorker::Execute() {
    try {
        this->pipe = std::make_shared<ov::genai::EmbeddingPipeline>(this->model_path, this->device, this->properties);
    } catch(const std::exception& ex) {
        SetError(ex.what());
    }
};

void EmbeddingPipelineInitWorker::OnOK() {
    Callback().Call({
        Env().Null()      // Error result
    });
};
