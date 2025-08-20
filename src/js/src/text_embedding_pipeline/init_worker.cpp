#include "include/text_embedding_pipeline/init_worker.hpp"
#include "include/helper.hpp"
#include <chrono>
#include <thread>

EmbeddingInitWorker::EmbeddingInitWorker(
    Function& callback,
    std::shared_ptr<ov::genai::TextEmbeddingPipeline>& pipe,
    const std::string model_path,
    const std::string device,
    Object config,
    Object properties
) : AsyncWorker(callback),
    pipe(pipe),
    model_path(model_path),
    device(device),
    config(js_to_cpp<ov::AnyMap>(Env(), config)),
    properties(js_to_cpp<ov::AnyMap>(Env(), properties)) {};

void EmbeddingInitWorker::Execute() {
    try {
        ov::genai::TextEmbeddingPipeline::Config config(this->config);
        this->pipe = std::make_shared<ov::genai::TextEmbeddingPipeline>(this->model_path, this->device, config, this->properties);
    } catch(const std::exception& ex) {
        SetError(ex.what());
    }
};

void EmbeddingInitWorker::OnOK() {
    Callback().Call({
        Env().Null()      // Error result
    });
};
