#include "include/helper.hpp"
#include "include/text_embedding_pipeline/embed_query_worker.hpp"

EmbedQueryWorker::EmbedQueryWorker(
    Function& callback,
    std::shared_ptr<ov::genai::TextEmbeddingPipeline>& pipe,
    String text
) : AsyncWorker(callback), pipe(pipe), text(text.ToString()) {};

void EmbedQueryWorker::Execute() {
    try {
        this->embed_result = this->pipe->embed_query(this->text);
    } catch(const std::exception& ex) {
        SetError(ex.what());
    }
};

void EmbedQueryWorker::OnOK() {
    Callback().Call({ 
        Env().Null(),                                                                   // Error result
        cpp_to_js<ov::genai::EmbeddingResult, Napi::Value>(Env(), this->embed_result)   // Ok result
    });
};