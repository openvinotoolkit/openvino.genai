#include "include/helper.hpp"
#include "include/text_embedding_pipeline/embed_documents_worker.hpp"

EmbedDocumentsWorker::EmbedDocumentsWorker(
    Function& callback,
    std::shared_ptr<ov::genai::TextEmbeddingPipeline>& pipe,
    Array documents
) : AsyncWorker(callback), pipe(pipe), documents(js_to_cpp<std::vector<std::string>>(Env(), documents)) {};

void EmbedDocumentsWorker::Execute() {
    try {
        this->embed_results = this->pipe->embed_documents(this->documents);
    } catch(const std::exception& ex) {
        SetError(ex.what());
    }
};

void EmbedDocumentsWorker::OnOK() {
    Callback().Call({
        Env().Null(),                                                                   // Error result
        cpp_to_js<ov::genai::EmbeddingResults, Napi::Value>(Env(), this->embed_results) // Ok result
    });
};
