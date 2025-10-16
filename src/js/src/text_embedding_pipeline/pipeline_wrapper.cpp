#include "include/helper.hpp"
#include "include/text_embedding_pipeline/pipeline_wrapper.hpp"
#include "include/text_embedding_pipeline/init_worker.hpp"
#include "include/text_embedding_pipeline/embed_query_worker.hpp"
#include "include/text_embedding_pipeline/embed_documents_worker.hpp"

TextEmbeddingPipelineWrapper::TextEmbeddingPipelineWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<TextEmbeddingPipelineWrapper>(info) {};

Napi::Function TextEmbeddingPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(
        env,
        "TextEmbeddingPipeline",
        {
            InstanceMethod("init", &TextEmbeddingPipelineWrapper::init),
            InstanceMethod("embedQuerySync", &TextEmbeddingPipelineWrapper::embed_query),
            InstanceMethod("embedDocumentsSync", &TextEmbeddingPipelineWrapper::embed_documents),
            InstanceMethod("embedQuery", &TextEmbeddingPipelineWrapper::embed_query_async),
            InstanceMethod("embedDocuments", &TextEmbeddingPipelineWrapper::embed_documents_async),
        }
    );
}

Napi::Value TextEmbeddingPipelineWrapper::init(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const std::string model_path = info[0].ToString();
    const std::string device = info[1].ToString();
    const Napi::Object config = info[2].As<Napi::Object>();
    const Napi::Object properties = info[3].As<Napi::Object>();
    Napi::Function callback = info[4].As<Napi::Function>();

    EmbeddingInitWorker* asyncWorker = new EmbeddingInitWorker(callback, this->pipe, model_path, device, config, properties);
    asyncWorker->Queue();

    return info.Env().Undefined();
}

Napi::Value TextEmbeddingPipelineWrapper::embed_query(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto text = info[0].ToString();

    auto embed_result = this->pipe->embed_query(text);
    return cpp_to_js<ov::genai::EmbeddingResult, Napi::Value>(env, embed_result);
}

Napi::Value TextEmbeddingPipelineWrapper::embed_documents(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto documents = js_to_cpp<std::vector<std::string>>(env, info[0]);

    auto embed_results = this->pipe->embed_documents(documents);

    return cpp_to_js<ov::genai::EmbeddingResults, Napi::Value>(env, embed_results);
}

Napi::Value TextEmbeddingPipelineWrapper::embed_query_async(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::String text = info[0].As<Napi::String>();
    Napi::Function callback = info[1].As<Napi::Function>();

    auto asyncWorker = new EmbedQueryWorker(callback, this->pipe, text);
    asyncWorker->Queue();

    return info.Env().Undefined();
}

Napi::Value TextEmbeddingPipelineWrapper::embed_documents_async(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Array document = info[0].As<Napi::Array>();
    Napi::Function callback = info[1].As<Napi::Function>();

    auto asyncWorker = new EmbedDocumentsWorker(callback, this->pipe, document);
    asyncWorker->Queue();

    return info.Env().Undefined();
}
