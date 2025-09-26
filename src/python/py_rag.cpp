// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/stl_bind.h>

#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "openvino/genai/rag/text_rerank_pipeline.hpp"
#include "py_utils.hpp"
#include "tokenizer/tokenizers_path.hpp"

namespace py = pybind11;
using ov::genai::EmbeddingResult;
using ov::genai::EmbeddingResults;
using ov::genai::TextEmbeddingPipeline;
using ov::genai::TextRerankPipeline;

namespace pyutils = ov::genai::pybind::utils;

namespace {
const auto text_embedding_config_docstring = R"(
Structure to keep TextEmbeddingPipeline configuration parameters.

Attributes:
    max_length (int, optional):
        Maximum length of tokens passed to the embedding model.
    pad_to_max_length (bool, optional):
        If 'True', model input tensors are padded to the maximum length.
    batch_size (int, optional):
        Batch size for the embedding model.
        Useful for database population. If set, the pipeline will fix model shape for inference optimization.
        Number of documents passed to pipeline should be equal to batch_size.
        For query embeddings, batch_size should be set to 1 or not set.
    pooling_type (TextEmbeddingPipeline.PoolingType, optional):
        Pooling strategy applied to the model output tensor. Defaults to PoolingType.CLS.
    normalize (bool, optional):
        If True, L2 normalization is applied to embeddings. Defaults to True.
    query_instruction (str, optional):
        Instruction to use for embedding a query.
    embed_instruction (str, optional):
        Instruction to use for embedding a document.
    padding_side (str, optional):
        Side to use for padding "left" or "right"
)";

const auto text_reranking_config_docstring = R"(
Structure to keep TextRerankPipeline configuration parameters.
Attributes:
    top_n (int, optional):
        Number of documents to return sorted by score.
    max_length (int, optional):
        Maximum length of tokens passed to the embedding model.
)";

}  // namespace

void init_rag_pipelines(py::module_& m) {
    auto text_embedding_pipeline =
        py::class_<TextEmbeddingPipeline>(m, "TextEmbeddingPipeline", "Text embedding pipeline")
            .def(
                "embed_documents",
                [](TextEmbeddingPipeline& pipe,
                   std::vector<std::string>& texts) -> py::typing::Union<EmbeddingResults> {
                    EmbeddingResults res;
                    {
                        py::gil_scoped_release rel;
                        res = pipe.embed_documents(texts);
                    }
                    return py::cast(res);
                },
                py::arg("texts"),
                "List of texts ",
                "Computes embeddings for a vector of texts")
            .def(
                "start_embed_documents_async",
                [](TextEmbeddingPipeline& pipe, std::vector<std::string>& texts) -> void {
                    py::gil_scoped_release rel;
                    pipe.start_embed_documents_async(texts);
                },
                py::arg("texts"),
                "List of texts ",
                "Asynchronously computes embeddings for a vector of texts")
            .def(
                "wait_embed_documents",
                [](TextEmbeddingPipeline& pipe) -> py::typing::Union<EmbeddingResults> {
                    EmbeddingResults res;
                    {
                        py::gil_scoped_release rel;
                        res = pipe.wait_embed_documents();
                    }
                    return py::cast(res);
                },
                "Waits computed embeddings of a vector of texts")
            .def(
                "embed_query",
                [](TextEmbeddingPipeline& pipe, std::string& text) -> py::typing::Union<EmbeddingResult> {
                    EmbeddingResult res;
                    {
                        py::gil_scoped_release rel;
                        res = pipe.embed_query(text);
                    }
                    return py::cast(res);
                },
                py::arg("text"),
                "text ",
                "Computes embeddings for a query")
            .def(
                "start_embed_query_async",
                [](TextEmbeddingPipeline& pipe, std::string& text) -> void {
                    py::gil_scoped_release rel;
                    pipe.start_embed_query_async(text);
                },
                py::arg("text"),
                "text ",
                "Asynchronously computes embeddings for a query")
            .def(
                "wait_embed_query",
                [](TextEmbeddingPipeline& pipe) -> py::typing::Union<EmbeddingResult> {
                    EmbeddingResult res;
                    {
                        py::gil_scoped_release rel;
                        res = pipe.wait_embed_query();
                    }
                    return py::cast(res);
                },
                "Waits computed embeddings for a query");

    py::enum_<TextEmbeddingPipeline::PoolingType>(text_embedding_pipeline, "PoolingType")
        .value("CLS", TextEmbeddingPipeline::PoolingType::CLS, "First token embeddings")
        .value("MEAN", TextEmbeddingPipeline::PoolingType::MEAN, "The average of all token embeddings")
        .value("LAST_TOKEN", TextEmbeddingPipeline::PoolingType::LAST_TOKEN, "Last token embeddings");

    py::class_<TextEmbeddingPipeline::Config>(text_embedding_pipeline, "Config", text_embedding_config_docstring)
        .def(py::init<>())
        .def(py::init([](py::kwargs kwargs) {
            return TextEmbeddingPipeline::Config(pyutils::kwargs_to_any_map(kwargs));
        }))
        .def("validate",
             &TextEmbeddingPipeline::Config::validate,
             "Checks that are no conflicting parameters. Raises exception if config is invalid.")
        .def_readwrite("max_length", &TextEmbeddingPipeline::Config::max_length)
        .def_readwrite("pad_to_max_length", &TextEmbeddingPipeline::Config::pad_to_max_length)
        .def_readwrite("batch_size", &TextEmbeddingPipeline::Config::batch_size)
        .def_readwrite("pooling_type", &TextEmbeddingPipeline::Config::pooling_type)
        .def_readwrite("normalize", &TextEmbeddingPipeline::Config::normalize)
        .def_readwrite("query_instruction", &TextEmbeddingPipeline::Config::query_instruction)
        .def_readwrite("embed_instruction", &TextEmbeddingPipeline::Config::embed_instruction)
        .def_readwrite("padding_side", &TextEmbeddingPipeline::Config::padding_side);

    text_embedding_pipeline.def(
        py::init([](const std::filesystem::path& models_path,
                    const std::string& device,
                    const std::optional<TextEmbeddingPipeline::Config>& config,
                    const py::kwargs& kwargs) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());

            if (config.has_value()) {
                return std::make_unique<TextEmbeddingPipeline>(models_path,
                                                               device,
                                                               *config,
                                                               pyutils::kwargs_to_any_map(kwargs));
            }
            return std::make_unique<TextEmbeddingPipeline>(models_path, device, pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("models_path"),
        "Path to the directory containing model xml/bin files and tokenizer",
        py::arg("device"),
        "Device to run the model on (e.g., CPU, GPU)",
        py::arg("config") = std::nullopt,
        "Optional pipeline configuration",
        "Plugin and/or config properties",
        R"(
Constructs a pipeline from xml/bin files, tokenizer and configuration in the same dir
models_path (os.PathLike): Path to the directory containing model xml/bin files and tokenizer
device (str): Device to run the model on (e.g., CPU, GPU).
config: (TextEmbeddingPipeline.Config): Optional pipeline configuration
kwargs: Plugin and/or config properties
)");

    auto text_rerank_pipeline =
        py::class_<ov::genai::TextRerankPipeline>(m, "TextRerankPipeline", "Text rerank pipeline")
            .def(
                "rerank",
                [](ov::genai::TextRerankPipeline& pipe,
                   const std::string& query,
                   const std::vector<std::string>& texts) -> py::typing::Union<std::vector<std::pair<size_t, float>>> {
                    std::vector<std::pair<size_t, float>> res;
                    {
                        py::gil_scoped_release rel;
                        res = pipe.rerank(query, texts);
                    }
                    return py::cast(res);
                },
                py::arg("query"),
                py::arg("texts"),
                "Reranks a vector of texts based on the query.")
            .def(
                "start_rerank_async",
                [](ov::genai::TextRerankPipeline& pipe,
                   const std::string& query,
                   const std::vector<std::string>& texts) -> void {
                    py::gil_scoped_release rel;
                    pipe.start_rerank_async(query, texts);
                },
                py::arg("query"),
                py::arg("texts"),
                "Asynchronously reranks a vector of texts based on the query.")
            .def(
                "wait_rerank",
                [](ov::genai::TextRerankPipeline& pipe) -> py::typing::Union<std::vector<std::pair<size_t, float>>> {
                    std::vector<std::pair<size_t, float>> res;
                    {
                        py::gil_scoped_release rel;
                        res = pipe.wait_rerank();
                    }
                    return py::cast(res);
                },
                "Waits for reranked texts.");

    py::class_<ov::genai::TextRerankPipeline::Config>(text_rerank_pipeline, "Config", text_reranking_config_docstring)
        .def(py::init<>())
        .def(py::init([](py::kwargs kwargs) {
            return ov::genai::TextRerankPipeline::Config(pyutils::kwargs_to_any_map(kwargs));
        }))
        .def_readwrite("top_n", &ov::genai::TextRerankPipeline::Config::top_n)
        .def_readwrite("max_length", &ov::genai::TextRerankPipeline::Config::max_length);

    text_rerank_pipeline.def(
        py::init([](const std::filesystem::path& models_path,
                    const std::string& device,
                    const std::optional<ov::genai::TextRerankPipeline::Config>& config,
                    const py::kwargs& kwargs) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            if (config.has_value()) {
                return std::make_unique<ov::genai::TextRerankPipeline>(models_path,
                                                                       device,
                                                                       *config,
                                                                       pyutils::kwargs_to_any_map(kwargs));
            }
            return std::make_unique<ov::genai::TextRerankPipeline>(models_path,
                                                                   device,
                                                                   pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("models_path"),
        "Path to the directory containing model xml/bin files and tokenizer",
        py::arg("device"),
        "Device to run the model on (e.g., CPU, GPU)",
        py::arg("config") = std::nullopt,
        "Optional pipeline configuration",
        "Plugin and/or config properties",
        R"(
Constructs a pipeline from xml/bin files, tokenizer and configuration in the same dir
models_path (os.PathLike): Path to the directory containing model xml/bin files and tokenizer
device (str): Device to run the model on (e.g., CPU, GPU).
config: (TextRerankPipeline.Config): Optional pipeline configuration
kwargs: Plugin and/or config properties
)");
}
