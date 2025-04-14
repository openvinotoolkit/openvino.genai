// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/stl_bind.h>

#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "py_utils.hpp"
#include "tokenizers_path.hpp"

namespace py = pybind11;
using ov::genai::EmbeddingResult;
using ov::genai::TextEmbeddingPipeline;

namespace pyutils = ov::genai::pybind::utils;

void init_rag_pipelines(py::module_& m) {
    m.doc() = "Pybind11 binding for RAG Pipelines";

    auto text_embedding_pipeline =
        py::class_<TextEmbeddingPipeline>(m, "TextEmbeddingPipeline", "Text embedding pipeline")
            .def(py::init(
                     [](const std::filesystem::path& models_path, const std::string& device, const py::kwargs& kwargs) {
                         ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
                         return std::make_unique<TextEmbeddingPipeline>(models_path,
                                                                        device,
                                                                        pyutils::kwargs_to_any_map(kwargs));
                     }),
                 py::arg("models_path"),
                 "folder with openvino_model.xml and openvino_tokenizer[detokenizer].xml files",
                 py::arg("device"),
                 "device on which inference will be done",
                 "openvino.properties map",
                 R"(
        TextEmbeddingPipeline class constructor.
        models_path (os.PathLike): Path to the model file.
        device (str): Device to run the model on (e.g., CPU, GPU).
        kwargs: Config or device properties.
    )")
            .def(
                "embed_documents",
                [](TextEmbeddingPipeline& pipe,
                   std::vector<std::string>& texts) -> py::typing::Union<std::vector<EmbeddingResult>> {
                    std::vector<EmbeddingResult> res;
                    {
                        py::gil_scoped_release rel;
                        res = pipe.embed_documents(texts);
                    }
                    return py::cast(res);
                },
                py::arg("texts"),
                "List of texts ",
                "Computes embeddings for a vector of texts");

    py::enum_<TextEmbeddingPipeline::PoolingType>(text_embedding_pipeline, "PoolingType")
        .value("CLS", TextEmbeddingPipeline::PoolingType::CLS)
        .value("MEAN", TextEmbeddingPipeline::PoolingType::MEAN);

    py::class_<TextEmbeddingPipeline::Config>(text_embedding_pipeline,
                                              "Config",
                                              "This class is used for storing TextEmbeddingPipeline config.")
        .def_readwrite("max_length", &TextEmbeddingPipeline::Config::max_length)
        .def_readwrite("pooling_type", &TextEmbeddingPipeline::Config::pooling_type)
        .def_readwrite("normalize", &TextEmbeddingPipeline::Config::normalize)
        .def_readwrite("query_instruction", &TextEmbeddingPipeline::Config::query_instruction)
        .def_readwrite("embed_instruction", &TextEmbeddingPipeline::Config::embed_instruction);
}
