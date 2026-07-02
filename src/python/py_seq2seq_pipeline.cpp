// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/seq2seq_pipeline.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/operators.h"

namespace py = pybind11;
using namespace ov::genai;

void init_seq2seq_pipeline(py::module_& m) {
    py::class_<Seq2SeqDecodedResults> seq2seq_decoded_results(m, "Seq2SeqDecodedResults", py::module_local());
    seq2seq_decoded_results
        .def(py::init<>())
        .def_readwrite("token_ids", &Seq2SeqDecodedResults::token_ids)
        .def_readwrite("texts", &Seq2SeqDecodedResults::texts)
        .def_readwrite("scores", &Seq2SeqDecodedResults::scores)
        .def_readwrite("finish_reasons", &Seq2SeqDecodedResults::finish_reasons)
        .def_readwrite("perf_metrics", &Seq2SeqDecodedResults::perf_metrics);

    py::class_<Seq2SeqPipeline> seq2seq_pipeline(m, "Seq2SeqPipeline", py::module_local());
    seq2seq_pipeline
        .def(
            py::init<const std::filesystem::path&, const std::string&, const ov::AnyMap&>(),
            py::arg("models_path"),
            py::arg("device") = "CPU",
            py::arg("properties") = ov::AnyMap()
        )
        .def(
            "generate",
            py::overload_cast<const std::string&, const ov::AnyMap&>(&Seq2SeqPipeline::generate),
            py::arg("input_text"),
            py::arg("properties") = ov::AnyMap()
        )
        .def(
            "generate",
            py::overload_cast<const std::vector<std::string>&, const ov::AnyMap&>(&Seq2SeqPipeline::generate),
            py::arg("input_texts"),
            py::arg("properties") = ov::AnyMap()
        )
        .def("get_generation_config", &Seq2SeqPipeline::get_generation_config)
        .def("set_generation_config", &Seq2SeqPipeline::set_generation_config)
        .def("get_encoder_output_name", &Seq2SeqPipeline::get_encoder_output_name)
        .def("get_decoder_io_names", &Seq2SeqPipeline::get_decoder_io_names);
}
