#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "llm_pipeline.hpp"

namespace py = pybind11;
using namespace ov;

PYBIND11_MODULE(py_generate_pipeline, m) {
    m.doc() = "Pybind11 binding for LLM Pipeline";

    py::class_<LLMPipeline>(m, "LLMPipeline")
        .def(py::init<std::string&, std::string&, std::string&, std::string, const ov::AnyMap&>(),
             py::arg("model_path"), py::arg("tokenizer_path"), py::arg("detokenizer_path"),
             py::arg("device") = "CPU", py::arg("plugin_config") = ov::AnyMap{})
        .def(py::init<std::string&, std::string, const ov::AnyMap&>(),
             py::arg("path"), py::arg("device") = "CPU", py::arg("plugin_config") = ov::AnyMap{})
        .def("__call__", (std::string (LLMPipeline::*)(std::string)) &LLMPipeline::operator())
        .def("__call__", (std::string (LLMPipeline::*)(std::string, GenerationConfig)) &LLMPipeline::operator())
        .def("generate", (EncodedResults (LLMPipeline::*)(ov::Tensor, ov::Tensor, GenerationConfig)) &LLMPipeline::generate)
        .def("generate", (EncodedResults (LLMPipeline::*)(ov::Tensor, ov::Tensor)) &LLMPipeline::generate)
        // Bind other methods similarly
        .def("get_tokenizer", &LLMPipeline::get_tokenizer)
        .def("apply_chat_template", &LLMPipeline::apply_chat_template);


}
