#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "openvino/genai/text_streamer.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace py = pybind11;
using namespace ov::genai;

PYBIND11_MODULE(genai_test_utils, m) {
    py::class_<TextStreamer>(m, "TextCallbackStreamer")
        .def(py::init<const Tokenizer&, std::function<CallbackTypeVariant(std::string)>>(), py::arg("tokenizer"), py::arg("callback"))
        .def("write", &TextStreamer::write)
        .def("end", &TextStreamer::end);
}
