#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "text_callback_streamer.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace py = pybind11;
using namespace ov::genai;

PYBIND11_MODULE(ov_genai_test_utils, m) {
    py::class_<TextCallbackStreamer>(m, "TextCallbackStreamer")
        .def(py::init<const Tokenizer&, std::function<CallbackTypeVariant(std::string)>>(), py::arg("tokenizer"), py::arg("callback"))
        .def("write", &TextCallbackStreamer::write)
        .def("end", &TextCallbackStreamer::end);
}
