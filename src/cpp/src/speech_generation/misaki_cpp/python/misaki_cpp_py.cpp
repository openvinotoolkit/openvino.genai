#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "misaki/g2p.hpp"

namespace py = pybind11;

namespace {

py::str to_py_str_lossy(const std::string& value) {
    PyObject* decoded = PyUnicode_DecodeUTF8(value.data(), static_cast<Py_ssize_t>(value.size()), "replace");
    if (decoded != nullptr) {
        return py::reinterpret_steal<py::str>(decoded);
    }
    PyErr_Clear();
    return py::str(value);
}

py::object to_py_optional_str(const std::optional<std::string>& value) {
    if (!value.has_value()) {
        return py::none();
    }
    return to_py_str_lossy(*value);
}

py::dict to_py_token(const misaki::MToken& token) {
    py::dict out;
    out["text"] = to_py_str_lossy(token.text);
    out["tag"] = to_py_str_lossy(token.tag);
    out["whitespace"] = to_py_str_lossy(token.whitespace);
    out["phonemes"] = to_py_optional_str(token.phonemes);
    out["start_ts"] = token.start_ts.has_value() ? py::cast(*token.start_ts) : py::none();
    out["end_ts"] = token.end_ts.has_value() ? py::cast(*token.end_ts) : py::none();

    if (!token._.has_value()) {
        out["_"] = py::none();
        return out;
    }

    py::dict meta;
    const auto& u = *token._;
    meta["is_head"] = u.is_head.has_value() ? py::cast(*u.is_head) : py::none();
    meta["alias"] = u.alias.has_value() ? py::cast(*u.alias) : py::none();
    meta["stress"] = u.stress.has_value() ? py::cast(*u.stress) : py::none();
    meta["currency"] = u.currency.has_value() ? py::cast(*u.currency) : py::none();
    meta["num_flags"] = u.num_flags;
    meta["prespace"] = u.prespace.has_value() ? py::cast(*u.prespace) : py::none();
    meta["rating"] = u.rating.has_value() ? py::cast(*u.rating) : py::none();
    out["_"] = std::move(meta);
    return out;
}

class PyG2PEngine {
public:
    PyG2PEngine(std::string lang, std::string variant)
        : m_engine(misaki::make_engine(lang, variant)) {}

    py::str phonemize(const std::string& text) const {
        return to_py_str_lossy(m_engine->phonemize(text));
    }

    py::dict phonemize_with_tokens(const std::string& text) const {
        const auto result = m_engine->phonemize_with_tokens(text);
        py::list tokens;
        for (const auto& token : result.tokens) {
            tokens.append(to_py_token(token));
        }

        py::dict out;
        out["phonemes"] = to_py_str_lossy(result.phonemes);
        out["tokens"] = std::move(tokens);
        return out;
    }

    void set_unknown_token(const std::string& unknown_token) {
        m_engine->set_unknown_token(unknown_token);
    }

    void set_lexicon_data_root(const std::string& path) {
        misaki::set_english_lexicon_data_root(path);
    }

    void clear_lexicon_data_root() {
        misaki::clear_english_lexicon_data_root();
    }

private:
    std::unique_ptr<misaki::G2P> m_engine;
};

}  // namespace

PYBIND11_MODULE(misaki_cpp_py, m) {
    m.doc() = "Standalone Python bindings for embedded misaki_cpp";

    py::class_<PyG2PEngine>(m, "Engine")
        .def(py::init<std::string, std::string>(), py::arg("lang") = "en", py::arg("variant") = "en-us")
        .def("phonemize", &PyG2PEngine::phonemize, py::arg("text"))
        .def("phonemize_with_tokens", &PyG2PEngine::phonemize_with_tokens, py::arg("text"))
        .def("set_unknown_token", &PyG2PEngine::set_unknown_token, py::arg("unknown_token"))
        .def("set_lexicon_data_root", &PyG2PEngine::set_lexicon_data_root, py::arg("path"))
        .def("clear_lexicon_data_root", &PyG2PEngine::clear_lexicon_data_root);
}
