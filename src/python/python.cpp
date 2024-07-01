// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include "continuous_batching_pipeline.hpp"
#include "../cpp/src/tokenizers_path.hpp"

namespace py = pybind11;

std::ostream& operator << (std::ostream& stream, const GenerationResult& generation_result) {
    stream << generation_result.m_request_id << std::endl;
    const bool has_scores = !generation_result.m_scores.empty();
    for (size_t i = 0; i < generation_result.m_generation_ids.size(); ++i) {
        stream << "{ ";
        if (has_scores)
            stream << generation_result.m_scores[i] << ", ";
        stream << generation_result.m_generation_ids[i] << " }" << std::endl;
    }
    return stream << std::endl;
}

std::string ov_tokenizers_module_path() {
    // Try a path relative to build artifacts folder first.
    std::filesystem::path from_relative = tokenizers_relative_to_genai();
    if (std::filesystem::exists(from_relative)) {
        return from_relative.string();
    }
    return py::str(py::module_::import("openvino_tokenizers").attr("_ext_path"));
}

PYBIND11_MODULE(py_continuous_batching, m) {
    py::class_<GenerationResult>(m, "GenerationResult")
        .def(py::init<>())
        .def_readonly("m_request_id", &GenerationResult::m_request_id)
        .def_property("m_generation_ids",
            [](GenerationResult &r) -> py::list {
                py::list res;
                for (auto s: r.m_generation_ids) {
                
                    PyObject* py_s = PyUnicode_DecodeUTF8(s.data(), s.length(), "replace");
                    res.append(py_s);
                }
                return res;
            },
            [](GenerationResult &r, std::vector<std::string> &generation_ids) {
                r.m_generation_ids = generation_ids;
            })
        .def_readwrite("m_scores", &GenerationResult::m_scores)
        .def("__repr__",
            [](const GenerationResult &r) -> py::str{
                std::stringstream stream;
                stream << "<py_continuous_batching.GenerationResult " << r << ">";
                std::string str = stream.str();
                PyObject* py_s = PyUnicode_DecodeUTF8(str.data(), str.length(), "replace");
                return py::reinterpret_steal<py::str>(py_s);
            }
        )
        .def("get_generation_ids", 
        [](GenerationResult &r) -> py::list {
            py::list res;
            for (auto s: r.m_generation_ids) {
                PyObject* py_s = PyUnicode_DecodeUTF8(s.data(), s.length(), "replace");
                res.append(py_s);
            }
            return res;
        });

    py::class_<SchedulerConfig>(m, "SchedulerConfig")
        .def(py::init<>())
        .def_readwrite("max_num_batched_tokens", &SchedulerConfig::max_num_batched_tokens)
        .def_readwrite("num_kv_blocks", &SchedulerConfig::num_kv_blocks)
        .def_readwrite("cache_size", &SchedulerConfig::cache_size)
        .def_readwrite("block_size", &SchedulerConfig::block_size)
        .def_readwrite("cache_size", &SchedulerConfig::cache_size)
        .def_readwrite("dynamic_split_fuse", &SchedulerConfig::dynamic_split_fuse)
        .def_readwrite("max_num_seqs", &SchedulerConfig::max_num_seqs);

    py::class_<ContinuousBatchingPipeline>(m, "ContinuousBatchingPipeline")
        .def(py::init([](const std::string& model_path, const SchedulerConfig& config) {
            ScopedVar env_manager(ov_tokenizers_module_path());
            return std::make_unique<ContinuousBatchingPipeline>(model_path, config);
        }))
        .def("get_tokenizer", &ContinuousBatchingPipeline::get_tokenizer)
        .def("get_config", &ContinuousBatchingPipeline::get_config)
        .def("add_request", &ContinuousBatchingPipeline::add_request)
        .def("step", &ContinuousBatchingPipeline::step)
        .def("has_non_finished_requests", &ContinuousBatchingPipeline::has_non_finished_requests)
        .def("generate", &ContinuousBatchingPipeline::generate);
}
