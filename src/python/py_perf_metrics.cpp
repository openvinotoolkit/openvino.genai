// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"
#include "py_utils.hpp"
#include "bindings_utils.hpp"

namespace py = pybind11;

using ov::genai::SummaryStats;
using ov::genai::MeanStdPair;
using ov::genai::PerfMetrics;
using ov::genai::RawPerfMetrics;
using ov::genai::ExtendedPerfMetrics;
using ov::genai::SDPerfMetrics;
using ov::genai::SDPerModelsPerfMetrics;

namespace pyutils = ov::genai::pybind::utils;
namespace common_utils = ov::genai::common_bindings::utils;

namespace {

auto raw_perf_metrics_docstring = R"(
    Structure with raw performance metrics for each generation before any statistics are calculated.

    :param generate_durations: Durations for each generate call in milliseconds.
    :type generate_durations: list[float]

    :param tokenization_durations: Durations for the tokenization process in milliseconds.
    :type tokenization_durations: list[float]

    :param detokenization_durations: Durations for the detokenization process in milliseconds.
    :type detokenization_durations: list[float]

    :param m_times_to_first_token: Times to the first token for each call in milliseconds.
    :type m_times_to_first_token: list[float]

    :param m_new_token_times: Timestamps of generation every token or batch of tokens in milliseconds.
    :type m_new_token_times: list[double]

    :param token_infer_durations : Inference time for each token in milliseconds.
    :type batch_sizes: list[float]

    :param m_batch_sizes: Batch sizes for each generate call.
    :type m_batch_sizes: list[int]

    :param m_durations: Total durations for each generate call in milliseconds.
    :type m_durations: list[float]

    :param inference_durations : Total inference duration for each generate call in milliseconds.
    :type batch_sizes: list[float]

    :param grammar_compile_times: Time to compile the grammar in milliseconds.
    :type grammar_compile_times: list[float]
)";

auto perf_metrics_docstring = R"(
    Holds performance metrics for each generate call.

    PerfMetrics holds the following metrics with mean and standard deviations:
    - Time To the First Token (TTFT), ms
    - Time per Output Token (TPOT), ms/token
    - Inference time per Output Token (IPOT), ms/token
    - Generate total duration, ms
    - Inference duration, ms
    - Tokenization duration, ms
    - Detokenization duration, ms
    - Throughput, tokens/s

    Additional metrics include:
    - Load time, ms
    - Number of generated tokens
    - Number of tokens in the input prompt
    - Time to initialize grammar compiler for each backend, ms
    - Time to compile grammar, ms

    Preferable way to access metrics is via getter methods. Getter methods calculate mean and std values from raw_metrics and return pairs.
    If mean and std were already calculated, getters return cached values.

    :param get_load_time: Returns the load time in milliseconds.
    :type get_load_time: float

    :param get_num_generated_tokens: Returns the number of generated tokens.
    :type get_num_generated_tokens: int

    :param get_num_input_tokens: Returns the number of tokens in the input prompt.
    :type get_num_input_tokens: int

    :param get_ttft: Returns the mean and standard deviation of TTFT in milliseconds.
    :type get_ttft: MeanStdPair

    :param get_tpot: Returns the mean and standard deviation of TPOT in milliseconds.
    :type get_tpot: MeanStdPair

    :param get_ipot: Returns the mean and standard deviation of IPOT in milliseconds.
    :type get_ipot: MeanStdPair

    :param get_throughput: Returns the mean and standard deviation of throughput in tokens per second.
    :type get_throughput: MeanStdPair

    :param get_inference_duration: Returns the mean and standard deviation of the time spent on model inference during generate call in milliseconds.
    :type get_inference_duration: MeanStdPair

    :param get_generate_duration: Returns the mean and standard deviation of generate durations in milliseconds.
    :type get_generate_duration: MeanStdPair

    :param get_tokenization_duration: Returns the mean and standard deviation of tokenization durations in milliseconds.
    :type get_tokenization_duration: MeanStdPair

    :param get_detokenization_duration: Returns the mean and standard deviation of detokenization durations in milliseconds.
    :type get_detokenization_duration: MeanStdPair

    :param get_grammar_compiler_init_times: Returns a map with the time to initialize the grammar compiler for each backend in milliseconds.
    :type get_grammar_compiler_init_times: dict[str, float]

    :param get_grammar_compile_time: Returns the mean, standard deviation, min, and max of grammar compile times in milliseconds.
    :type get_grammar_compile_time: SummaryStats

    :param raw_metrics: A structure of RawPerfMetrics type that holds raw metrics.
    :type raw_metrics: RawPerfMetrics
)";

auto sd_perf_metrics_docstring = R"(
    Holds performance metrics for draft and main models of SpeculativeDecoding Pipeline.

    SDPerfMetrics holds fields with mean and standard deviations for the all PerfMetrics fields and following metrics:
    - Time to the Second Token (TTFT), ms
    - avg_latency, ms/inference

    Preferable way to access values is via get functions. Getters calculate mean and std values from raw_metrics and return pairs.
    If mean and std were already calculated, getters return cached values.

    :param get_ttst: Returns the mean and standard deviation of TTST in milliseconds.
                     The second token is presented separately as for some plugins this can be expected to take longer than next tokens.
                     In case of GPU plugin: Async compilation of some opt kernels can be completed after second token.
                     Also, additional memory manipulation can happen at second token time.
    :type get_ttst: MeanStdPair

    :param get_latency: Returns the mean and standard deviation of the latency from the third token in milliseconds per inference,
                        which includes also prev and post processing. First and second token time is presented separately as ttft and ttst.
    :type get_latency: MeanStdPair

    Additional points:
      - TPOT is calculated from the third token. The reasons for this, please, see in the description for avg_latency.
      - `total number of iterations` of the model can be taken from raw performance metrics raw_metrics.m_durations.size().
)";

auto sd_per_models_perf_metrics_docstring = R"(
    Holds performance metrics for each generate call.

    :param main_model_metrics: performance metrics for main model
    :type main_model_metrics: SDPerfMetrics

    :param draft_model_metrics: performance metrics for draft model
    :type draft_model_metrics: SDPerfMetrics

    :param get_num_accepted_tokens: total number of tokens, which was generated by draft model and accepted by main model
    :type get_num_accepted_tokens: int
)";

} // namespace

void init_perf_metrics(py::module_& m) {
    py::class_<RawPerfMetrics>(m, "RawPerfMetrics", raw_perf_metrics_docstring)
        .def(py::init<>())
        .def_property_readonly("generate_durations", [](const RawPerfMetrics &rw) {
            return common_utils::get_ms(rw, &RawPerfMetrics::generate_durations);
        })
        .def_property_readonly("tokenization_durations", [](const RawPerfMetrics &rw) { 
            return common_utils::get_ms(rw, &RawPerfMetrics::tokenization_durations);
        })
        .def_property_readonly("detokenization_durations", [](const RawPerfMetrics &rw) { 
            return common_utils::get_ms(rw, &RawPerfMetrics::detokenization_durations); 
        })
        .def_property_readonly("m_times_to_first_token", [](const RawPerfMetrics &rw) {
            return common_utils::get_ms(rw, &RawPerfMetrics::m_times_to_first_token);
        })
        .def_property_readonly("m_new_token_times", [](const RawPerfMetrics &rw) {
            return common_utils::timestamp_to_ms(rw, &RawPerfMetrics::m_new_token_times);
        })
        .def_property_readonly("token_infer_durations", [](const RawPerfMetrics &rw) {
            return common_utils::get_ms(rw, &RawPerfMetrics::m_token_infer_durations);
        })
        .def_readonly("m_batch_sizes", &RawPerfMetrics::m_batch_sizes)
        .def_property_readonly("m_durations", [](const RawPerfMetrics &rw) {
            return common_utils::get_ms(rw, &RawPerfMetrics::m_durations);
        })
        .def_property_readonly("inference_durations", [](const RawPerfMetrics &rw) {
            return common_utils::get_ms(rw, &RawPerfMetrics::m_inference_durations);
        })
        .def_property_readonly("grammar_compile_times", [](const RawPerfMetrics &rw) {
            return common_utils::get_ms(rw, &RawPerfMetrics::m_grammar_compile_times);
        });

    py::class_<SummaryStats>(m, "SummaryStats")
        .def(py::init<>())
        .def_readonly("mean", &SummaryStats::mean)
        .def_readonly("std", &SummaryStats::std)
        .def_readonly("min", &SummaryStats::min)
        .def_readonly("max", &SummaryStats::max)
        .def("as_tuple", [](const SummaryStats& self) {
            return py::make_tuple(self.mean, self.std, self.min, self.max);
        });

    py::class_<MeanStdPair>(m, "MeanStdPair")
        .def(py::init<>())
        .def_readonly("mean", &MeanStdPair::mean)
        .def_readonly("std", &MeanStdPair::std)
        .def("__iter__", [](const MeanStdPair &self) {
            return py::make_iterator(&self.mean, &self.std + 1);
        }, py::keep_alive<0, 1>());  // Keep object alive while the iterator is used;

    py::class_<PerfMetrics>(m, "PerfMetrics", perf_metrics_docstring)
        .def(py::init<>())
        .def("get_load_time", &PerfMetrics::get_load_time)
        .def("get_grammar_compiler_init_times", &PerfMetrics::get_grammar_compiler_init_times)
        .def("get_grammar_compile_time", &PerfMetrics::get_grammar_compile_time)
        .def("get_num_generated_tokens", &PerfMetrics::get_num_generated_tokens)
        .def("get_num_input_tokens", &PerfMetrics::get_num_input_tokens)
        .def("get_ttft", &PerfMetrics::get_ttft)
        .def("get_tpot", &PerfMetrics::get_tpot)
        .def("get_ipot", &PerfMetrics::get_ipot)
        .def("get_throughput", &PerfMetrics::get_throughput)
        .def("get_generate_duration", &PerfMetrics::get_generate_duration)
        .def("get_inference_duration", &PerfMetrics::get_inference_duration)
        .def("get_tokenization_duration", &PerfMetrics::get_tokenization_duration)
        .def("get_detokenization_duration", &PerfMetrics::get_detokenization_duration)
        .def("__add__", &PerfMetrics::operator+, py::arg("metrics"))
        .def("__iadd__", &PerfMetrics::operator+=, py::arg("right"))
        .def_readonly("raw_metrics", &PerfMetrics::raw_metrics);

    py::class_<ExtendedPerfMetrics, std::shared_ptr<ExtendedPerfMetrics>>(m, "ExtendedPerfMetrics", perf_metrics_docstring)
        .def(py::init<>())
        .def("get_load_time", &ExtendedPerfMetrics::get_load_time)
        .def("get_num_generated_tokens", &ExtendedPerfMetrics::get_num_generated_tokens)
        .def("get_num_input_tokens", &ExtendedPerfMetrics::get_num_input_tokens)
        .def("get_ttft", &ExtendedPerfMetrics::get_ttft)
        .def("get_tpot", &ExtendedPerfMetrics::get_tpot)
        .def("get_ipot", &ExtendedPerfMetrics::get_ipot)
        .def("get_throughput", &ExtendedPerfMetrics::get_throughput)
        .def("get_generate_duration", &ExtendedPerfMetrics::get_generate_duration)
        .def("get_inference_duration", &ExtendedPerfMetrics::get_inference_duration)
        .def("get_tokenization_duration", &ExtendedPerfMetrics::get_tokenization_duration)
        .def("get_detokenization_duration", &ExtendedPerfMetrics::get_detokenization_duration)
        .def("__add__", &ExtendedPerfMetrics::operator+, py::arg("metrics"))
        .def("__iadd__", &ExtendedPerfMetrics::operator+=, py::arg("right"))
        .def_readonly("raw_metrics", &ExtendedPerfMetrics::raw_metrics);

    py::class_<SDPerfMetrics, ExtendedPerfMetrics, std::shared_ptr<SDPerfMetrics>>(m, "SDPerfMetrics", sd_perf_metrics_docstring)
        .def("get_ttst", &SDPerfMetrics::get_ttst)
        .def("get_latency", &SDPerfMetrics::get_latency);

    py::class_<SDPerModelsPerfMetrics, SDPerfMetrics, std::shared_ptr<SDPerModelsPerfMetrics>>(m, "SDPerModelsPerfMetrics", sd_per_models_perf_metrics_docstring)
        .def("get_num_accepted_tokens", &SDPerModelsPerfMetrics::get_num_accepted_tokens)
        .def_readonly("main_model_metrics", &SDPerModelsPerfMetrics::main_model_metrics)
        .def_readonly("draft_model_metrics", &SDPerModelsPerfMetrics::draft_model_metrics);
}

