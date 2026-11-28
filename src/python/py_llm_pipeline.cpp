// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>

#include "openvino/genai/llm_pipeline.hpp"

#include "tokenizer/tokenizers_path.hpp"
#include "py_utils.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

using ov::genai::OptionalGenerationConfig;
using ov::genai::LLMPipeline;
using ov::genai::TokenizedInputs;
using ov::genai::EncodedInputs;
using ov::genai::StreamerVariant;
using ov::genai::DecodedResults;
using ov::genai::Tokenizer;
using ov::genai::draft_model;
using ov::genai::ChatHistory;

namespace {

auto generate_docstring = R"(
    Generates sequences or tokens for LLMs. If input is a string or list of strings then resulting sequences will be already detokenized.

    :param inputs: inputs in the form of string, list of strings, chat history or tokenized input_ids
    :type inputs: str, list[str], ov.genai.TokenizedInputs, or ov.Tensor

    :param generation_config: generation_config
    :type generation_config: GenerationConfig or a dict

    :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
    :type : Callable[[str], bool], ov.genai.StreamerBase

    :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
    :type : dict

    :return: return results in encoded, or decoded form depending on inputs type
    :rtype: DecodedResults, EncodedResults, str
)";

py::object call_common_generate(
    LLMPipeline& pipe,
    const std::variant<ov::Tensor, TokenizedInputs, std::string, std::vector<std::string>, ChatHistory>& inputs,
    const OptionalGenerationConfig& config,
    const pyutils::PyBindStreamerVariant& py_streamer,
    const py::kwargs& kwargs
) {
    const ov::genai::GenerationConfig& default_config = config.value_or(pipe.get_generation_config());
    auto updated_config = pyutils::update_config_from_kwargs(default_config, kwargs);

    py::object results;
    StreamerVariant streamer = pyutils::pystreamer_to_streamer(py_streamer);

    // Call suitable generate overload for each type of input.
    std::visit(pyutils::overloaded {
    [&](ov::Tensor ov_tensor) {
        ov::genai::EncodedResults encoded_results;
        {
            py::gil_scoped_release rel;
            encoded_results = pipe.generate(ov_tensor, updated_config, streamer);
        }
        results = py::cast(encoded_results);
    },
    [&](TokenizedInputs tokenized_input) {
        ov::genai::EncodedResults encoded_results;
        {
            py::gil_scoped_release rel;
            encoded_results = pipe.generate(tokenized_input, updated_config, streamer);
        }
        results = py::cast(encoded_results);
    },
    [&](std::string string_input) {
        DecodedResults res;
        {
            py::gil_scoped_release rel;
            res = pipe.generate(string_input, updated_config, streamer);
        }
        // If input was a string return a single string otherwise return DecodedResults.
        if (updated_config.num_return_sequences == 1) {
            results = py::cast<py::object>(pyutils::handle_utf8(res.texts[0]));
        } else {
            results = py::cast(res);
        }
    },
    [&](std::vector<std::string> string_input) {
        // For DecodedResults texts getter already handles utf8 decoding.
        DecodedResults res;
        {
            py::gil_scoped_release rel;
            res = pipe.generate(string_input, updated_config, streamer);
        }
        results = py::cast(res);
    },
    [&](ChatHistory history) {
        DecodedResults res;
        {
            py::gil_scoped_release rel;
            res = pipe.generate(history, updated_config, streamer);
        }
        results = py::cast(res);
    }},
    inputs);
    return results;
}

// Lets Python classes implement ov::genai::Drafter.
class PyDrafter : public ov::genai::Drafter {
public:
    using ov::genai::Drafter::Drafter;

    size_t max_draft_tokens() const override {
        PYBIND11_OVERRIDE_PURE(size_t, ov::genai::Drafter, max_draft_tokens);
    }

    void start(const std::vector<int64_t>& prompt, const ov::genai::GenerationConfig& config) override {
        PYBIND11_OVERRIDE_PURE(void, ov::genai::Drafter, start, prompt, config);
    }

    ov::genai::DraftTree propose(int64_t root, int64_t position, size_t max_nodes) override {
        PYBIND11_OVERRIDE_PURE(ov::genai::DraftTree, ov::genai::Drafter, propose, root, position, max_nodes);
    }

    void accept(const std::vector<int64_t>& tokens,
                const std::vector<int64_t>& positions,
                const ov::Tensor& features) override {
        PYBIND11_OVERRIDE_PURE(void, ov::genai::Drafter, accept, tokens, positions, features);
    }

    void finish() override {
        PYBIND11_OVERRIDE(void, ov::genai::Drafter, finish);
    }
};

} // namespace

extern char generation_config_docstring[];

void init_llm_pipeline(py::module_& m) {
    py::class_<LLMPipeline>(m, "LLMPipeline", "This class is used for generation with LLMs")
        // init(model_path, tokenizer, device, config, kwargs) should be defined before init(model_path, device, config, kwargs) 
        // to prevent tokenizer treated as kwargs argument
        .def(py::init([](
            const std::filesystem::path& models_path,
            const Tokenizer& tokenizer,
            const std::string& device,
            const std::map<std::string, py::object>& config,
            const py::kwargs& kwargs
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            ov::AnyMap properties = pyutils::kwargs_to_any_map(kwargs);
            if (config.size()) {
                PyErr_WarnEx(PyExc_DeprecationWarning, 
                         "'config' parameters is deprecated, please use kwargs to pass config properties instead.", 
                         1);
                auto config_properties = pyutils::properties_to_any_map(config);
                properties.insert(config_properties.begin(), config_properties.end());
            }
            py::gil_scoped_release rel;
            return std::make_unique<LLMPipeline>(models_path, tokenizer, device, properties);
        }),
        py::arg("models_path"),
        py::arg("tokenizer"),
        py::arg("device"),
        py::arg("config") = ov::AnyMap({}), "openvino.properties map",
        R"(
            LLMPipeline class constructor for manually created openvino_genai.Tokenizer.
            models_path (os.PathLike): Path to the model file.
            tokenizer (openvino_genai.Tokenizer): tokenizer object.
            device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
            Add {"scheduler_config": ov_genai.SchedulerConfig} to config properties to create continuous batching pipeline.
            kwargs: Device properties.
        )")

        .def(py::init([](
            const std::filesystem::path& models_path,
            const std::string& device,
            const std::map<std::string, py::object>& config,
            const py::kwargs& kwargs
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            ov::AnyMap properties = pyutils::kwargs_to_any_map(kwargs);
            if (config.size()) {
                PyErr_WarnEx(PyExc_DeprecationWarning, 
                         "'config' parameters is deprecated, please use kwargs to pass config properties instead.", 
                         1);
                auto config_properties = pyutils::properties_to_any_map(config);
                properties.insert(config_properties.begin(), config_properties.end());
            }
            py::gil_scoped_release rel;
            return std::make_unique<LLMPipeline>(models_path, device, properties);
        }),
        py::arg("models_path"), "folder with openvino_model.xml and openvino_tokenizer[detokenizer].xml files",
        py::arg("device"), "device on which inference will be done",
        py::arg("config") = ov::AnyMap({}), "openvino.properties map",
        R"(
            LLMPipeline class constructor.
            models_path (os.PathLike): Path to the model file.
            device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
            Add {"scheduler_config": ov_genai.SchedulerConfig} to config properties to create continuous batching pipeline.
            kwargs: Device properties.
        )")

        .def(py::init([](
            const std::string& model,
            const ov::Tensor& weights,
            const ov::genai::Tokenizer& tokenizer,
            const std::string& device,
            OptionalGenerationConfig generation_config,
            const py::kwargs& kwargs
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            ov::AnyMap properties = pyutils::kwargs_to_any_map(kwargs);
            if (!generation_config.has_value()) {
                generation_config = ov::genai::GenerationConfig();
            }
            py::gil_scoped_release rel;
            return std::make_unique<LLMPipeline>(model, weights, tokenizer, device, properties, *generation_config);
        }),
        py::arg("model"), "string with pre-read model",
        py::arg("weights"), "ov::Tensor with pre-read model weights",
        py::arg("tokenizer"), "genai Tokenizers",
        py::arg("device"), "device on which inference will be done",
        py::arg("generation_config") = py::none(), "genai GenerationConfig (default: None, will use empty config)",
        R"(
            LLMPipeline class constructor.
            model (str): Pre-read model.
            weights (ov.Tensor): Pre-read model weights.
            tokenizer (str): Genai Tokenizers.
            device (str): Device to run the model on (e.g., CPU, GPU).
            generation_config {ov_genai.GenerationConfig} Genai GenerationConfig. Default is an empty config.
            kwargs: Device properties.
        )")

        .def(
            "generate",
            [](LLMPipeline& pipe,
                const std::variant<ov::Tensor, TokenizedInputs, std::string, std::vector<std::string>, ChatHistory>& inputs,
                const OptionalGenerationConfig& generation_config,
                const pyutils::PyBindStreamerVariant& streamer,
                const py::kwargs& kwargs
            ) -> py::typing::Union<ov::genai::EncodedResults, ov::genai::DecodedResults> {
                return call_common_generate(pipe, inputs, generation_config, streamer, kwargs);
            },
            py::arg("inputs"), "Input string, or list of string, chat history or encoded tokens",
            py::arg("generation_config") = std::nullopt, "generation_config",
            py::arg("streamer") = std::monostate(), "streamer",
            (generate_docstring + std::string(" \n ") + generation_config_docstring).c_str()
        )

        .def(
            "__call__",
            [](LLMPipeline& pipe,
                const std::variant<ov::Tensor, TokenizedInputs, std::string, std::vector<std::string>, ChatHistory>& inputs,
                const OptionalGenerationConfig& generation_config,
                const pyutils::PyBindStreamerVariant& streamer,
                const py::kwargs& kwargs
            ) -> py::typing::Union<ov::genai::EncodedResults, ov::genai::DecodedResults> {
                return call_common_generate(pipe, inputs, generation_config, streamer, kwargs);
            },
            py::arg("inputs"), "Input string, or list of string, chat history or encoded tokens",
            py::arg("generation_config") = std::nullopt, "generation_config",
            py::arg("streamer") = std::monostate(), "streamer",
            (generate_docstring + std::string(" \n ") + generation_config_docstring).c_str()
        )

        .def("get_tokenizer", &LLMPipeline::get_tokenizer)
        .def("start_chat", [](LLMPipeline& pipe, const std::string& system_message) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "start_chat() / finish_chat() API is deprecated and will be removed in the next major release. "
                         "Please use generate() with ChatHistory argument.",
                         1);
            pipe.start_chat(system_message);
        }, py::arg("system_message") = "")
        .def("finish_chat", [](LLMPipeline& pipe) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "start_chat() / finish_chat() API is deprecated and will be removed in the next major release. "
                         "Please use generate() with ChatHistory argument.",
                         1);
            pipe.finish_chat();
        })
        .def("get_generation_config", &LLMPipeline::get_generation_config, py::return_value_policy::copy)
        .def("set_generation_config", &LLMPipeline::set_generation_config, py::arg("config"));

    m.def("draft_model", [](
            const std::filesystem::path& models_path,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return draft_model(models_path, device, pyutils::kwargs_to_any_map(kwargs)).second;
        },
        py::arg("models_path"), "folder with openvino_model.xml and openvino_tokenizer[detokenizer].xml files",
        py::arg("device") = "", "device on which inference will be performed");

    py::class_<ov::genai::DraftTree>(m,
                                     "DraftTree",
                                     R"(
            Candidate tokens proposed by a Drafter for one verification step, as a tree.
            Node i holds token tokens[i]; parents[i] is -1 when its parent is the root (the last generated token) or
            the index of an earlier node. A chain of k guesses has parents [-1, 0, ..., k - 2].
        )")
        .def(py::init<>())
        .def(py::init([](std::vector<int64_t> tokens, std::vector<int64_t> parents) {
                 return ov::genai::DraftTree{std::move(tokens), std::move(parents)};
             }),
             py::arg("tokens"),
             py::arg("parents"))
        .def_readwrite("tokens", &ov::genai::DraftTree::tokens)
        .def_readwrite("parents", &ov::genai::DraftTree::parents);

    py::class_<ov::genai::Drafter, PyDrafter, std::shared_ptr<ov::genai::Drafter>>(m,
                                                                                   "Drafter",
                                                                                   R"(
            Base class for user-defined drafters of speculative decoding: LLMPipeline(models_path, "NPU", drafter=...).
            Each step the pipeline asks for a DraftTree following the last generated token, verifies it with the main
            model in one inference (greedy) and reports the tokens that entered the main model's KV cache (the prompt,
            then the accepted tokens) with the main model's last_hidden_state rows for them. Supported with the main
            model on NPU, greedy decoding and batch size 1. Tensors passed to the drafter are only valid during the
            call.
        )")
        .def(py::init<>())
        .def("max_draft_tokens",
             &ov::genai::Drafter::max_draft_tokens,
             "Largest tree a proposal may have; the main model verifies max_draft_tokens() + 1 tokens per step.")
        .def("start",
             &ov::genai::Drafter::start,
             py::arg("prompt"),
             py::arg("config"),
             "New generate() call, before the main model processes the prompt: prompt token ids and the generation "
             "config. The prompt's tokens and features follow in accept().")
        .def("propose",
             &ov::genai::Drafter::propose,
             py::arg("root"),
             py::arg("position"),
             py::arg("max_nodes"),
             "Returns a DraftTree of at most max_nodes candidates following root, the last generated token at "
             "position.")
        .def("accept",
             &ov::genai::Drafter::accept,
             py::arg("tokens"),
             py::arg("positions"),
             py::arg("features"),
             "Tokens that entered the main model's KV cache, their positions and main model last_hidden_state rows "
             "(empty without that output): first the prompt, then after every step the root followed by the accepted "
             "tokens. The main model's next token (the next root) follows them.")
        .def("finish", &ov::genai::Drafter::finish, "Called when a generate() call ends.");
}
