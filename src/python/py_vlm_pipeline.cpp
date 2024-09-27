// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>
#include "openvino/genai/vlm_pipeline.hpp"
#include "../cpp/src/tokenizers_path.hpp"
#include "./utils.hpp"

namespace py = pybind11;
namespace utils = ov::genai::pybind::utils;


auto vlm_generate_docstring = R"(
    Generates sequences for VLMs.

    :param prompt: input prompt
    :type prompt: str

    :param images: list of images 
    :type inputs: List[ov.Tensor]

    :param generation_config: generation_config
    :type generation_config: GenerationConfig or a Dict

    :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
    :type : Callable[[str], bool], ov.genai.StreamerBase

    :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
    :type : Dict

    :return: return results in decoded form
    :rtype: DecodedResults
)";

py::object call_vlm_generate(
    ov::genai::VLMPipeline& pipe, 
    const std::string& prompt,
    const std::vector<ov::Tensor>& images,
    const ov::genai::GenerationConfig& generation_config, 
    const utils::PyBindStreamerVariant& py_streamer, 
    const py::kwargs& kwargs
) {
    auto updated_config = *ov::genai::pybind::utils::update_config_from_kwargs(generation_config, kwargs);
    py::object results;
   // EncodedInputs tensor_data;
    ov::genai::StreamerVariant streamer = std::monostate();
    
    std::visit(utils::overloaded {
    [&streamer](const std::function<bool(py::str)>& py_callback){
        // Wrap python streamer with manual utf-8 decoding. Do not rely
        // on pybind automatic decoding since it raises exceptions on incomplete strings.
        auto callback_wrapped = [&py_callback](std::string subword) -> bool {
            auto py_str = PyUnicode_DecodeUTF8(subword.data(), subword.length(), "replace");
            return py_callback(py::reinterpret_borrow<py::str>(py_str));
        };
        streamer = callback_wrapped;
    },
    [&streamer](std::shared_ptr<StreamerBase> streamer_cls){
        streamer = streamer_cls;
    },
    [](std::monostate none){ /*streamer is already a monostate */ }
    }, py_streamer);
    
    return py::cast(pipe.generate(prompt, images, updated_config, streamer));
}

void init_vlm_pipeline(py::module_& m) {
    py::class_<ov::genai::VLMPipeline>(m, "VLMPipeline", "This class is used for generation with VLMs")
        .def(py::init([](
            const std::string& model_path, 
            const std::string& device,
            const std::map<std::string, py::object>& config,
            const ov::Core& core
        ) {
            ScopedVar env_manager(utils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::VLMPipeline>(model_path, device, utils::properties_to_any_map(config));
        }),
        py::arg("model_path"), "folder with openvino_model.xml and openvino_tokenizer[detokenizer].xml files", 
        py::arg("device") = "CPU", "device on which inference will be done",
        py::arg("config") = ov::AnyMap({}), "openvino.properties map",
        py::arg("core") = ov::Core(), "openvino.Core object",
        R"(
            VLMPipeline class constructor.
            model_path (str): Path to the model file.
            device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
            Add {"scheduler_config": ov_genai.SchedulerConfig} to config properties to create continuous batching pipeline.
            core (openvino.Core): openvino.Core instance.
        )")

        .def(py::init([](
            const std::string& model_path,
            const ov::genai::Tokenizer& tokenizer,
            const std::string& device,
            const std::map<std::string, py::object>& config,
            const ov::Core& core
        ) {
            ScopedVar env_manager(utils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::VLMPipeline>(model_path, tokenizer, device, utils::properties_to_any_map(config));
        }),
        py::arg("model_path"),
        py::arg("tokenizer"),
        py::arg("device") = "CPU",
        py::arg("config") = ov::AnyMap({}), "openvino.properties map",
        py::arg("core") = ov::Core(), "openvino.Core object",
        R"(
            VLMPipeline class constructor for manualy created openvino_genai.Tokenizer.
            model_path (str): Path to the model file.
            tokenizer (openvino_genai.Tokenizer): tokenizer object.
            device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
            Add {"scheduler_config": ov_genai.SchedulerConfig} to config properties to create continuous batching pipeline.
            core (openvino.Core): openvino.Core instance.
        )")
        .def("start_chat", &ov::genai::VLMPipeline::start_chat, py::arg("system_message") = "")
        .def("finish_chat", &ov::genai::VLMPipeline::finish_chat) 
        .def("get_generation_config", &ov::genai::VLMPipeline::get_generation_config)
        .def(
            "generate", 
            [](ov::genai::VLMPipeline& pipe, 
                const std::string& prompt,
                const std::vector<ov::Tensor>& images,
                const ov::genai::GenerationConfig& generation_config, 
                const utils::PyBindStreamerVariant& streamer,
                const py::kwargs& kwargs
            ) {
                return call_vlm_generate(pipe, prompt, images, generation_config, streamer, kwargs);
            },
            py::arg("prompt"), "Input string",
            py::arg("images"), "Input images",
            py::arg("generation_config") = std::nullopt, "generation_config",
            py::arg("streamer") = std::monostate(), "streamer",
            (vlm_generate_docstring + std::string(" \n ")).c_str()
        )
        .def(
            "__call__", 
            [](ov::genai::VLMPipeline& pipe, 
                const std::string& prompt,
                const std::vector<ov::Tensor>& images,
                const ov::genai::GenerationConfig& generation_config, 
                const utils::PyBindStreamerVariant& streamer,
                const py::kwargs& kwargs
            ) {
                return call_vlm_generate(pipe, prompt, images, generation_config, streamer, kwargs);
            },
            py::arg("prompt"), "Input string",
            py::arg("images"), "Input images",
            py::arg("generation_config") = std::nullopt, "generation_config",
            py::arg("streamer") = std::monostate(), "streamer",
            (vlm_generate_docstring + std::string(" \n ")).c_str()
        );
}