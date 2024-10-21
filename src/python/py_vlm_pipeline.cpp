// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>
#include "openvino/genai/visual_language/pipeline.hpp"
#include "tokenizers_path.hpp"
#include "py_utils.hpp"

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

auto vlm_generate_kwargs_docstring = R"(
    Generates sequences for VLMs.

    :param prompt: input prompt
    :type prompt: str

    :param kwargs: arbitrary keyword arguments with keys corresponding to generate params. 
    
    Expected parameters list: 
    image: ov.Tensor - input image,
    images: List[ov.Tensor] - input images,
    generation_config: GenerationConfig,
    streamer: Callable[[str], bool], ov.genai.StreamerBase - streamer either as a lambda with a boolean returning flag whether generation should be stopped

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
    ov::genai::StreamerVariant streamer = ov::genai::pybind::utils::pystreamer_to_streamer(py_streamer);
    
    return py::cast(pipe.generate(prompt, images, updated_config, streamer));
}

py::object call_vlm_generate(
    ov::genai::VLMPipeline& pipe, 
    const std::string& prompt,
    const py::kwargs& kwargs
) {
    ov::AnyMap params = {};

    for (const auto& item : kwargs) {
        std::string key = py::cast<std::string>(item.first);
        py::object value = py::cast<py::object>(item.second);

        if (key == "images") {
            params.insert({ov::genai::images(std::move(py::cast<std::vector<ov::Tensor>>(item.second)))});
        } else if (key == "image") {
            params.insert({ov::genai::image(std::move(py::cast<ov::Tensor>(item.second)))});
        } else if (key == "generation_config") {
            params.insert({ov::genai::generation_config(std::move(py::cast<ov::genai::GenerationConfig>(item.second)))});
        } else if (key == "streamer") {
            auto py_streamer = py::cast<utils::PyBindStreamerVariant>(value);
            params.insert({ov::genai::streamer(std::move(ov::genai::pybind::utils::pystreamer_to_streamer(py_streamer)))});

        } else {
            throw(std::invalid_argument("'" + key + "' is unexpected parameter name. "
                                        "Use help(openvino_genai.VLMPipeline.generate) to get list of acceptable parameters."));
        }
    }
    
    return py::cast(pipe.generate(prompt, params));
}

void init_vlm_pipeline(py::module_& m) {
    py::class_<ov::genai::VLMPipeline>(m, "VLMPipeline", "This class is used for generation with VLMs")
        .def(py::init([](
            const std::string& models_path, 
            const std::string& device,
            const std::map<std::string, py::object>& config
        ) {
            ScopedVar env_manager(utils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::VLMPipeline>(models_path, device, utils::properties_to_any_map(config));
        }),
        py::arg("models_path"), "folder with exported model files", 
        py::arg("device"), "device on which inference will be done",
        py::arg("config") = ov::AnyMap({}), "openvino.properties map"
        R"(
            VLMPipeline class constructor.
            models_path (str): Path to the folder with exported model files.
            device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
        )")

        .def("start_chat", &ov::genai::VLMPipeline::start_chat, py::arg("system_message") = "")
        .def("finish_chat", &ov::genai::VLMPipeline::finish_chat) 
        .def("get_tokenizer", &ov::genai::VLMPipeline::get_tokenizer)
        .def("get_generation_config", &ov::genai::VLMPipeline::get_generation_config)
        .def("set_generation_config", &ov::genai::VLMPipeline::set_generation_config)
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
            "generate", 
            [](ov::genai::VLMPipeline& pipe, 
                const std::string& prompt,
                const py::kwargs& kwargs
            ) {
                return call_vlm_generate(pipe, prompt, kwargs);
            },
            py::arg("prompt"), "Input string",
            (vlm_generate_kwargs_docstring + std::string(" \n ")).c_str()
        );
}
