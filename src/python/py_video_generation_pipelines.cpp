// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <filesystem>

#include "openvino/genai/video_generation/text2video_pipeline.hpp"
#include "py_utils.hpp"
#include "tokenizer/tokenizers_path.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

namespace {

auto text2video_pipeline_class_docstring = R"(
    Text-to-video generation pipeline.

    This pipeline generates videos from text prompts using exported video generation models.
    Use :meth:`generate` for inference and :meth:`set_generation_config` / :meth:`reshape`
    to customize generation behavior.
)";

auto text2video_pipeline_init_docstring = R"(
    Text2VideoPipeline class constructor.

    :param models_path: Path to a directory with exported text-to-video model files.
    :type models_path: os.PathLike

    .. note::
       This constructor does not compile models internally. Call :meth:`compile`
       (and typically :meth:`reshape`) before invoking :meth:`generate`.
)";

auto text2video_pipeline_init_with_device_docstring = R"(
    Text2VideoPipeline class constructor.

    :param models_path: Path to a directory with exported text-to-video model files.
    :type models_path: os.PathLike

    :param device: Device on which inference will be done (for example, ``CPU`` or ``GPU``).
    :type device: str

    :param kwargs: Optional OpenVINO device properties.
    :type kwargs: dict

    This constructor compiles the underlying models for the specified device,
    so the pipeline is ready for inference immediately after construction and
    :meth:`compile` does not need to be called explicitly.
)";

auto text2video_pipeline_generate_docstring = R"(
    Generates a video from a text prompt.

    :param prompt: Input text prompt.
    :type prompt: str

    :param kwargs: Optional generation overrides and runtime options.
        Typical keys include ``negative_prompt``, ``num_frames``, ``height``,
        ``width``, ``num_inference_steps``, ``guidance_scale``, and ``callback``.
    :type kwargs: dict

    :return: Video generation output with generated frames and performance metrics.
    :rtype: VideoGenerationResult

    **Example**

    .. code-block:: python

        import openvino_genai as ov_genai

        pipe = ov_genai.Text2VideoPipeline("./tiny-random-ltx-video", "CPU")
        result = pipe.generate(
            "A calm lake at sunrise",
            num_frames=9,
            height=32,
            width=32,
            num_inference_steps=2,
        )
        video = result.video
)";

} // namespace

void init_video_generation_pipelines(py::module_& m) {
    py::class_<ov::genai::VideoGenerationPerfMetrics, ov::genai::ImageGenerationPerfMetrics>(m, "VideoGenerationPerfMetrics")
        .def(py::init<>());

    py::class_<ov::genai::VideoGenerationConfig>(m, "VideoGenerationConfig")
        .def(py::init<>())
        .def_readwrite("guidance_rescale", &ov::genai::VideoGenerationConfig::guidance_rescale)
        .def_readwrite("num_frames", &ov::genai::VideoGenerationConfig::num_frames)
        .def_readwrite("frame_rate", &ov::genai::VideoGenerationConfig::frame_rate)
        .def_readwrite("num_videos_per_prompt", &ov::genai::VideoGenerationConfig::num_videos_per_prompt)
        .def_readwrite("negative_prompt", &ov::genai::VideoGenerationConfig::negative_prompt)
        .def_readwrite("generator", &ov::genai::VideoGenerationConfig::generator)
        .def_readwrite("guidance_scale", &ov::genai::VideoGenerationConfig::guidance_scale)
        .def_readwrite("height", &ov::genai::VideoGenerationConfig::height)
        .def_readwrite("width", &ov::genai::VideoGenerationConfig::width)
        .def_readwrite("num_inference_steps", &ov::genai::VideoGenerationConfig::num_inference_steps)
        .def_readwrite("max_sequence_length", &ov::genai::VideoGenerationConfig::max_sequence_length);

    py::class_<ov::genai::VideoGenerationResult>(m, "VideoGenerationResult")
        .def_readonly("video", &ov::genai::VideoGenerationResult::video)
        .def_readonly("perf_metrics", &ov::genai::VideoGenerationResult::performance_stat);

    py::class_<ov::genai::Text2VideoPipeline>(m, "Text2VideoPipeline", text2video_pipeline_class_docstring)
        .def(py::init([](const std::filesystem::path& models_path) {
                 ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
                 return std::make_unique<ov::genai::Text2VideoPipeline>(models_path);
             }),
             py::arg("models_path"),
             text2video_pipeline_init_docstring)
        .def(
            py::init([](const std::filesystem::path& models_path, const std::string& device, const py::kwargs& kwargs) {
                ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
                return std::make_unique<ov::genai::Text2VideoPipeline>(models_path,
                                                                       device,
                                                                       pyutils::kwargs_to_any_map(kwargs));
            }),
            py::arg("models_path"),
            py::arg("device"),
            text2video_pipeline_init_with_device_docstring)
        .def("get_generation_config",
             &ov::genai::Text2VideoPipeline::get_generation_config,
             py::return_value_policy::copy)
        .def("set_generation_config", &ov::genai::Text2VideoPipeline::set_generation_config, py::arg("config"))
        .def("reshape",
             &ov::genai::Text2VideoPipeline::reshape,
             py::arg("num_videos_per_prompt"),
             py::arg("num_frames"),
             py::arg("height"),
             py::arg("width"),
             py::arg("guidance_scale"))
        .def(
            "compile",
            [](ov::genai::Text2VideoPipeline& pipe, const std::string& device, const py::kwargs& kwargs) {
                auto properties = pyutils::kwargs_to_any_map(kwargs);
                py::gil_scoped_release rel;
                pipe.compile(device, properties);
            },
            py::arg("device"))
        .def(
            "compile",
            [](ov::genai::Text2VideoPipeline& pipe,
               const std::string& text_encode_device,
               const std::string& denoise_device,
               const std::string& vae_device,
               const py::kwargs& kwargs) {
                auto properties = pyutils::kwargs_to_any_map(kwargs);
                py::gil_scoped_release rel;
                pipe.compile(text_encode_device, denoise_device, vae_device, properties);
            },
            py::arg("text_encode_device"),
            py::arg("denoise_device"),
            py::arg("vae_device"))
        .def(
            "generate",
            [](ov::genai::Text2VideoPipeline& pipe, const std::string& prompt, const py::kwargs& kwargs) {
                auto properties = pyutils::kwargs_to_any_map(kwargs);
                ov::genai::VideoGenerationResult result;
                {
                    py::gil_scoped_release rel;
                    result = pipe.generate(prompt, properties);
                }
                return result;
            },
            py::arg("prompt"),
            text2video_pipeline_generate_docstring);
}
