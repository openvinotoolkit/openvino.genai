// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>

#include "openvino/genai/image_generation/text2image_pipeline.hpp"
#include "openvino/genai/image_generation/image2image_pipeline.hpp"
#include "openvino/genai/image_generation/inpainting_pipeline.hpp"

#include "tokenizers_path.hpp"
#include "py_utils.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

namespace ov {
namespace genai {

/// Trampoline class to support inheritance from Generator in Python
class PyGenerator : public ov::genai::Generator {
public:
    float next() override {
        PYBIND11_OVERRIDE_PURE(float, Generator, next);
    }

    ov::Tensor randn_tensor(const ov::Shape& shape) override {
        PYBIND11_OVERRIDE(ov::Tensor, Generator, randn_tensor, shape);
    }
};

} // namespace genai
} // namespace ov

namespace {

auto text2image_generate_docstring = R"(
    Generates images for text-to-image models.

    :param prompt: input prompt
    :type prompt: str

    :param kwargs: arbitrary keyword arguments with keys corresponding to generate params.

    Expected parameters list:
    prompt_2: str - second prompt,
    prompt_3: str - third prompt,
    negative_prompt: str - negative prompt,
    negative_prompt_2: str - second negative prompt,
    negative_prompt_3: str - third negative prompt,
    num_images_per_prompt: int - number of images, that should be generated per prompt,
    guidance_scale: float - guidance scale,
    generation_config: GenerationConfig,
    height: int - height of resulting images,
    width: int - width of resulting images,
    num_inference_steps: int - number of inference steps,
    generator: openvino_genai.CppStdGenerator or class inherited from openvino_genai.Generator - random generator,
    adapters: LoRA adapters,
    strength: strength for image to image generation. 1.0f means initial image is fully noised,
    max_sequence_length: int - length of t5_encoder_model input

    :return: ov.Tensor with resulting images
    :rtype: ov.Tensor
)";



} // namespace

void init_clip_text_model(py::module_& m);
void init_clip_text_model_with_projection(py::module_& m);
void init_t5_encoder_model(py::module_& m);
void init_unet2d_condition_model(py::module_& m);
void init_sd3_transformer_2d_model(py::module_& m);
void init_flux_transformer_2d_model(py::module_& m);
void init_autoencoder_kl(py::module_& m);

void init_image_generation_pipelines(py::module_& m) {
    py::class_<ov::genai::Generator, ov::genai::PyGenerator, std::shared_ptr<ov::genai::Generator>>(m, "Generator", "This class is used for storing pseudo-random generator.")
        .def(py::init<>());

    py::class_<ov::genai::CppStdGenerator, ov::genai::Generator, std::shared_ptr<ov::genai::CppStdGenerator>>(m, "CppStdGenerator", "This class wraps std::mt19937 pseudo-random generator.")
        .def(py::init([](uint32_t seed) {
            return std::make_unique<ov::genai::CppStdGenerator>(seed);
        }), 
        py::arg("seed"))
        .def("next", &ov::genai::CppStdGenerator::next)
        .def("randn_tensor", &ov::genai::CppStdGenerator::randn_tensor, py::arg("shape"));

    // init image generation models
    init_clip_text_model(m);
    init_clip_text_model_with_projection(m);
    init_t5_encoder_model(m);
    init_unet2d_condition_model(m);
    init_sd3_transformer_2d_model(m);
    init_flux_transformer_2d_model(m);
    init_autoencoder_kl(m);

    auto image_generation_scheduler = py::class_<ov::genai::Scheduler, std::shared_ptr<ov::genai::Scheduler>>(m, "Scheduler", "Scheduler for image generation pipelines.");
    py::enum_<ov::genai::Scheduler::Type>(image_generation_scheduler, "Type")
        .value("AUTO", ov::genai::Scheduler::Type::AUTO)
        .value("LCM", ov::genai::Scheduler::Type::LCM)
        .value("LMS_DISCRETE", ov::genai::Scheduler::Type::LMS_DISCRETE)
        .value("DDIM", ov::genai::Scheduler::Type::DDIM)
        .value("EULER_DISCRETE", ov::genai::Scheduler::Type::EULER_DISCRETE)
        .value("FLOW_MATCH_EULER_DISCRETE", ov::genai::Scheduler::Type::FLOW_MATCH_EULER_DISCRETE);
    image_generation_scheduler.def_static("from_config",
        &ov::genai::Scheduler::from_config,
        py::arg("scheduler_config_path"),
        py::arg_v("scheduler_type", ov::genai::Scheduler::Type::AUTO, "Scheduler.Type.AUTO"));

    py::class_<ov::genai::ImageGenerationConfig>(m, "ImageGenerationConfig", "This class is used for storing generation config for image generation pipeline.")
        .def(py::init<>())
        .def_readwrite("prompt_2", &ov::genai::ImageGenerationConfig::prompt_2)
        .def_readwrite("prompt_3", &ov::genai::ImageGenerationConfig::prompt_3)
        .def_readwrite("negative_prompt", &ov::genai::ImageGenerationConfig::negative_prompt)
        .def_readwrite("negative_prompt_2", &ov::genai::ImageGenerationConfig::negative_prompt_2)
        .def_readwrite("negative_prompt_3", &ov::genai::ImageGenerationConfig::negative_prompt_3)
        .def_readwrite("generator", &ov::genai::ImageGenerationConfig::generator)
        .def_readwrite("guidance_scale", &ov::genai::ImageGenerationConfig::guidance_scale)
        .def_readwrite("height", &ov::genai::ImageGenerationConfig::height)
        .def_readwrite("width", &ov::genai::ImageGenerationConfig::width)
        .def_readwrite("num_inference_steps", &ov::genai::ImageGenerationConfig::num_inference_steps)
        .def_readwrite("num_images_per_prompt", &ov::genai::ImageGenerationConfig::num_images_per_prompt)
        .def_readwrite("adapters", &ov::genai::ImageGenerationConfig::adapters)
        .def_readwrite("strength", &ov::genai::ImageGenerationConfig::strength)
        .def_readwrite("max_sequence_length", &ov::genai::ImageGenerationConfig::max_sequence_length)
        .def("validate", &ov::genai::ImageGenerationConfig::validate)
        .def("update_generation_config", [](
            ov::genai::ImageGenerationConfig config,
            const py::kwargs& kwargs) {
            config.update_generation_config(pyutils::kwargs_to_any_map(kwargs));
        });

    auto text2image_pipeline = py::class_<ov::genai::Text2ImagePipeline>(m, "Text2ImagePipeline", "This class is used for generation with text-to-image models.")
        .def(py::init([](const std::filesystem::path& models_path) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::Text2ImagePipeline>(models_path);
        }),
        py::arg("models_path"), "folder with exported model files.",
        R"(
            Text2ImagePipeline class constructor.
            models_path (os.PathLike): Path to the folder with exported model files.
        )")
        .def(py::init([](
            const std::filesystem::path& models_path,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::Text2ImagePipeline>(models_path, device, pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("models_path"), "folder with exported model files.",
        py::arg("device"), "device on which inference will be done",
        R"(
            Text2ImagePipeline class constructor.
            models_path (os.PathLike): Path with exported model files.
            device (str): Device to run the model on (e.g., CPU, GPU).
            kwargs: Text2ImagePipeline properties
        )")
        .def("get_generation_config", &ov::genai::Text2ImagePipeline::get_generation_config)
        .def("set_generation_config", &ov::genai::Text2ImagePipeline::set_generation_config, py::arg("generation_config"))
        .def("set_scheduler", &ov::genai::Text2ImagePipeline::set_scheduler, py::arg("scheduler"))
        .def("reshape", &ov::genai::Text2ImagePipeline::reshape, py::arg("num_images_per_prompt"), py::arg("height"), py::arg("width"), py::arg("guidance_scale"))
        .def_static("stable_diffusion", &ov::genai::Text2ImagePipeline::stable_diffusion, py::arg("scheduler"), py::arg("clip_text_model"), py::arg("unet"), py::arg("vae"))
        .def_static("latent_consistency_model", &ov::genai::Text2ImagePipeline::latent_consistency_model, py::arg("scheduler"), py::arg("clip_text_model"), py::arg("unet"), py::arg("vae"))
        .def_static("stable_diffusion_xl", &ov::genai::Text2ImagePipeline::stable_diffusion_xl, py::arg("scheduler"), py::arg("clip_text_model"), py::arg("clip_text_model_with_projection"), py::arg("unet"), py::arg("vae"))
        .def_static("stable_diffusion_3", py::overload_cast<const std::shared_ptr<ov::genai::Scheduler>&, const ov::genai::CLIPTextModelWithProjection&, const ov::genai::CLIPTextModelWithProjection&, const ov::genai::T5EncoderModel&,
                                                            const ov::genai::SD3Transformer2DModel&, const ov::genai::AutoencoderKL&>(&ov::genai::Text2ImagePipeline::stable_diffusion_3),
            py::arg("scheduler"), py::arg("clip_text_model_1"), py::arg("clip_text_model_2"), py::arg("t5_encoder_model"), py::arg("transformer"), py::arg("vae"))
        .def_static("stable_diffusion_3", py::overload_cast<const std::shared_ptr<ov::genai::Scheduler>&, const ov::genai::CLIPTextModelWithProjection&, const ov::genai::CLIPTextModelWithProjection&,
                                                            const ov::genai::SD3Transformer2DModel&, const ov::genai::AutoencoderKL&>(&ov::genai::Text2ImagePipeline::stable_diffusion_3),
            py::arg("scheduler"), py::arg("clip_text_model_1"), py::arg("clip_text_model_2"), py::arg("transformer"), py::arg("vae"))
        .def_static("flux", &ov::genai::Text2ImagePipeline::flux, py::arg("scheduler"), py::arg("clip_text_model"), py::arg("t5_encoder_model"), py::arg("transformer"), py::arg("vae"))
        .def(
            "compile",
            [](ov::genai::Text2ImagePipeline& pipe,
                const std::string& device,
                const py::kwargs& kwargs
            ) {
                pipe.compile(device,  pyutils::kwargs_to_any_map(kwargs));
            },
            py::arg("device"), "device on which inference will be done",
            R"(
                Compiles the model.
                device (str): Device to run the model on (e.g., CPU, GPU).
                kwargs: Device properties.
            )")
        .def(
            "generate",
            [](ov::genai::Text2ImagePipeline& pipe,
                const std::string& prompt,
                const py::kwargs& kwargs
            ) -> py::typing::Union<ov::genai::ImageResults> {
                ov::AnyMap params = pyutils::kwargs_to_any_map(kwargs);
                return py::cast(pipe.generate(prompt, params));
            },
            py::arg("prompt"), "Input string",
            (text2image_generate_docstring + std::string(" \n ")).c_str())
        .def("decode", &ov::genai::Text2ImagePipeline::decode, py::arg("latent"));


    auto image2image_pipeline = py::class_<ov::genai::Image2ImagePipeline>(m, "Image2ImagePipeline", "This class is used for generation with image-to-image models.")
        .def(py::init([](const std::filesystem::path& models_path) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::Image2ImagePipeline>(models_path);
        }),
        py::arg("models_path"), "folder with exported model files.",
        R"(
            Image2ImagePipeline class constructor.
            models_path (os.PathLike): Path to the folder with exported model files.
        )")
        .def(py::init([](
            const std::filesystem::path& models_path,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::Image2ImagePipeline>(models_path, device, pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("models_path"), "folder with exported model files.",
        py::arg("device"), "device on which inference will be done",
        R"(
            Image2ImagePipeline class constructor.
            models_path (os.PathLike): Path with exported model files.
            device (str): Device to run the model on (e.g., CPU, GPU).
            kwargs: Image2ImagePipeline properties
        )")
        .def("get_generation_config", &ov::genai::Image2ImagePipeline::get_generation_config)
        .def("set_generation_config", &ov::genai::Image2ImagePipeline::set_generation_config, py::arg("generation_config"))
        .def("set_scheduler", &ov::genai::Image2ImagePipeline::set_scheduler, py::arg("scheduler"))
        .def("reshape", &ov::genai::Image2ImagePipeline::reshape, py::arg("num_images_per_prompt"), py::arg("height"), py::arg("width"), py::arg("guidance_scale"))
        .def_static("stable_diffusion", &ov::genai::Image2ImagePipeline::stable_diffusion, py::arg("scheduler"), py::arg("clip_text_model"), py::arg("unet"), py::arg("vae"))
        .def_static("latent_consistency_model", &ov::genai::Image2ImagePipeline::latent_consistency_model, py::arg("scheduler"), py::arg("clip_text_model"), py::arg("unet"), py::arg("vae"))
        .def_static("stable_diffusion_xl", &ov::genai::Image2ImagePipeline::stable_diffusion_xl, py::arg("scheduler"), py::arg("clip_text_model"), py::arg("clip_text_model_with_projection"), py::arg("unet"), py::arg("vae"))
        .def(
            "compile",
            [](ov::genai::Image2ImagePipeline& pipe,
                const std::string& device,
                const py::kwargs& kwargs
            ) {
                pipe.compile(device,  pyutils::kwargs_to_any_map(kwargs));
            },
            py::arg("device"), "device on which inference will be done",
            R"(
                Compiles the model.
                device (str): Device to run the model on (e.g., CPU, GPU).
                kwargs: Device properties.
            )")
        .def(
            "generate",
            [](ov::genai::Image2ImagePipeline& pipe,
                const std::string& prompt,
                const ov::Tensor& image,
                const py::kwargs& kwargs
            ) -> py::typing::Union<ov::genai::ImageResults> {
                ov::AnyMap params = pyutils::kwargs_to_any_map(kwargs);
                return py::cast(pipe.generate(prompt, image, params));
            },
            py::arg("prompt"), "Input string",
            py::arg("image"), "Initial image",
            (text2image_generate_docstring + std::string(" \n ")).c_str())
        .def("decode", &ov::genai::Image2ImagePipeline::decode, py::arg("latent"));


    auto inpainting_pipeline = py::class_<ov::genai::InpaintingPipeline>(m, "InpaintingPipeline", "This class is used for generation with inpainting models.")
        .def(py::init([](const std::filesystem::path& models_path) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::InpaintingPipeline>(models_path);
        }),
        py::arg("models_path"), "folder with exported model files.",
        R"(
            InpaintingPipeline class constructor.
            models_path (os.PathLike): Path to the folder with exported model files.
        )")
        .def(py::init([](
            const std::filesystem::path& models_path,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::InpaintingPipeline>(models_path, device, pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("models_path"), "folder with exported model files.",
        py::arg("device"), "device on which inference will be done",
        R"(
            InpaintingPipeline class constructor.
            models_path (os.PathLike): Path with exported model files.
            device (str): Device to run the model on (e.g., CPU, GPU).
            kwargs: InpaintingPipeline properties
        )")
        .def("get_generation_config", &ov::genai::InpaintingPipeline::get_generation_config)
        .def("set_generation_config", &ov::genai::InpaintingPipeline::set_generation_config, py::arg("generation_config"))
        .def("set_scheduler", &ov::genai::InpaintingPipeline::set_scheduler, py::arg("scheduler"))
        .def("reshape", &ov::genai::InpaintingPipeline::reshape, py::arg("num_images_per_prompt"), py::arg("height"), py::arg("width"), py::arg("guidance_scale"))
        .def_static("stable_diffusion", &ov::genai::InpaintingPipeline::stable_diffusion, py::arg("scheduler"), py::arg("clip_text_model"), py::arg("unet"), py::arg("vae"))
        .def_static("latent_consistency_model", &ov::genai::InpaintingPipeline::latent_consistency_model, py::arg("scheduler"), py::arg("clip_text_model"), py::arg("unet"), py::arg("vae"))
        .def_static("stable_diffusion_xl", &ov::genai::InpaintingPipeline::stable_diffusion_xl, py::arg("scheduler"), py::arg("clip_text_model"), py::arg("clip_text_model_with_projection"), py::arg("unet"), py::arg("vae"))
        .def(
            "compile",
            [](ov::genai::InpaintingPipeline& pipe,
                const std::string& device,
                const py::kwargs& kwargs
            ) {
                pipe.compile(device,  pyutils::kwargs_to_any_map(kwargs));
            },
            py::arg("device"), "device on which inference will be done",
            R"(
                Compiles the model.
                device (str): Device to run the model on (e.g., CPU, GPU).
                kwargs: Device properties.
            )")
        .def(
            "generate",
            [](ov::genai::InpaintingPipeline& pipe,
                const std::string& prompt,
                const ov::Tensor& image,
                const ov::Tensor& mask_image,
                const py::kwargs& kwargs
            ) -> py::typing::Union<ov::genai::ImageResults> {
                ov::AnyMap params = pyutils::kwargs_to_any_map(kwargs);
                return py::cast(pipe.generate(prompt, image, mask_image, params));
            },
            py::arg("prompt"), "Input string",
            py::arg("image"), "Initial image",
            py::arg("mask_image"), "Mask image",
            (text2image_generate_docstring + std::string(" \n ")).c_str())
        .def("decode", &ov::genai::InpaintingPipeline::decode, py::arg("latent"));

    // define constructors to create one pipeline from another
    // NOTE: needs to be defined once all pipelines are created

    text2image_pipeline
        .def(py::init([](const ov::genai::Image2ImagePipeline& pipe) {
            return std::make_unique<ov::genai::Text2ImagePipeline>(pipe);
        }), py::arg("pipe"))
        .def(py::init([](const ov::genai::InpaintingPipeline& pipe) {
            return std::make_unique<ov::genai::Text2ImagePipeline>(pipe);
        }), py::arg("pipe"));

    image2image_pipeline
        .def(py::init([](const ov::genai::InpaintingPipeline& pipe) {
            return std::make_unique<ov::genai::Image2ImagePipeline>(pipe);
        }), py::arg("pipe"));

    inpainting_pipeline
        .def(py::init([](const ov::genai::Image2ImagePipeline& pipe) {
            return std::make_unique<ov::genai::InpaintingPipeline>(pipe);
        }), py::arg("pipe"));
}
