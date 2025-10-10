// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>

#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "openvino/genai/image_generation/clip_text_model_with_projection.hpp"
#include "openvino/genai/image_generation/t5_encoder_model.hpp"
#include "openvino/genai/image_generation/unet2d_condition_model.hpp"
#include "openvino/genai/image_generation/sd3_transformer_2d_model.hpp"
#include "openvino/genai/image_generation/flux_transformer_2d_model.hpp"

#include "tokenizer/tokenizers_path.hpp"
#include "py_utils.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

void init_clip_text_model(py::module_& m) {
    auto clip_text_model = py::class_<ov::genai::CLIPTextModel>(m, "CLIPTextModel", "CLIPTextModel class.")
        .def(py::init([](const std::filesystem::path& root_dir) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::CLIPTextModel>(root_dir);
        }),
        py::arg("root_dir"), "Model root directory",
        R"(
            CLIPTextModel class
            root_dir (os.PathLike): Model root directory.
        )")
        .def(py::init([](
            const std::filesystem::path& root_dir,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::CLIPTextModel>(root_dir, device, pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("root_dir"), "Model root directory",
        py::arg("device"), "Device on which inference will be done",
        R"(
            CLIPTextModel class
            root_dir (os.PathLike): Model root directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )")
        .def(py::init([](const ov::genai::CLIPTextModel& model) {
            return std::make_unique<ov::genai::CLIPTextModel>(model);
        }),
        py::arg("model"), "CLIPText model"
        R"(
            CLIPTextModel class
            model (CLIPTextModel): CLIPText model
        )");

    py::class_<ov::genai::CLIPTextModel::Config>(clip_text_model, "Config", "This class is used for storing CLIPTextModel config.")
        .def(py::init([](const std::filesystem::path& config_path) {
            return std::make_unique<ov::genai::CLIPTextModel::Config>(config_path);
        }),
        py::arg("config_path"))
        .def_readwrite("max_position_embeddings", &ov::genai::CLIPTextModel::Config::max_position_embeddings)
        .def_readwrite("num_hidden_layers", &ov::genai::CLIPTextModel::Config::num_hidden_layers);

    clip_text_model.def("get_config", &ov::genai::CLIPTextModel::get_config)
        .def("reshape", &ov::genai::CLIPTextModel::reshape, py::arg("batch_size"))
        .def("set_adapters", &ov::genai::CLIPTextModel::set_adapters, py::arg("adapters"))
        .def("infer", 
            &ov::genai::CLIPTextModel::infer, 
            py::call_guard<py::gil_scoped_release>(), 
            py::arg("pos_prompt"), 
            py::arg("neg_prompt"), 
            py::arg("do_classifier_free_guidance"))
        .def("get_output_tensor", &ov::genai::CLIPTextModel::get_output_tensor, py::arg("idx"))
        .def(
            "compile",
            [](ov::genai::CLIPTextModel& self,
                const std::string& device,
                const py::kwargs& kwargs
            ) {
                auto map = pyutils::kwargs_to_any_map(kwargs);
                {
                    py::gil_scoped_release rel;
                    self.compile(device,  map);
                }
            },
            py::arg("device"), "device on which inference will be done",
            R"(
                Compiles the model.
                device (str): Device to run the model on (e.g., CPU, GPU).
                kwargs: Device properties.
            )")
        .def(
            "export_model",
             &ov::genai::CLIPTextModel::export_model,
             py::arg("export_path"),
             R"(
                Exports compiled model to a specified directory. Can significantly reduce model load time, especially for large models.
                export_path (os.PathLike): A path to a directory to export compiled model to.

                Use `blob_path` property to load previously exported models.
            )");
}

void init_clip_text_model_with_projection(py::module_& m) {
    py::class_<ov::genai::CLIPTextModelWithProjection, ov::genai::CLIPTextModel>(m, "CLIPTextModelWithProjection", "CLIPTextModelWithProjection class.")
        .def(py::init([](const std::filesystem::path& root_dir) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::CLIPTextModelWithProjection>(root_dir);
        }),
        py::arg("root_dir"), "Model root directory",
        R"(
            CLIPTextModelWithProjection class
            root_dir (os.PathLike): Model root directory.
        )")
        .def(py::init([](
            const std::filesystem::path& root_dir,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::CLIPTextModelWithProjection>(root_dir, device, pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("root_dir"), "Model root directory",
        py::arg("device"), "Device on which inference will be done",
        R"(
            CLIPTextModelWithProjection class
            root_dir (os.PathLike): Model root directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )")
        .def(py::init([](const ov::genai::CLIPTextModelWithProjection& model) {
            return std::make_unique<ov::genai::CLIPTextModelWithProjection>(model);
        }),
        py::arg("model"), "CLIPText model"
        R"(
            CLIPTextModelWithProjection class
            model (CLIPTextModelWithProjection): CLIPText model with projection
        )");
}

void init_t5_encoder_model(py::module_& m) {
    auto t5_encoder_model = py::class_<ov::genai::T5EncoderModel>(m, "T5EncoderModel", "T5EncoderModel class.")
        .def(py::init([](const std::filesystem::path& root_dir) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::T5EncoderModel>(root_dir);
        }),
        py::arg("root_dir"), "Model root directory",
        R"(
            T5EncoderModel class
            root_dir (os.PathLike): Model root directory.
        )")
        .def(py::init([](
            const std::filesystem::path& root_dir,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::T5EncoderModel>(root_dir, device,  pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("root_dir"), "Model root directory",
        py::arg("device"), "Device on which inference will be done",
        R"(
            T5EncoderModel class
            root_dir (os.PathLike): Model root directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )")
        .def(py::init([](const ov::genai::T5EncoderModel& model) {
            return std::make_unique<ov::genai::T5EncoderModel>(model);
        }),
        py::arg("model"), "T5EncoderModel model"
        R"(
            T5EncoderModel class
            model (T5EncoderModel): T5EncoderModel model
        )")
        .def("reshape", &ov::genai::T5EncoderModel::reshape, py::arg("batch_size"), py::arg("max_sequence_length"))
        .def("infer", 
            &ov::genai::T5EncoderModel::infer, 
            py::call_guard<py::gil_scoped_release>(), 
            py::arg("pos_prompt"), 
            py::arg("neg_prompt"), 
            py::arg("do_classifier_free_guidance"), 
            py::arg("max_sequence_length"))
        .def("get_output_tensor", &ov::genai::T5EncoderModel::get_output_tensor, py::arg("idx"))
        .def(
            "compile",
            [](ov::genai::T5EncoderModel& self,
                const std::string& device,
                const py::kwargs& kwargs
            ) {
                auto map = pyutils::kwargs_to_any_map(kwargs);
                {
                    py::gil_scoped_release rel;
                    self.compile(device, map);
                }
            },
            py::arg("device"), "device on which inference will be done",
            R"(
                Compiles the model.
                device (str): Device to run the model on (e.g., CPU, GPU).
                kwargs: Device properties.
            )");
}

void init_unet2d_condition_model(py::module_& m) {
    auto unet2d_condition_model = py::class_<ov::genai::UNet2DConditionModel>(m, "UNet2DConditionModel", "UNet2DConditionModel class.")
        .def(py::init([](const std::filesystem::path& root_dir) {
            return std::make_unique<ov::genai::UNet2DConditionModel>(root_dir);
        }),
        py::arg("root_dir"), "Model root directory",
        R"(
            UNet2DConditionModel class
            root_dir (os.PathLike): Model root directory.
        )")
        .def(py::init([](
            const std::filesystem::path& root_dir,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            return std::make_unique<ov::genai::UNet2DConditionModel>(root_dir, device,  pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("root_dir"), "Model root directory",
        py::arg("device"), "Device on which inference will be done",
        R"(
            UNet2DConditionModel class
            root_dir (os.PathLike): Model root directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )")
        .def(py::init([](const ov::genai::UNet2DConditionModel& model) {
            return std::make_unique<ov::genai::UNet2DConditionModel>(model);
        }),
        py::arg("model"), "UNet2DConditionModel model"
        R"(
            UNet2DConditionModel class
            model (UNet2DConditionModel): UNet2DConditionModel model
        )");

    py::class_<ov::genai::UNet2DConditionModel::Config>(unet2d_condition_model, "Config", "This class is used for storing UNet2DConditionModel config.")
        .def(py::init([](const std::filesystem::path& config_path) {
            return std::make_unique<ov::genai::UNet2DConditionModel::Config>(config_path);
        }),
        py::arg("config_path"))
        .def_readwrite("in_channels", &ov::genai::UNet2DConditionModel::Config::in_channels)
        .def_readwrite("sample_size", &ov::genai::UNet2DConditionModel::Config::sample_size)
        .def_readwrite("time_cond_proj_dim", &ov::genai::UNet2DConditionModel::Config::time_cond_proj_dim);

    unet2d_condition_model.def("get_config", &ov::genai::UNet2DConditionModel::get_config)
        .def("reshape", &ov::genai::UNet2DConditionModel::reshape, py::arg("batch_size"), py::arg("height"), py::arg("width"), py::arg("tokenizer_model_max_length"))
        .def("set_adapters", &ov::genai::UNet2DConditionModel::set_adapters, py::arg("adapters"))
        .def("infer", 
            &ov::genai::UNet2DConditionModel::infer, 
            py::call_guard<py::gil_scoped_release>(),
            py::arg("sample"), 
            py::arg("timestep"))
        .def("set_hidden_states", &ov::genai::UNet2DConditionModel::set_hidden_states, py::arg("tensor_name"), py::arg("encoder_hidden_states"))
        .def("do_classifier_free_guidance", &ov::genai::UNet2DConditionModel::do_classifier_free_guidance, py::arg("guidance_scale"))
        .def(
            "compile",
            [](ov::genai::UNet2DConditionModel& self,
                const std::string& device,
                const py::kwargs& kwargs
            ) {
                auto map = pyutils::kwargs_to_any_map(kwargs);
                {
                    py::gil_scoped_release rel;
                    self.compile(device,  map);
                }
            },
            py::arg("device"), "device on which inference will be done",
            R"(
                Compiles the model.
                device (str): Device to run the model on (e.g., CPU, GPU).
                kwargs: Device properties.
            )")
        .def("export_model", &ov::genai::UNet2DConditionModel::export_model, py::arg("export_path"), R"(
                Exports compiled model to a specified directory. Can significantly reduce model load time, especially for large models.
                export_path (os.PathLike): A path to a directory to export compiled model to.

                Use `blob_path` property to load previously exported models.
            )");
}

void init_sd3_transformer_2d_model(py::module_& m) {
    auto sd3_transformer_2d_model = py::class_<ov::genai::SD3Transformer2DModel>(m, "SD3Transformer2DModel", "SD3Transformer2DModel class.")
        .def(py::init([](const std::filesystem::path& root_dir) {
            return std::make_unique<ov::genai::SD3Transformer2DModel>(root_dir);
        }),
        py::arg("root_dir"), "Model root directory",
        R"(
            SD3Transformer2DModel class
            root_dir (os.PathLike): Model root directory.
        )")
        .def(py::init([](
            const std::filesystem::path& root_dir,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            return std::make_unique<ov::genai::SD3Transformer2DModel>(root_dir, device,  pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("root_dir"), "Model root directory",
        py::arg("device"), "Device on which inference will be done",
        R"(
            SD3Transformer2DModel class
            root_dir (os.PathLike): Model root directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )")
        .def(py::init([](const ov::genai::SD3Transformer2DModel& model) {
            return std::make_unique<ov::genai::SD3Transformer2DModel>(model);
        }),
        py::arg("model"), "SD3Transformer2DModel model"
        R"(
            SD3Transformer2DModel class
            model (SD3Transformer2DModel): SD3Transformer2DModel model
        )");

    py::class_<ov::genai::SD3Transformer2DModel::Config>(sd3_transformer_2d_model, "Config", "This class is used for storing SD3Transformer2DModel config.")
        .def(py::init([](const std::filesystem::path& config_path) {
            return std::make_unique<ov::genai::SD3Transformer2DModel::Config>(config_path);
        }),
        py::arg("config_path"))
        .def_readwrite("in_channels", &ov::genai::SD3Transformer2DModel::Config::in_channels)
        .def_readwrite("sample_size", &ov::genai::SD3Transformer2DModel::Config::sample_size)
        .def_readwrite("patch_size", &ov::genai::SD3Transformer2DModel::Config::patch_size)
        .def_readwrite("joint_attention_dim", &ov::genai::SD3Transformer2DModel::Config::joint_attention_dim);

    sd3_transformer_2d_model.def("get_config", &ov::genai::SD3Transformer2DModel::get_config)
        .def("reshape", &ov::genai::SD3Transformer2DModel::reshape, py::arg("batch_size"), py::arg("height"), py::arg("width"), py::arg("tokenizer_model_max_length"))
        .def("infer", 
            &ov::genai::SD3Transformer2DModel::infer, 
            py::call_guard<py::gil_scoped_release>(),
            py::arg("latent"), 
            py::arg("timestep"))
        .def("set_hidden_states", &ov::genai::SD3Transformer2DModel::set_hidden_states, py::arg("tensor_name"), py::arg("encoder_hidden_states"))
        .def(
            "compile",
            [](ov::genai::SD3Transformer2DModel& self,
                const std::string& device,
                const py::kwargs& kwargs
            ) {
                auto map = pyutils::kwargs_to_any_map(kwargs);
                {
                    py::gil_scoped_release rel;
                    self.compile(device,  map);
                }
            },
            py::arg("device"), "device on which inference will be done",
            R"(
                Compiles the model.
                device (str): Device to run the model on (e.g., CPU, GPU).
                kwargs: Device properties.
            )");
}

void init_flux_transformer_2d_model(py::module_& m) {
    auto flux_transformer_2d_model = py::class_<ov::genai::FluxTransformer2DModel>(m, "FluxTransformer2DModel", "FluxTransformer2DModel class.")
        .def(py::init([](const std::filesystem::path& root_dir) {
            return std::make_unique<ov::genai::FluxTransformer2DModel>(root_dir);
        }),
        py::arg("root_dir"), "Model root directory",
        R"(
            FluxTransformer2DModel class
            root_dir (os.PathLike): Model root directory.
        )")
        .def(py::init([](
            const std::filesystem::path& root_dir,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            return std::make_unique<ov::genai::FluxTransformer2DModel>(root_dir, device,  pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("root_dir"), "Model root directory",
        py::arg("device"), "Device on which inference will be done",
        R"(
            UNet2DConditionModel class
            root_dir (os.PathLike): Model root directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )")
        .def(py::init([](const ov::genai::FluxTransformer2DModel& model) {
            return std::make_unique<ov::genai::FluxTransformer2DModel>(model);
        }),
        py::arg("model"), "FluxTransformer2DModel model"
        R"(
            FluxTransformer2DModel class
            model (FluxTransformer2DModel): FluxTransformer2DModel model
        )");

    py::class_<ov::genai::FluxTransformer2DModel::Config>(flux_transformer_2d_model, "Config", "This class is used for storing FluxTransformer2DModel config.")
        .def(py::init([](const std::filesystem::path& config_path) {
            return std::make_unique<ov::genai::FluxTransformer2DModel::Config>(config_path);
        }),
        py::arg("config_path"))
        .def_readwrite("in_channels", &ov::genai::FluxTransformer2DModel::Config::in_channels)
        .def_readwrite("default_sample_size", &ov::genai::FluxTransformer2DModel::Config::m_default_sample_size);

    flux_transformer_2d_model.def("get_config", &ov::genai::FluxTransformer2DModel::get_config)
        .def("reshape", &ov::genai::FluxTransformer2DModel::reshape, py::arg("batch_size"), py::arg("height"), py::arg("width"), py::arg("tokenizer_model_max_length"))
        .def("infer", 
            &ov::genai::FluxTransformer2DModel::infer, 
            py::call_guard<py::gil_scoped_release>(), 
            py::arg("latent"), 
            py::arg("timestep"))
        .def("set_hidden_states", &ov::genai::FluxTransformer2DModel::set_hidden_states, py::arg("tensor_name"), py::arg("encoder_hidden_states"))
        .def(
            "compile",
            [](ov::genai::FluxTransformer2DModel& self,
               const std::string& device,
               const py::kwargs& kwargs
            ) {
                auto map = pyutils::kwargs_to_any_map(kwargs);
                {
                    py::gil_scoped_release rel;
                    self.compile(device,  map);
                }
            },
            py::arg("device"), "device on which inference will be done",
            R"(
                Compiles the model.
                device (str): Device to run the model on (e.g., CPU, GPU).
                kwargs: Device properties.
            )");
}

void init_autoencoder_kl(py::module_& m) {
    auto autoencoder_kl = py::class_<ov::genai::AutoencoderKL>(m, "AutoencoderKL", "AutoencoderKL class.")
        .def(py::init([](const std::filesystem::path& vae_decoder_path) {
            return std::make_unique<ov::genai::AutoencoderKL>(vae_decoder_path);
        }),
        py::arg("vae_decoder_path"), "VAE decoder directory",
        R"(
            AutoencoderKL class initialized only with decoder model.
            vae_decoder_path (os.PathLike): VAE decoder directory.
        )")
        .def(py::init([](
            const std::filesystem::path& vae_encoder_path,
            const std::filesystem::path& vae_decoder_path
        ) {
            return std::make_unique<ov::genai::AutoencoderKL>(vae_encoder_path, vae_decoder_path);
        }),
        py::arg("vae_encoder_path"), "VAE encoder directory",
        py::arg("vae_decoder_path"), "VAE decoder directory",
        R"(
            AutoencoderKL class initialized with both encoder and decoder models.
            vae_encoder_path (os.PathLike): VAE encoder directory.
            vae_decoder_path (os.PathLike): VAE decoder directory.
        )")
        .def(py::init([](
            const std::filesystem::path& vae_decoder_path,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            return std::make_unique<ov::genai::AutoencoderKL>(vae_decoder_path, device, pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("vae_decoder_path"), "Root directory",
        py::arg("device"), "Device on which inference will be done",
        R"(
            AutoencoderKL class initialized only with decoder model.
            vae_decoder_path (os.PathLike): VAE decoder directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )")
        .def(py::init([](
            const std::filesystem::path& vae_encoder_path,
            const std::filesystem::path& vae_decoder_path,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            return std::make_unique<ov::genai::AutoencoderKL>(vae_encoder_path, vae_decoder_path, device, pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("vae_encoder_path"), "VAE encoder directory",
        py::arg("vae_decoder_path"), "VAE decoder directory",
        py::arg("device"), "Device on which inference will be done",
        R"(
            AutoencoderKL class initialized only with both encoder and decoder models.
            vae_encoder_path (os.PathLike): VAE encoder directory.
            vae_decoder_path (os.PathLike): VAE decoder directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )")
        .def(py::init([](const ov::genai::AutoencoderKL& model) {
            return std::make_unique<ov::genai::AutoencoderKL>(model);
        }),
        py::arg("model"), "AutoencoderKL model"
        R"(
            AutoencoderKL class.
            model (AutoencoderKL): AutoencoderKL model.
        )");

    py::class_<ov::genai::AutoencoderKL::Config>(autoencoder_kl, "Config", "This class is used for storing AutoencoderKL config.")
        .def(py::init([](const std::filesystem::path& config_path) {
            return std::make_unique<ov::genai::AutoencoderKL::Config>(config_path);
        }),
        py::arg("config_path"))
        .def_readwrite("in_channels", &ov::genai::AutoencoderKL::Config::in_channels)
        .def_readwrite("latent_channels", &ov::genai::AutoencoderKL::Config::latent_channels)
        .def_readwrite("out_channels", &ov::genai::AutoencoderKL::Config::out_channels)
        .def_readwrite("scaling_factor", &ov::genai::AutoencoderKL::Config::scaling_factor)
        .def_readwrite("block_out_channels", &ov::genai::AutoencoderKL::Config::block_out_channels);

    autoencoder_kl.def("reshape", &ov::genai::AutoencoderKL::reshape, py::arg("batch_size"), py::arg("height"), py::arg("width"))
        .def(
            "compile",
            [](ov::genai::AutoencoderKL& self,
                const std::string& device,
                const py::kwargs& kwargs
            ) {
                auto map = pyutils::kwargs_to_any_map(kwargs);
                {
                    py::gil_scoped_release rel;
                    self.compile(device,  map);
                }
            },
            py::arg("device"), "device on which inference will be done"
            R"(
                Compiles the model.
                device (str): Device to run the model on (e.g., CPU, GPU).
                kwargs: Device properties.
            )")
        .def("decode", &ov::genai::AutoencoderKL::decode, py::call_guard<py::gil_scoped_release>(), py::arg("latent"))
        .def("encode", &ov::genai::AutoencoderKL::encode, py::call_guard<py::gil_scoped_release>(), py::arg("image"), py::arg("generator"))
        .def("get_config", &ov::genai::AutoencoderKL::get_config)
        .def("get_vae_scale_factor", &ov::genai::AutoencoderKL::get_vae_scale_factor)
        .def("export_model",
            &ov::genai::AutoencoderKL::export_model,
            py::arg("export_path"),
            R"(
                Exports compiled models to a specified directory. Can significantly reduce model load time, especially for large models.
                export_path (os.PathLike): A path to a directory to export compiled models to.

                Use `blob_path` property to load previously exported models.
            )"
        );
}
