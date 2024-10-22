// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>

#include "openvino/genai/text2image/pipeline.hpp"

#include "tokenizers_path.hpp"
#include "py_utils.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

void init_clip_text_model(py::module_& m) {
    auto clip_text_model = py::class_<ov::genai::CLIPTextModel>(m, "CLIPTextModel", "CLIPTextModel class.")
        .def(py::init([](
            const std::filesystem::path& root_dir
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::CLIPTextModel>(root_dir);
        }),
        py::arg("root_dir"), "Model root directory", 
        R"(
            CLIPTextModel class
            root_dir (str): Model root directory.
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
            root_dir (str): Model root directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )") 
        .def(py::init([](
            const ov::genai::CLIPTextModel& model
        ) {
            return std::make_unique<ov::genai::CLIPTextModel>(model);
        }),
        py::arg("model"), "CLIPText model"
        R"(
            CLIPTextModel class
            model (CLIPTextModel): CLIPText model
        )");
    
    py::class_<ov::genai::CLIPTextModel::Config>(clip_text_model, "Config", "This class is used for storing CLIPTextModel config.")
        .def(py::init([](
            const std::string& config_path
        ) {
            return std::make_unique<ov::genai::CLIPTextModel::Config>(config_path);
        }))
        .def_readwrite("max_position_embeddings", &ov::genai::CLIPTextModel::Config::max_position_embeddings)
        .def_readwrite("hidden_size", &ov::genai::CLIPTextModel::Config::hidden_size)
        .def_readwrite("num_hidden_layers", &ov::genai::CLIPTextModel::Config::num_hidden_layers);

    clip_text_model.def("get_config", &ov::genai::CLIPTextModel::get_config);
    clip_text_model.def("reshape", &ov::genai::CLIPTextModel::reshape);
    clip_text_model.def("set_adapters", &ov::genai::CLIPTextModel::set_adapters);
    clip_text_model.def("infer", &ov::genai::CLIPTextModel::infer);
    clip_text_model.def("get_output_tensor", &ov::genai::CLIPTextModel::get_output_tensor);
    clip_text_model.def(
            "compile", 
            [](ov::genai::CLIPTextModel& self, 
                const std::string& device,
                const py::kwargs& kwargs
            ) {
                self.compile(device,  pyutils::kwargs_to_any_map(kwargs));
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
        .def(py::init([](
            const std::filesystem::path& root_dir
        ) {
            return std::make_unique<ov::genai::UNet2DConditionModel>(root_dir);
        }),
        py::arg("root_dir"), "Model root directory", 
        R"(
            UNet2DConditionModel class
            root_dir (str): Model root directory.
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
            root_dir (str): Model root directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )") 
        .def(py::init([](
            const ov::genai::UNet2DConditionModel& model
        ) {
            return std::make_unique<ov::genai::UNet2DConditionModel>(model);
        }),
        py::arg("model"), "UNet2DConditionModel model"
        R"(
            UNet2DConditionModel class
            model (UNet2DConditionModel): UNet2DConditionModel model
        )");

    py::class_<ov::genai::UNet2DConditionModel::Config>(unet2d_condition_model, "Config", "This class is used for storing UNet2DConditionModel config.")
        .def(py::init([](
            const std::filesystem::path& config_path
        ) {
            return std::make_unique<ov::genai::UNet2DConditionModel::Config>(config_path);
        }))
        .def_readwrite("in_channels", &ov::genai::UNet2DConditionModel::Config::in_channels)
        .def_readwrite("sample_size", &ov::genai::UNet2DConditionModel::Config::sample_size)
        .def_readwrite("block_out_channels", &ov::genai::UNet2DConditionModel::Config::block_out_channels)
        .def_readwrite("time_cond_proj_dim", &ov::genai::UNet2DConditionModel::Config::time_cond_proj_dim);

    unet2d_condition_model.def("get_config", &ov::genai::UNet2DConditionModel::get_config);
    unet2d_condition_model.def("reshape", &ov::genai::UNet2DConditionModel::reshape);
    unet2d_condition_model.def("set_adapters", &ov::genai::UNet2DConditionModel::set_adapters);
    unet2d_condition_model.def("infer", &ov::genai::UNet2DConditionModel::infer);
    unet2d_condition_model.def("get_vae_scale_factor", &ov::genai::UNet2DConditionModel::get_vae_scale_factor);
    unet2d_condition_model.def("set_hidden_states", &ov::genai::UNet2DConditionModel::set_hidden_states);
    unet2d_condition_model.def(
            "compile", 
            [](ov::genai::UNet2DConditionModel& self,
                const std::string& device,
                const py::kwargs& kwargs
            ) {
                self.compile(device,  pyutils::kwargs_to_any_map(kwargs));
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
        .def(py::init([](
            const std::filesystem::path& root_dir
        ) {
            return std::make_unique<ov::genai::AutoencoderKL>(root_dir);
        }),
        py::arg("root_dir"), "Root directory", 
        R"(
            AutoencoderKL class
            root_dir (str): Root directory.
        )")
        .def(py::init([](
            const std::filesystem::path& root_dir,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            return std::make_unique<ov::genai::AutoencoderKL>(root_dir, device,  pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("root_dir"), "Root directory", 
        py::arg("device"), "Device on which inference will be done",
        R"(
            AutoencoderKL class
            root_dir (str): Root directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )") 
        .def(py::init([](
            const ov::genai::AutoencoderKL& model
        ) {
            return std::make_unique<ov::genai::AutoencoderKL>(model);
        }),
        py::arg("model"), "AutoencoderKL model"
        R"(
            AutoencoderKL class
            model (AutoencoderKL): AutoencoderKL model
        )");

    py::class_<ov::genai::AutoencoderKL::Config>(autoencoder_kl, "Config", "This class is used for storing AutoencoderKL config.")
        .def(py::init([](
            const std::filesystem::path& config_path
        ) {
            return std::make_unique<ov::genai::AutoencoderKL::Config>(config_path);
        }))
        .def_readwrite("in_channels", &ov::genai::AutoencoderKL::Config::in_channels)
        .def_readwrite("latent_channels", &ov::genai::AutoencoderKL::Config::latent_channels)
        .def_readwrite("out_channels", &ov::genai::AutoencoderKL::Config::out_channels)
        .def_readwrite("scaling_factor", &ov::genai::AutoencoderKL::Config::scaling_factor)
        .def_readwrite("block_out_channels", &ov::genai::AutoencoderKL::Config::block_out_channels);

    autoencoder_kl.def("reshape", &ov::genai::AutoencoderKL::reshape);
    autoencoder_kl.def("decode", &ov::genai::AutoencoderKL::decode);
    autoencoder_kl.def(
            "compile", 
            [](ov::genai::AutoencoderKL& self,
                const std::string& device,
                const py::kwargs& kwargs
            ) {
                self.compile(device,  pyutils::kwargs_to_any_map(kwargs));
            },
            py::arg("device"), "device on which inference will be done"
            R"(
                Compiles the model.
                device (str): Device to run the model on (e.g., CPU, GPU).
                kwargs: Device properties.
            )");
}

void init_clip_text_model_with_projection(py::module_& m) {
    auto clip_text_model_with_projection = py::class_<ov::genai::CLIPTextModelWithProjection>(m, "CLIPTextModelWithProjection", "CLIPTextModelWithProjection class.")
        .def(py::init([](
            const std::filesystem::path& root_dir
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::CLIPTextModelWithProjection>(root_dir);
        }),
        py::arg("root_dir"), "Model root directory", 
        R"(
            CLIPTextModelWithProjection class
            root_dir (str): Model root directory.
        )")
        .def(py::init([](
            const std::filesystem::path& root_dir,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::CLIPTextModelWithProjection>(root_dir, device,  pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("root_dir"), "Model root directory", 
        py::arg("device"), "Device on which inference will be done",
        R"(
            CLIPTextModelWithProjection class
            root_dir (str): Model root directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )") 
        .def(py::init([](
            const ov::genai::CLIPTextModelWithProjection& model
        ) {
            return std::make_unique<ov::genai::CLIPTextModelWithProjection>(model);
        }),
        py::arg("model"), "CLIPTextModelWithProjection model"
        R"(
            CLIPTextModelWithProjection class
            model (CLIPTextModelWithProjection): CLIPTextModelWithProjection model
        )");

    py::class_<ov::genai::CLIPTextModelWithProjection::Config>(clip_text_model_with_projection, "Config", "This class is used for storing CLIPTextModelWithProjection config.")
        .def(py::init([](
            const std::filesystem::path& config_path
        ) {
            return std::make_unique<ov::genai::CLIPTextModelWithProjection::Config>(config_path);
        }))
        .def_readwrite("max_position_embeddings", &ov::genai::CLIPTextModelWithProjection::Config::max_position_embeddings)
        .def_readwrite("hidden_size", &ov::genai::CLIPTextModelWithProjection::Config::hidden_size)
        .def_readwrite("num_hidden_layers", &ov::genai::CLIPTextModelWithProjection::Config::num_hidden_layers);

    clip_text_model_with_projection.def("reshape", &ov::genai::CLIPTextModelWithProjection::reshape);
    clip_text_model_with_projection.def("infer", &ov::genai::CLIPTextModelWithProjection::infer);
    clip_text_model_with_projection.def("get_config", &ov::genai::CLIPTextModelWithProjection::get_config);
    clip_text_model_with_projection.def("get_output_tensor", &ov::genai::CLIPTextModelWithProjection::get_config);
    clip_text_model_with_projection.def(
            "compile", 
            [](ov::genai::CLIPTextModelWithProjection& self, 
                const std::string& device,
                const py::kwargs& kwargs
            ) {
                self.compile(device,  pyutils::kwargs_to_any_map(kwargs));
            },
            py::arg("device"), "device on which inference will be done",
            R"(
                Compiles the model.
                device (str): Device to run the model on (e.g., CPU, GPU).
                kwargs: Device properties.
            )");
}
