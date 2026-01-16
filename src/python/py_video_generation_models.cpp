// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <filesystem>

#include "openvino/genai/video_generation/autoencoder_kl_ltx_video.hpp"
#include "openvino/genai/video_generation/ltx_video_transformer_3d_model.hpp"
#include "py_utils.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

void init_ltx_video_transformer_3d_model(py::module_& m) {
    auto ltx_transformer =
        py::class_<ov::genai::LTXVideoTransformer3DModel>(m,
                                                          "LTXVideoTransformer3DModel",
                                                          "LTXVideoTransformer3DModel class for LTX-Video denoising.")
            .def(py::init([](const std::filesystem::path& root_dir) {
                     return std::make_unique<ov::genai::LTXVideoTransformer3DModel>(root_dir);
                 }),
                 py::arg("root_dir"),
                 R"(
            LTXVideoTransformer3DModel class constructor.
            root_dir (os.PathLike): Model root directory.
        )")
            .def(py::init(
                     [](const std::filesystem::path& root_dir, const std::string& device, const py::kwargs& kwargs) {
                         return std::make_unique<ov::genai::LTXVideoTransformer3DModel>(
                             root_dir,
                             device,
                             pyutils::kwargs_to_any_map(kwargs));
                     }),
                 py::arg("root_dir"),
                 py::arg("device"),
                 R"(
            LTXVideoTransformer3DModel class constructor.
            root_dir (os.PathLike): Model root directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )")
            .def(py::init([](const ov::genai::LTXVideoTransformer3DModel& model) {
                     return std::make_unique<ov::genai::LTXVideoTransformer3DModel>(model);
                 }),
                 py::arg("model"),
                 R"(
            LTXVideoTransformer3DModel copy constructor.
            model (LTXVideoTransformer3DModel): Model to copy.
        )");

    py::class_<ov::genai::LTXVideoTransformer3DModel::Config>(ltx_transformer,
                                                              "Config",
                                                              "Configuration for LTXVideoTransformer3DModel.")
        .def(py::init([](const std::filesystem::path& config_path) {
                 return std::make_unique<ov::genai::LTXVideoTransformer3DModel::Config>(config_path);
             }),
             py::arg("config_path"))
        .def_readonly("in_channels", &ov::genai::LTXVideoTransformer3DModel::Config::in_channels)
        .def_readonly("patch_size", &ov::genai::LTXVideoTransformer3DModel::Config::patch_size)
        .def_readonly("patch_size_t", &ov::genai::LTXVideoTransformer3DModel::Config::patch_size_t);

    ltx_transformer.def("get_config", &ov::genai::LTXVideoTransformer3DModel::get_config)
        .def(
            "compile",
            [](ov::genai::LTXVideoTransformer3DModel& self, const std::string& device, const py::kwargs& kwargs) {
                auto properties = pyutils::kwargs_to_any_map(kwargs);
                py::gil_scoped_release rel;
                return self.compile(device, properties);
            },
            py::arg("device"),
            R"(
                Compiles the model.
                device (str): Device to run the model on (e.g., CPU, GPU).
                kwargs: Device properties.
            )")
        .def("reshape",
             &ov::genai::LTXVideoTransformer3DModel::reshape,
             py::arg("batch_size"),
             py::arg("num_frames"),
             py::arg("height"),
             py::arg("width"),
             py::arg("tokenizer_model_max_length"),
             R"(
                Reshapes the model for specific input dimensions.
                batch_size (int): Batch size.
                num_frames (int): Number of video frames.
                height (int): Video height.
                width (int): Video width.
                tokenizer_model_max_length (int): Maximum sequence length for tokenizer.
            )")
        .def("set_hidden_states",
             &ov::genai::LTXVideoTransformer3DModel::set_hidden_states,
             py::arg("tensor_name"),
             py::arg("encoder_hidden_states"),
             R"(
                Sets encoder hidden states tensor.
                tensor_name (str): Name of the tensor input.
                encoder_hidden_states (ov.Tensor): Hidden states from text encoder.
            )")
        .def("infer",
             &ov::genai::LTXVideoTransformer3DModel::infer,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("latent"),
             py::arg("timestep"),
             R"(
                Performs inference.
                latent (ov.Tensor): Latent video tensor.
                timestep (ov.Tensor): Current timestep tensor.
                Returns: Denoised latent tensor.
            )");
}

void init_autoencoder_kl_ltx_video(py::module_& m) {
    auto vae =
        py::class_<ov::genai::AutoencoderKLLTXVideo>(m,
                                                     "AutoencoderKLLTXVideo",
                                                     "AutoencoderKLLTXVideo class for LTX-Video VAE decoding.")
            .def(py::init([](const std::filesystem::path& vae_decoder_path) {
                     return std::make_unique<ov::genai::AutoencoderKLLTXVideo>(vae_decoder_path);
                 }),
                 py::arg("vae_decoder_path"),
                 R"(
            AutoencoderKLLTXVideo class constructor with decoder only.
            vae_decoder_path (os.PathLike): VAE decoder directory.
        )")
            .def(py::init(
                     [](const std::filesystem::path& vae_encoder_path, const std::filesystem::path& vae_decoder_path) {
                         return std::make_unique<ov::genai::AutoencoderKLLTXVideo>(vae_encoder_path, vae_decoder_path);
                     }),
                 py::arg("vae_encoder_path"),
                 py::arg("vae_decoder_path"),
                 R"(
            AutoencoderKLLTXVideo class constructor with encoder and decoder.
            vae_encoder_path (os.PathLike): VAE encoder directory.
            vae_decoder_path (os.PathLike): VAE decoder directory.
        )")
            .def(py::init([](const std::filesystem::path& vae_decoder_path,
                             const std::string& device,
                             const py::kwargs& kwargs) {
                     return std::make_unique<ov::genai::AutoencoderKLLTXVideo>(vae_decoder_path,
                                                                               device,
                                                                               pyutils::kwargs_to_any_map(kwargs));
                 }),
                 py::arg("vae_decoder_path"),
                 py::arg("device"),
                 R"(
            AutoencoderKLLTXVideo class constructor with decoder only.
            vae_decoder_path (os.PathLike): VAE decoder directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )")
            .def(py::init([](const std::filesystem::path& vae_encoder_path,
                             const std::filesystem::path& vae_decoder_path,
                             const std::string& device,
                             const py::kwargs& kwargs) {
                     return std::make_unique<ov::genai::AutoencoderKLLTXVideo>(vae_encoder_path,
                                                                               vae_decoder_path,
                                                                               device,
                                                                               pyutils::kwargs_to_any_map(kwargs));
                 }),
                 py::arg("vae_encoder_path"),
                 py::arg("vae_decoder_path"),
                 py::arg("device"),
                 R"(
            AutoencoderKLLTXVideo class constructor with encoder and decoder.
            vae_encoder_path (os.PathLike): VAE encoder directory.
            vae_decoder_path (os.PathLike): VAE decoder directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )");

    py::class_<ov::genai::AutoencoderKLLTXVideo::Config>(vae, "Config", "Configuration for AutoencoderKLLTXVideo.")
        .def(py::init([](const std::filesystem::path& config_path) {
                 return std::make_unique<ov::genai::AutoencoderKLLTXVideo::Config>(config_path);
             }),
             py::arg("config_path"))
        .def_readonly("in_channels", &ov::genai::AutoencoderKLLTXVideo::Config::in_channels)
        .def_readonly("latent_channels", &ov::genai::AutoencoderKLLTXVideo::Config::latent_channels)
        .def_readonly("out_channels", &ov::genai::AutoencoderKLLTXVideo::Config::out_channels)
        .def_readonly("scaling_factor", &ov::genai::AutoencoderKLLTXVideo::Config::scaling_factor)
        .def_readonly("block_out_channels", &ov::genai::AutoencoderKLLTXVideo::Config::block_out_channels)
        .def_readonly("patch_size", &ov::genai::AutoencoderKLLTXVideo::Config::patch_size)
        .def_readonly("patch_size_t", &ov::genai::AutoencoderKLLTXVideo::Config::patch_size_t);

    vae.def("get_config", &ov::genai::AutoencoderKLLTXVideo::get_config)
        .def("get_vae_scale_factor", &ov::genai::AutoencoderKLLTXVideo::get_vae_scale_factor)
        .def(
            "compile",
            [](ov::genai::AutoencoderKLLTXVideo& self, const std::string& device, const py::kwargs& kwargs) {
                auto properties = pyutils::kwargs_to_any_map(kwargs);
                py::gil_scoped_release rel;
                return self.compile(device, properties);
            },
            py::arg("device"),
            R"(
                Compiles the model.
                device (str): Device to run the model on (e.g., CPU, GPU).
                kwargs: Device properties.
            )")
        .def("reshape",
             &ov::genai::AutoencoderKLLTXVideo::reshape,
             py::arg("batch_size"),
             py::arg("num_frames"),
             py::arg("height"),
             py::arg("width"),
             R"(
                Reshapes the model for specific input dimensions.
                batch_size (int): Batch size.
                num_frames (int): Number of video frames.
                height (int): Video height.
                width (int): Video width.
            )")
        .def("decode",
             &ov::genai::AutoencoderKLLTXVideo::decode,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("latent"),
             R"(
                Decodes latent video to pixel space.
                latent (ov.Tensor): Latent video tensor.
                Returns: Decoded video tensor.
            )");
}

void init_video_generation_models(py::module_& m) {
    init_ltx_video_transformer_3d_model(m);
    init_autoencoder_kl_ltx_video(m);
}
