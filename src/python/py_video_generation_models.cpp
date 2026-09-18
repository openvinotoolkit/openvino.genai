// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <filesystem>

#include "openvino/genai/video_generation/autoencoder_kl_ltx2_audio.hpp"
#include "openvino/genai/video_generation/autoencoder_kl_ltx2_video.hpp"
#include "openvino/genai/video_generation/autoencoder_kl_ltx_video.hpp"
#include "openvino/genai/video_generation/ltx2_text_connectors.hpp"
#include "openvino/genai/video_generation/ltx2_video_transformer_3d_model.hpp"
#include "openvino/genai/video_generation/ltx2_vocoder.hpp"
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
        .def("set_adapters",
             &ov::genai::LTXVideoTransformer3DModel::set_adapters,
             py::arg("adapters"),
             R"(
                Sets LoRA adapters for the transformer model.
                adapters (AdapterConfig or None): Adapter configuration to apply.
                Passing None keeps currently configured adapters unchanged.
                Pass an empty AdapterConfig() to disable all adapters.
            )")
        .def("infer",
             py::overload_cast<const ov::Tensor&, const ov::Tensor&>(&ov::genai::LTXVideoTransformer3DModel::infer),
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
        .def("encode",
             [](ov::genai::AutoencoderKLLTXVideo& self,
                const ov::Tensor& video,
                std::shared_ptr<ov::genai::Generator> generator) {
                 py::gil_scoped_release release;
                 return self.encode(video, generator);
             },
             py::arg("video"),
             py::arg("generator") = py::none(),
             R"(
                Encodes a video tensor to latent space.
                video (ov.Tensor): Input video tensor [B, C, F, H, W].
                generator (Generator, optional): Random generator for sampling from the latent
                    distribution. Required only when the encoder outputs latent parameters
                    (mean + logvar); unused when it outputs a latent sample directly.
                Returns: Normalized latent tensor.
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

void init_ltx2_video_transformer_3d_model(py::module_& m) {
    auto ltx2_transformer =
        py::class_<ov::genai::LTX2VideoTransformer3DModel>(m,
                                                           "LTX2VideoTransformer3DModel",
                                                           "LTX2VideoTransformer3DModel class for joint LTX2 video and audio denoising.")
            .def(py::init([](const std::filesystem::path& root_dir) {
                     return std::make_unique<ov::genai::LTX2VideoTransformer3DModel>(root_dir);
                 }),
                 py::arg("root_dir"),
                 R"(
            LTX2VideoTransformer3DModel class constructor.
            root_dir (os.PathLike): Model root directory.
        )")
            .def(py::init(
                     [](const std::filesystem::path& root_dir, const std::string& device, const py::kwargs& kwargs) {
                         return std::make_unique<ov::genai::LTX2VideoTransformer3DModel>(
                             root_dir,
                             device,
                             pyutils::kwargs_to_any_map(kwargs));
                     }),
                 py::arg("root_dir"),
                 py::arg("device"),
                 R"(
            LTX2VideoTransformer3DModel class constructor.
            root_dir (os.PathLike): Model root directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )")
            .def(py::init([](const ov::genai::LTX2VideoTransformer3DModel& model) {
                     return std::make_unique<ov::genai::LTX2VideoTransformer3DModel>(model);
                 }),
                 py::arg("model"),
                 R"(
            LTX2VideoTransformer3DModel copy constructor.
            model (LTX2VideoTransformer3DModel): Model to copy.
        )");

    py::class_<ov::genai::LTX2VideoTransformer3DModel::Config>(ltx2_transformer,
                                                               "Config",
                                                               "Configuration for LTX2VideoTransformer3DModel.")
        .def(py::init([](const std::filesystem::path& config_path) {
                 return std::make_unique<ov::genai::LTX2VideoTransformer3DModel::Config>(config_path);
             }),
             py::arg("config_path"))
        .def_readonly("in_channels", &ov::genai::LTX2VideoTransformer3DModel::Config::in_channels)
        .def_readonly("audio_in_channels", &ov::genai::LTX2VideoTransformer3DModel::Config::audio_in_channels)
        .def_readonly("patch_size", &ov::genai::LTX2VideoTransformer3DModel::Config::patch_size)
        .def_readonly("patch_size_t", &ov::genai::LTX2VideoTransformer3DModel::Config::patch_size_t)
        .def_readonly("vae_scale_factors", &ov::genai::LTX2VideoTransformer3DModel::Config::vae_scale_factors)
        .def_readonly("audio_scale_factor", &ov::genai::LTX2VideoTransformer3DModel::Config::audio_scale_factor)
        .def_readonly("audio_sampling_rate", &ov::genai::LTX2VideoTransformer3DModel::Config::audio_sampling_rate)
        .def_readonly("audio_hop_length", &ov::genai::LTX2VideoTransformer3DModel::Config::audio_hop_length);

    ltx2_transformer.def("get_config", &ov::genai::LTX2VideoTransformer3DModel::get_config)
        .def(
            "compile",
            [](ov::genai::LTX2VideoTransformer3DModel& self, const std::string& device, const py::kwargs& kwargs) {
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
             &ov::genai::LTX2VideoTransformer3DModel::reshape,
             py::arg("batch_size"),
             py::arg("num_frames"),
             py::arg("height"),
             py::arg("width"),
             py::arg("audio_num_frames"),
             R"(
                Reshapes the model for specific input dimensions.
                batch_size (int): Batch size.
                num_frames (int): Number of video frames.
                height (int): Video height.
                width (int): Video width.
                audio_num_frames (int): Number of audio latent frames.
            )")
        .def("set_hidden_states",
             &ov::genai::LTX2VideoTransformer3DModel::set_hidden_states,
             py::arg("tensor_name"),
             py::arg("tensor"),
             R"(
                Sets a model input tensor by name.
                tensor_name (str): Name of the tensor input.
                tensor (ov.Tensor): Tensor to set.
            )")
        .def("infer",
             [](ov::genai::LTX2VideoTransformer3DModel& self,
                const ov::Tensor& video_latent,
                const ov::Tensor& audio_latent,
                float timestep) {
                 py::gil_scoped_release rel;
                 return self.infer(video_latent, audio_latent, timestep);
             },
             py::arg("video_latent"),
             py::arg("audio_latent"),
             py::arg("timestep"),
             R"(
                Performs joint video and audio inference.
                video_latent (ov.Tensor): Packed video latent tensor.
                audio_latent (ov.Tensor): Packed audio latent tensor.
                timestep (float): Current timestep.
                Returns: Tuple of video and audio velocity predictions.
            )");
}

void init_autoencoder_kl_ltx2_video(py::module_& m) {
    auto vae = py::class_<ov::genai::AutoencoderKLLTX2Video>(m,
                                                             "AutoencoderKLLTX2Video",
                                                             "AutoencoderKLLTX2Video class for LTX2 VAE decoding.")
                   .def(py::init([](const std::filesystem::path& vae_decoder_path) {
                            return std::make_unique<ov::genai::AutoencoderKLLTX2Video>(vae_decoder_path);
                        }),
                        py::arg("vae_decoder_path"),
                        R"(
            AutoencoderKLLTX2Video class constructor.
            vae_decoder_path (os.PathLike): VAE decoder directory.
        )")
                   .def(py::init([](const std::filesystem::path& vae_decoder_path,
                                    const std::string& device,
                                    const py::kwargs& kwargs) {
                            return std::make_unique<ov::genai::AutoencoderKLLTX2Video>(
                                vae_decoder_path,
                                device,
                                pyutils::kwargs_to_any_map(kwargs));
                        }),
                        py::arg("vae_decoder_path"),
                        py::arg("device"),
                        R"(
            AutoencoderKLLTX2Video class constructor.
            vae_decoder_path (os.PathLike): VAE decoder directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )");

    py::class_<ov::genai::AutoencoderKLLTX2Video::Config>(vae, "Config", "Configuration for AutoencoderKLLTX2Video.")
        .def(py::init([](const std::filesystem::path& config_path) {
                 return std::make_unique<ov::genai::AutoencoderKLLTX2Video::Config>(config_path);
             }),
             py::arg("config_path"))
        .def_readonly("latent_channels", &ov::genai::AutoencoderKLLTX2Video::Config::latent_channels)
        .def_readonly("scaling_factor", &ov::genai::AutoencoderKLLTX2Video::Config::scaling_factor)
        .def_readonly("spatial_compression_ratio",
                      &ov::genai::AutoencoderKLLTX2Video::Config::spatial_compression_ratio)
        .def_readonly("temporal_compression_ratio",
                      &ov::genai::AutoencoderKLLTX2Video::Config::temporal_compression_ratio);

    vae.def("get_config", &ov::genai::AutoencoderKLLTX2Video::get_config)
        .def(
            "compile",
            [](ov::genai::AutoencoderKLLTX2Video& self, const std::string& device, const py::kwargs& kwargs) {
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
             &ov::genai::AutoencoderKLLTX2Video::reshape,
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
             &ov::genai::AutoencoderKLLTX2Video::decode,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("latent"),
             R"(
                Decodes latent video to pixel space.
                latent (ov.Tensor): Latent video tensor.
                Returns: Decoded video tensor.
            )");
}

void init_autoencoder_kl_ltx2_audio(py::module_& m) {
    auto audio_vae = py::class_<ov::genai::AutoencoderKLLTX2Audio>(m,
                                                                   "AutoencoderKLLTX2Audio",
                                                                   "AutoencoderKLLTX2Audio class for LTX2 audio VAE decoding.")
                         .def(py::init([](const std::filesystem::path& decoder_path) {
                                  return std::make_unique<ov::genai::AutoencoderKLLTX2Audio>(decoder_path);
                              }),
                              py::arg("decoder_path"),
                              R"(
            AutoencoderKLLTX2Audio class constructor.
            decoder_path (os.PathLike): Audio VAE decoder directory.
        )")
                         .def(py::init([](const std::filesystem::path& decoder_path,
                                          const std::string& device,
                                          const py::kwargs& kwargs) {
                                  return std::make_unique<ov::genai::AutoencoderKLLTX2Audio>(
                                      decoder_path,
                                      device,
                                      pyutils::kwargs_to_any_map(kwargs));
                              }),
                              py::arg("decoder_path"),
                              py::arg("device"),
                              R"(
            AutoencoderKLLTX2Audio class constructor.
            decoder_path (os.PathLike): Audio VAE decoder directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )");

    py::class_<ov::genai::AutoencoderKLLTX2Audio::Config>(audio_vae,
                                                          "Config",
                                                          "Configuration for AutoencoderKLLTX2Audio.")
        .def(py::init([](const std::filesystem::path& config_path) {
                 return std::make_unique<ov::genai::AutoencoderKLLTX2Audio::Config>(config_path);
             }),
             py::arg("config_path"))
        .def_readonly("latent_channels", &ov::genai::AutoencoderKLLTX2Audio::Config::latent_channels)
        .def_readonly("mel_bins", &ov::genai::AutoencoderKLLTX2Audio::Config::mel_bins)
        .def_readonly("sample_rate", &ov::genai::AutoencoderKLLTX2Audio::Config::sample_rate)
        .def_readonly("mel_hop_length", &ov::genai::AutoencoderKLLTX2Audio::Config::mel_hop_length)
        .def_readonly("mel_compression_ratio", &ov::genai::AutoencoderKLLTX2Audio::Config::mel_compression_ratio)
        .def_readonly("temporal_compression_ratio",
                      &ov::genai::AutoencoderKLLTX2Audio::Config::temporal_compression_ratio);

    audio_vae.def("get_config", &ov::genai::AutoencoderKLLTX2Audio::get_config)
        .def(
            "compile",
            [](ov::genai::AutoencoderKLLTX2Audio& self, const std::string& device, const py::kwargs& kwargs) {
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
             &ov::genai::AutoencoderKLLTX2Audio::reshape,
             py::arg("batch_size"),
             py::arg("audio_num_frames"),
             R"(
                Reshapes the model for specific input dimensions.
                batch_size (int): Batch size.
                audio_num_frames (int): Number of audio latent frames.
            )")
        .def("decode",
             &ov::genai::AutoencoderKLLTX2Audio::decode,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("latent"),
             R"(
                Decodes audio latents to a mel spectrogram.
                latent (ov.Tensor): Audio latent tensor [B, C, L, M].
                Returns: Mel spectrogram tensor.
            )");
}

void init_ltx2_text_connectors(py::module_& m) {
    py::class_<ov::genai::LTX2TextConnectors>(m,
                                              "LTX2TextConnectors",
                                              "LTX2TextConnectors class projecting text embeddings per modality.")
        .def(py::init([](const std::filesystem::path& root_dir) {
                 return std::make_unique<ov::genai::LTX2TextConnectors>(root_dir);
             }),
             py::arg("root_dir"),
             R"(
            LTX2TextConnectors class constructor.
            root_dir (os.PathLike): Model root directory.
        )")
        .def(py::init([](const std::filesystem::path& root_dir, const std::string& device, const py::kwargs& kwargs) {
                 return std::make_unique<ov::genai::LTX2TextConnectors>(root_dir,
                                                                        device,
                                                                        pyutils::kwargs_to_any_map(kwargs));
             }),
             py::arg("root_dir"),
             py::arg("device"),
             R"(
            LTX2TextConnectors class constructor.
            root_dir (os.PathLike): Model root directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )")
        .def(
            "compile",
            [](ov::genai::LTX2TextConnectors& self, const std::string& device, const py::kwargs& kwargs) {
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
             &ov::genai::LTX2TextConnectors::reshape,
             py::arg("batch_size"),
             R"(
                Reshapes the model for a specific batch size.
                batch_size (int): Batch size.
            )")
        .def("infer",
             [](ov::genai::LTX2TextConnectors& self,
                const ov::Tensor& text_encoder_hidden_states,
                const ov::Tensor& attention_mask) {
                 ov::genai::LTX2TextConnectors::Output output;
                 {
                     py::gil_scoped_release rel;
                     output = self.infer(text_encoder_hidden_states, attention_mask);
                 }
                 return py::make_tuple(output.video_text_embedding,
                                       output.audio_text_embedding,
                                       output.connector_attention_mask);
             },
             py::arg("text_encoder_hidden_states"),
             py::arg("attention_mask"),
             R"(
                Projects text encoder hidden states into per-modality embeddings.
                text_encoder_hidden_states (ov.Tensor): Stacked text encoder hidden states.
                attention_mask (ov.Tensor): Prompt attention mask.
                Returns: Tuple of video text embedding, audio text embedding and connector attention mask.
            )");
}

void init_ltx2_vocoder(py::module_& m) {
    auto vocoder = py::class_<ov::genai::LTX2Vocoder>(m,
                                                      "LTX2Vocoder",
                                                      "LTX2Vocoder class converting mel spectrograms to waveforms.")
                       .def(py::init([](const std::filesystem::path& root_dir) {
                                return std::make_unique<ov::genai::LTX2Vocoder>(root_dir);
                            }),
                            py::arg("root_dir"),
                            R"(
            LTX2Vocoder class constructor.
            root_dir (os.PathLike): Model root directory.
        )")
                       .def(py::init([](const std::filesystem::path& root_dir,
                                        const std::string& device,
                                        const py::kwargs& kwargs) {
                                return std::make_unique<ov::genai::LTX2Vocoder>(root_dir,
                                                                                device,
                                                                                pyutils::kwargs_to_any_map(kwargs));
                            }),
                            py::arg("root_dir"),
                            py::arg("device"),
                            R"(
            LTX2Vocoder class constructor.
            root_dir (os.PathLike): Model root directory.
            device (str): Device on which inference will be done.
            kwargs: Device properties.
        )");

    py::class_<ov::genai::LTX2Vocoder::Config>(vocoder, "Config", "Configuration for LTX2Vocoder.")
        .def(py::init([](const std::filesystem::path& config_path) {
                 return std::make_unique<ov::genai::LTX2Vocoder::Config>(config_path);
             }),
             py::arg("config_path"))
        .def_readonly("output_sampling_rate", &ov::genai::LTX2Vocoder::Config::output_sampling_rate);

    vocoder.def("get_config", &ov::genai::LTX2Vocoder::get_config)
        .def(
            "compile",
            [](ov::genai::LTX2Vocoder& self, const std::string& device, const py::kwargs& kwargs) {
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
             &ov::genai::LTX2Vocoder::reshape,
             py::arg("batch_size"),
             R"(
                Reshapes the model for a specific batch size.
                batch_size (int): Batch size.
            )")
        .def("infer",
             &ov::genai::LTX2Vocoder::infer,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("mel_spectrogram"),
             R"(
                Converts a mel spectrogram to an audio waveform.
                mel_spectrogram (ov.Tensor): Mel spectrogram tensor.
                Returns: Waveform tensor [B, num_channels, num_samples].
            )");
}

void init_video_generation_models(py::module_& m) {
    init_ltx_video_transformer_3d_model(m);
    init_autoencoder_kl_ltx_video(m);
    init_ltx2_video_transformer_3d_model(m);
    init_autoencoder_kl_ltx2_video(m);
    init_autoencoder_kl_ltx2_audio(m);
    init_ltx2_text_connectors(m);
    init_ltx2_vocoder(m);
}
