// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/autoencoder_kl_ltx_video.hpp"

#include <fstream>
#include <memory>
#include <numeric>

#include "openvino/runtime/core.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/constant.hpp"

#include "utils.hpp"
#include "json_utils.hpp"
#include "lora/helper.hpp"

using namespace ov::genai;

namespace {

class DiagonalGaussianDistribution {
public:
    explicit DiagonalGaussianDistribution(ov::Tensor parameters)
        : m_parameters(parameters) {
        ov::Shape shape = parameters.get_shape();
        OPENVINO_ASSERT(shape[0] == 1, "Batch size must be 1");
        shape[1] /= 2;

        m_mean = ov::Tensor(parameters.get_element_type(), shape, parameters.data());
        m_std = ov::Tensor(m_mean.get_element_type(), shape);
        ov::Tensor logvar(parameters.get_element_type(), shape, m_mean.data<float>() + m_mean.get_size());

        float* logvar_data = logvar.data<float>();
        float* std_data = m_std.data<float>();

        for (size_t i = 0; i < logvar.get_size(); ++i) {
            logvar_data[i] = std::min(std::max(logvar_data[i], -30.0f), 20.0f);
            std_data[i] = std::exp(0.5f * logvar_data[i]);
        }
    }

    ov::Tensor sample(std::shared_ptr<ov::genai::Generator> generator) const {
        OPENVINO_ASSERT(generator, "Generator must not be nullptr");

        ov::Tensor rand_tensor = generator->randn_tensor(m_mean.get_shape());

        float* rand_tensor_data = rand_tensor.data<float>();
        const float* mean_data = m_mean.data<float>();
        const float* std_data = m_std.data<float>();

        for (size_t i = 0; i < rand_tensor.get_size(); ++i) {
            rand_tensor_data[i] = mean_data[i] + std_data[i] * rand_tensor_data[i];
        }

        return rand_tensor;
    }

private:
    ov::Tensor m_parameters;
    ov::Tensor m_mean, m_std;
};

// for BW compatibility with 2024.6.0
ov::AnyMap handle_scale_factor(std::shared_ptr<ov::Model> model, const std::string& device, ov::AnyMap properties) {
    auto it = properties.find("WA_INFERENCE_PRECISION_HINT");
    ov::element::Type wa_inference_precision = it != properties.end() ? it->second.as<ov::element::Type>() : ov::element::dynamic;
    if (it != properties.end()) {
        properties.erase(it);
    }

    const std::vector<std::string> activation_scale_factor_path = { "runtime_options", ov::hint::activations_scale_factor.name() };
    const bool activation_scale_factor_defined = model->has_rt_info(activation_scale_factor_path);

    // convert WA inference precision to actual inference precision if activation_scale_factor is not defined in IR
    if (device.find("GPU") != std::string::npos && !activation_scale_factor_defined && wa_inference_precision != ov::element::dynamic) {
        properties[ov::hint::inference_precision.name()] = wa_inference_precision;
    }

    return properties;
}

std::pair<int64_t, int64_t> get_transformer_patch_size(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);
    nlohmann::json data = nlohmann::json::parse(file);

    int64_t patch_size, patch_size_t;
    utils::read_json_param(data, "patch_size", patch_size);
    utils::read_json_param(data, "patch_size_t", patch_size_t);

    return {patch_size, patch_size_t};
}

} // namespace

AutoencoderKLLTXVideo::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "latent_channels", latent_channels);
    read_json_param(data, "out_channels", out_channels);
    read_json_param(data, "scaling_factor", scaling_factor);
    read_json_param(data, "block_out_channels", block_out_channels);
    read_json_param(data, "patch_size", patch_size);
    read_json_param(data, "patch_size_t", patch_size_t);
    read_json_param(data, "spatio_temporal_scaling", spatio_temporal_scaling);
    read_json_param(data, "latents_mean_data", latents_mean_data);
    read_json_param(data, "latents_std_data", latents_std_data);
    read_json_param(data, "timestep_conditioning", timestep_conditioning);

    if (latents_mean_data.empty()) {
        latents_mean_data.assign(latent_channels, 0.0f);
    }
    if (latents_std_data.empty()) {
        latents_std_data.assign(latent_channels, 1.0f);
    }
}

AutoencoderKLLTXVideo::AutoencoderKLLTXVideo(const std::filesystem::path& vae_decoder_path)
    : m_config(vae_decoder_path / "config.json") {
    m_decoder_model = utils::singleton_core().read_model(vae_decoder_path / "openvino_model.xml");
    std::tie(m_transformer_patch_size, m_transformer_patch_size_t) = get_transformer_patch_size(vae_decoder_path.parent_path() / "transformer" / "config.json");
    // apply VaeImageProcessor postprocessing steps by merging them into the VAE decoder model
    merge_vae_video_post_processing();
}

AutoencoderKLLTXVideo::AutoencoderKLLTXVideo(const std::filesystem::path& vae_encoder_path,
                                const std::filesystem::path& vae_decoder_path)
    : AutoencoderKLLTXVideo(vae_decoder_path) {
    m_encoder_model = utils::singleton_core().read_model(vae_encoder_path / "openvino_model.xml");
}

AutoencoderKLLTXVideo::AutoencoderKLLTXVideo(const std::filesystem::path& vae_decoder_path,
                             const std::string& device,
                             const ov::AnyMap& properties)
    : AutoencoderKLLTXVideo(vae_decoder_path) {
    compile(device, *extract_adapters_from_properties(properties));
}

AutoencoderKLLTXVideo::AutoencoderKLLTXVideo(const std::filesystem::path& vae_encoder_path,
                             const std::filesystem::path& vae_decoder_path,
                             const std::string& device,
                             const ov::AnyMap& properties)
    : AutoencoderKLLTXVideo(vae_encoder_path, vae_decoder_path) {
    compile(device, *extract_adapters_from_properties(properties));
}

AutoencoderKLLTXVideo& AutoencoderKLLTXVideo::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_decoder_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::Core core = utils::singleton_core();

    std::optional<AdapterConfig> unused;
    auto filtered_properties = extract_adapters_from_properties(properties, &unused);

    if (m_encoder_model) {
        ov::CompiledModel encoder_compiled_model = core.compile_model(m_encoder_model, device, handle_scale_factor(m_encoder_model, device, *filtered_properties));
        ov::genai::utils::print_compiled_model_properties(encoder_compiled_model, "Auto encoder KL LTX video encoder model");
        m_encoder_request = encoder_compiled_model.create_infer_request();
        m_encoder_model.reset();
    }

    ov::CompiledModel decoder_compiled_model = core.compile_model(m_decoder_model, device, handle_scale_factor(m_decoder_model, device, *filtered_properties));
    ov::genai::utils::print_compiled_model_properties(decoder_compiled_model, "Auto encoder KL LTX video decoder model");
    m_decoder_request = decoder_compiled_model.create_infer_request();
    // release the original model
    m_decoder_model.reset();

    return *this;
}

AutoencoderKLLTXVideo& AutoencoderKLLTXVideo::reshape(int64_t batch_size,
                                                      int64_t num_frames,
                                                      int64_t height,
                                                      int64_t width) {
    OPENVINO_ASSERT(m_decoder_model, "Model has been already compiled. Cannot reshape already compiled model");
    OPENVINO_ASSERT(height > 0, "Height must be positive");
    OPENVINO_ASSERT(height % 32 == 0, "Height have to be divisible by 32 but got ", height);
    OPENVINO_ASSERT(width > 0, "Width must be positive");
    OPENVINO_ASSERT(width % 32 == 0, "Width have to be divisible by 32 but got ", width);

    if (m_encoder_model) {
        ov::PartialShape input_shape = m_encoder_model->input(0).get_partial_shape();
        std::map<size_t, ov::PartialShape> idx_to_shape{{0, {batch_size, input_shape[1], num_frames, height, width}}};
        m_encoder_model->reshape(idx_to_shape);
    }

    int64_t spatial_compression_ratio =
        get_config().patch_size *
        std::pow(
            2,
            std::accumulate(get_config().spatio_temporal_scaling.begin(), get_config().spatio_temporal_scaling.end(), 0));
    int64_t temporal_compression_ratio =
        get_config().patch_size_t *
        std::pow(
            2,
            std::accumulate(get_config().spatio_temporal_scaling.begin(), get_config().spatio_temporal_scaling.end(), 0));

    num_frames = ((num_frames - 1) / temporal_compression_ratio + 1) / m_transformer_patch_size_t;
    height /= (spatial_compression_ratio * m_transformer_patch_size);
    width /= (spatial_compression_ratio * m_transformer_patch_size);

    ov::PartialShape input_shape = m_decoder_model->input(0).get_partial_shape();
    std::map<size_t, ov::PartialShape> idx_to_shape{{0, {batch_size, input_shape[1], num_frames, height, width}}};
    m_decoder_model->reshape(idx_to_shape);

    return *this;
}

ov::Tensor AutoencoderKLLTXVideo::decode(const ov::Tensor& latent) {
    OPENVINO_ASSERT(m_decoder_request, "VAE decoder model must be compiled first. Cannot infer non-compiled model");

    m_decoder_request.set_input_tensor(latent);
    m_decoder_request.infer();
    return m_decoder_request.get_output_tensor();
}

ov::Tensor AutoencoderKLLTXVideo::encode(const ov::Tensor& video, std::shared_ptr<Generator> generator) {
    OPENVINO_ASSERT(m_encoder_request || m_encoder_model,
        "AutoencoderKLLTXVideo is created without 'VAE encoder' capability. "
        "Please, pass 'vae_encoder_path' argument to constructor.");
    OPENVINO_ASSERT(m_encoder_request,
        "VAE encoder model must be compiled first. Cannot infer non-compiled model");

    m_encoder_request.set_input_tensor(video);
    m_encoder_request.infer();

    ov::Tensor output = m_encoder_request.get_output_tensor(), latent;

    ov::CompiledModel compiled_model = m_encoder_request.get_compiled_model();
    auto outputs = compiled_model.outputs();
    OPENVINO_ASSERT(outputs.size() == 1, "AutoencoderKLLTXVideo encoder model is expected to have a single output");

    const std::string output_name = outputs[0].get_any_name();
    if (output_name == "latent_sample") {
        latent = output;
    } else if (output_name == "latent_parameters") {
        latent = DiagonalGaussianDistribution(output).sample(generator);
    } else {
        OPENVINO_THROW("Unexpected output name for AutoencoderKLLTXVideo encoder '", output_name, "'");
    }

    // normalize latents channel-wise: (latents - latents_mean) * scaling_factor / latents_std
    // inverse of denormalize_latents used in the decode path
    const ov::Shape shape = latent.get_shape();
    OPENVINO_ASSERT(shape.size() == 5, "Encoder output expected to be [B, C, F, H, W]");
    const size_t B = shape[0], C = shape[1], spatial = shape[2] * shape[3] * shape[4];

    float* latent_data = latent.data<float>();
    const auto& mean = m_config.latents_mean_data;
    const auto& std_data = m_config.latents_std_data;
    const float scale = m_config.scaling_factor;

    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < C; ++c) {
            float* ptr = latent_data + (b * C + c) * spatial;
            const float m = mean[c], s = std_data[c];
            for (size_t i = 0; i < spatial; ++i) {
                ptr[i] = (ptr[i] - m) * scale / s;
            }
        }
    }

    return latent;
}

const AutoencoderKLLTXVideo::Config& AutoencoderKLLTXVideo::get_config() const {
    return m_config;
}

size_t AutoencoderKLLTXVideo::get_vae_scale_factor() const {  // TODO: compare with reference. Drop?
    return std::pow(2, m_config.block_out_channels.size() - 1);
}

void AutoencoderKLLTXVideo::merge_vae_video_post_processing() const {
    ov::preprocess::PrePostProcessor ppp(m_decoder_model);

    if (m_config.scaling_factor != 1.0f)
        ppp.input().preprocess().scale(m_config.scaling_factor);

    // (x / 2 + 0.5) -> clamp(0..1) -> *255 -> u8
    ppp.output().postprocess().custom([](const ov::Output<ov::Node>& port) {
        auto c_0_5  = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.5f);
        auto c_255  = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 255.0f);

        auto scaled = std::make_shared<ov::op::v1::Multiply>(port, c_0_5);
        auto shifted = std::make_shared<ov::op::v1::Add>(scaled, c_0_5);
        auto clamped = std::make_shared<ov::op::v0::Clamp>(shifted, 0.0f, 1.0f);

        return std::make_shared<ov::op::v1::Multiply>(clamped, c_255);
    });

    ppp.output().postprocess().convert_element_type(ov::element::u8);

    // [B, C, F, H, W] -> [B, F, H, W, C]
    ppp.output().model().set_layout("NCDHW");
    ppp.output().tensor().set_layout("NDHWC");

    ppp.build();
}
