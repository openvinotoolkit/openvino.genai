// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/autoencoder_kl_qwen_image.hpp"

#include <cmath>
#include <cstring>
#include <fstream>

#include "openvino/runtime/core.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/constant.hpp"

#include "utils.hpp"
#include "json_utils.hpp"
#include "lora/helper.hpp"

namespace ov {
namespace genai {

AutoencoderKLQwenImage::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "z_dim", z_dim);
    read_json_param(data, "input_channels", input_channels);

    if (data.contains("temperal_downsample")) {
        temperal_downsample = data["temperal_downsample"].get<std::vector<bool>>();
    }

    // optimum-intel exports latents_mean/std as latents_mean_data/latents_std_data
    if (data.contains("latents_mean_data")) {
        latents_mean = data["latents_mean_data"].get<std::vector<float>>();
    } else if (data.contains("latents_mean")) {
        latents_mean = data["latents_mean"].get<std::vector<float>>();
    }

    if (data.contains("latents_std_data")) {
        latents_std = data["latents_std_data"].get<std::vector<float>>();
    } else if (data.contains("latents_std")) {
        latents_std = data["latents_std"].get<std::vector<float>>();
    }
}

AutoencoderKLQwenImage::AutoencoderKLQwenImage(const std::filesystem::path& vae_decoder_path)
    : m_config(vae_decoder_path / "config.json") {
    m_decoder_model = utils::singleton_core().read_model(vae_decoder_path / "openvino_model.xml");
    merge_vae_image_post_processing();
}

AutoencoderKLQwenImage::AutoencoderKLQwenImage(const std::filesystem::path& vae_encoder_path,
                                               const std::filesystem::path& vae_decoder_path)
    : AutoencoderKLQwenImage(vae_decoder_path) {
    m_encoder_model = utils::singleton_core().read_model(vae_encoder_path / "openvino_model.xml");
}

AutoencoderKLQwenImage::AutoencoderKLQwenImage(const std::filesystem::path& vae_decoder_path,
                                               const std::string& device,
                                               const ov::AnyMap& properties)
    : m_config(vae_decoder_path / "config.json") {
    m_decoder_model = utils::singleton_core().read_model(vae_decoder_path / "openvino_model.xml");
    merge_vae_image_post_processing();
    compile(device, *extract_adapters_from_properties(properties));
}

AutoencoderKLQwenImage::AutoencoderKLQwenImage(const std::filesystem::path& vae_encoder_path,
                                               const std::filesystem::path& vae_decoder_path,
                                               const std::string& device,
                                               const ov::AnyMap& properties)
    : AutoencoderKLQwenImage(vae_encoder_path, vae_decoder_path) {
    compile(device, *extract_adapters_from_properties(properties));
}

AutoencoderKLQwenImage::AutoencoderKLQwenImage(const AutoencoderKLQwenImage&) = default;

AutoencoderKLQwenImage AutoencoderKLQwenImage::clone() {
    OPENVINO_ASSERT((m_decoder_model != nullptr) ^ static_cast<bool>(m_decoder_request),
                    "AutoencoderKLQwenImage must have exactly one of m_decoder_model or m_decoder_request initialized");

    AutoencoderKLQwenImage cloned = *this;

    if (m_decoder_model) {
        cloned.m_decoder_model = m_decoder_model->clone();
    } else {
        cloned.m_decoder_request = m_decoder_request.get_compiled_model().create_infer_request();
    }

    if (m_encoder_model) {
        cloned.m_encoder_model = m_encoder_model->clone();
    } else if (m_encoder_request) {
        cloned.m_encoder_request = m_encoder_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

AutoencoderKLQwenImage& AutoencoderKLQwenImage::reshape(int batch_size, int height, int width) {
    OPENVINO_ASSERT(m_decoder_model, "Model has been already compiled. Cannot reshape already compiled model");

    const size_t vae_scale_factor = get_vae_scale_factor();

    OPENVINO_ASSERT((height % vae_scale_factor == 0 || height < 0) &&
                    (width % vae_scale_factor == 0 || width < 0),
                    "Both 'width' and 'height' must be divisible by ", vae_scale_factor);

    if (m_encoder_model) {
        ov::PartialShape input_shape = m_encoder_model->input(0).get_partial_shape();
        // Encoder: (B, 3, 1, H, W)
        std::map<size_t, ov::PartialShape> idx_to_shape{{0, {batch_size, input_shape[1], 1, height, width}}};
        m_encoder_model->reshape(idx_to_shape);
    }

    const int lat_h = height / vae_scale_factor;
    const int lat_w = width / vae_scale_factor;

    ov::PartialShape input_shape = m_decoder_model->input(0).get_partial_shape();
    // Decoder: (B, z_dim, 1, H_lat, W_lat)
    std::map<size_t, ov::PartialShape> idx_to_shape{{0, {batch_size, input_shape[1], 1, lat_h, lat_w}}};
    m_decoder_model->reshape(idx_to_shape);

    return *this;
}

AutoencoderKLQwenImage& AutoencoderKLQwenImage::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_decoder_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::Core core = utils::singleton_core();

    std::optional<AdapterConfig> unused;
    auto filtered_properties = extract_adapters_from_properties(properties, &unused);

    if (m_encoder_model) {
        ov::CompiledModel compiled = core.compile_model(m_encoder_model, device, *filtered_properties);
        ov::genai::utils::print_compiled_model_properties(compiled, "AutoencoderKLQwenImage encoder model");
        m_encoder_request = compiled.create_infer_request();
        m_encoder_model.reset();
    }

    ov::CompiledModel compiled = core.compile_model(m_decoder_model, device, *filtered_properties);
    ov::genai::utils::print_compiled_model_properties(compiled, "AutoencoderKLQwenImage decoder model");
    m_decoder_request = compiled.create_infer_request();
    m_decoder_model.reset();

    return *this;
}

ov::Tensor AutoencoderKLQwenImage::decode(ov::Tensor latent) {
    OPENVINO_ASSERT(m_decoder_request, "VAE decoder model must be compiled first");

    m_decoder_request.set_input_tensor(latent);
    m_decoder_request.infer();
    ov::Tensor output = m_decoder_request.get_output_tensor();

    // Output after post-processing is u8 with shape (B, 3, 1, H, W).
    // Convert to NHWC: (B, H, W, 3) by squeezing temporal dim and transposing.
    const ov::Shape& shape = output.get_shape();
    const size_t batch = shape[0];
    const size_t channels = shape[1];
    const size_t height = shape[3];
    const size_t width = shape[4];

    ov::Tensor result(output.get_element_type(), {batch, height, width, channels});
    const uint8_t* src = output.data<uint8_t>();
    uint8_t* dst = result.data<uint8_t>();

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    dst[b * height * width * channels + h * width * channels + w * channels + c] =
                        src[b * channels * height * width + c * height * width + h * width + w];
                }
            }
        }
    }

    return result;
}

ov::Tensor AutoencoderKLQwenImage::encode(ov::Tensor image, std::shared_ptr<Generator> generator) {
    OPENVINO_ASSERT(m_encoder_request, "VAE encoder model must be compiled first");
    OPENVINO_ASSERT(generator, "Generator must not be nullptr");

    m_encoder_request.set_input_tensor(image);
    m_encoder_request.infer();

    ov::Tensor output = m_encoder_request.get_output_tensor();

    // Output is latent_parameters: (B, z_dim*2, T, H, W) — DiagonalGaussianDistribution
    const ov::Shape& shape = output.get_shape();
    const size_t batch = shape[0];
    const size_t total_channels = shape[1];
    const size_t z_dim = total_channels / 2;
    const size_t spatial = shape[2] * shape[3] * shape[4];

    ov::Tensor latent(ov::element::f32, {batch, z_dim, shape[2], shape[3], shape[4]});
    float* lat_data = latent.data<float>();
    const float* params = output.data<float>();

    // Sample: z = mean + std * randn
    ov::Tensor noise = generator->randn_tensor({batch, z_dim, shape[2], shape[3], shape[4]});
    const float* noise_data = noise.data<float>();

    for (size_t b = 0; b < batch; ++b) {
        const float* mean_ptr = params + b * total_channels * spatial;
        const float* logvar_ptr = mean_ptr + z_dim * spatial;
        float* out_ptr = lat_data + b * z_dim * spatial;
        const float* n_ptr = noise_data + b * z_dim * spatial;

        for (size_t i = 0; i < z_dim * spatial; ++i) {
            float logvar = std::min(std::max(logvar_ptr[i], -30.0f), 20.0f);
            float std_val = std::exp(0.5f * logvar);
            out_ptr[i] = mean_ptr[i] + std_val * n_ptr[i];
        }
    }

    return latent;
}

ov::Tensor AutoencoderKLQwenImage::encode(ov::Tensor image) {
    OPENVINO_ASSERT(m_encoder_request, "VAE encoder model must be compiled first");

    m_encoder_request.set_input_tensor(image);
    m_encoder_request.infer();

    ov::Tensor output = m_encoder_request.get_output_tensor();

    // Return mean only (first z_dim channels)
    const ov::Shape& shape = output.get_shape();
    const size_t batch = shape[0];
    const size_t z_dim = shape[1] / 2;
    const size_t spatial = shape[2] * shape[3] * shape[4];

    ov::Tensor latent(ov::element::f32, {batch, z_dim, shape[2], shape[3], shape[4]});
    float* lat_data = latent.data<float>();
    const float* params = output.data<float>();

    for (size_t b = 0; b < batch; ++b) {
        std::memcpy(lat_data + b * z_dim * spatial,
                    params + b * shape[1] * spatial,
                    z_dim * spatial * sizeof(float));
    }

    return latent;
}

const AutoencoderKLQwenImage::Config& AutoencoderKLQwenImage::get_config() const {
    return m_config;
}

size_t AutoencoderKLQwenImage::get_vae_scale_factor() const {
    return static_cast<size_t>(std::pow(2, m_config.temperal_downsample.size()));
}

void AutoencoderKLQwenImage::merge_vae_image_post_processing() const {
    ov::preprocess::PrePostProcessor ppp(m_decoder_model);

    // Decoder output is in [-1, 1]. Convert to [0, 255] u8.
    ppp.output().postprocess().custom([](const ov::Output<ov::Node>& port) {
        auto constant_0_5 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.5f);
        auto constant_255 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 255.0f);
        auto scaled_0_5 = std::make_shared<ov::op::v1::Multiply>(port, constant_0_5);
        auto added_0_5 = std::make_shared<ov::op::v1::Add>(scaled_0_5, constant_0_5);
        auto clamped = std::make_shared<ov::op::v0::Clamp>(added_0_5, 0.0f, 1.0f);
        return std::make_shared<ov::op::v1::Multiply>(clamped, constant_255);
    });
    ppp.output().postprocess().convert_element_type(ov::element::u8);

    ppp.build();
}

}  // namespace genai
}  // namespace ov
