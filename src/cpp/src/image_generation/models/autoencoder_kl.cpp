// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/autoencoder_kl.hpp"

#include <fstream>
#include <memory>

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

namespace ov {
namespace genai {

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

        float * logvar_data = logvar.data<float>();
        float * std_data = m_std.data<float>();

        for (size_t i = 0; i < logvar.get_size(); ++i) {
            logvar_data[i] = std::min(std::max(logvar_data[i], -30.0f), 20.0f);
            std_data[i] = std::exp(0.5 * logvar_data[i]);
        }
    }

    ov::Tensor sample(std::shared_ptr<Generator> generator) const {
        OPENVINO_ASSERT(generator, "Generator must not be nullptr");

        ov::Tensor rand_tensor = generator->randn_tensor(m_mean.get_shape());

        float * rand_tensor_data = rand_tensor.data<float>();
        const float * mean_data = m_mean.data<float>();
        const float * std_data = m_std.data<float>();

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

} // namespace

size_t get_vae_scale_factor(const std::filesystem::path& vae_config_path) {
    std::ifstream file(vae_config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", vae_config_path);
    nlohmann::json data = nlohmann::json::parse(file);

    std::vector<size_t> block_out_channels;
    utils::read_json_param(data, "block_out_channels", block_out_channels);
    return std::pow(2, block_out_channels.size() - 1);
}

AutoencoderKL::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "latent_channels", latent_channels);
    read_json_param(data, "out_channels", out_channels);
    read_json_param(data, "shift_factor", shift_factor);
    read_json_param(data, "scaling_factor", scaling_factor);
    read_json_param(data, "block_out_channels", block_out_channels);
}

AutoencoderKL::AutoencoderKL(const std::filesystem::path& vae_decoder_path)
    : m_config(vae_decoder_path / "config.json") {
    m_decoder_model = utils::singleton_core().read_model(vae_decoder_path / "openvino_model.xml");
    // apply VaeImageProcessor postprocessing steps by merging them into the VAE decoder model
    merge_vae_image_post_processing();
}

AutoencoderKL::AutoencoderKL(const std::filesystem::path& vae_encoder_path,
                             const std::filesystem::path& vae_decoder_path)
    : AutoencoderKL(vae_decoder_path) {
    m_encoder_model = utils::singleton_core().read_model(vae_encoder_path / "openvino_model.xml");
}

AutoencoderKL::AutoencoderKL(const std::filesystem::path& vae_decoder_path,
                             const std::string& device,
                             const ov::AnyMap& properties)
    : m_config(vae_decoder_path / "config.json") {

    const auto [properties_without_blob, blob_path] = utils::extract_export_properties(properties);

    if (blob_path.has_value()) {
        import_model(*blob_path, device, properties_without_blob);
        return;
    }

    m_decoder_model = utils::singleton_core().read_model(vae_decoder_path / "openvino_model.xml");
    // apply VaeImageProcessor postprocessing steps by merging them into the VAE decoder model
    merge_vae_image_post_processing();
    compile(device, *extract_adapters_from_properties(properties_without_blob));
}

AutoencoderKL::AutoencoderKL(const std::filesystem::path& vae_encoder_path,
                             const std::filesystem::path& vae_decoder_path,
                             const std::string& device,
                             const ov::AnyMap& properties)
    : AutoencoderKL(vae_encoder_path, vae_decoder_path) {
    compile(device, *extract_adapters_from_properties(properties));
}

AutoencoderKL::AutoencoderKL(const std::string& vae_decoder_model,
                             const Tensor& vae_decoder_weights,
                             const Config& vae_decoder_config)
    : m_config(vae_decoder_config) {
    m_decoder_model = utils::singleton_core().read_model(vae_decoder_model, vae_decoder_weights);
    // apply VaeImageProcessor postprocessing steps by merging them into the VAE decoder model
    merge_vae_image_post_processing();
}

AutoencoderKL::AutoencoderKL(const std::string& vae_encoder_model,
                             const Tensor& vae_encoder_weights,
                             const std::string& vae_decoder_model,
                             const Tensor& vae_decoder_weights,
                             const Config& vae_decoder_config)
    : AutoencoderKL(vae_decoder_model, vae_decoder_weights, vae_decoder_config) {
    m_encoder_model = utils::singleton_core().read_model(vae_encoder_model, vae_encoder_weights);
}

AutoencoderKL::AutoencoderKL(const std::string& vae_decoder_model,
                             const Tensor& vae_decoder_weights,
                             const Config& vae_decoder_config,
                             const std::string& device,
                             const ov::AnyMap& properties)
    : AutoencoderKL(vae_decoder_model, vae_decoder_weights, vae_decoder_config) {
    compile(device, *extract_adapters_from_properties(properties));
}

AutoencoderKL::AutoencoderKL(const std::string& vae_encoder_model,
                             const Tensor& vae_encoder_weights,
                             const std::string& vae_decoder_model,
                             const Tensor& vae_decoder_weights,
                             const Config& vae_decoder_config,
                             const std::string& device,
                             const ov::AnyMap& properties)
    : AutoencoderKL(vae_encoder_model,
                    vae_encoder_weights,
                    vae_decoder_model,
                    vae_decoder_weights,
                    vae_decoder_config) {
    compile(device, *extract_adapters_from_properties(properties));
}

AutoencoderKL::AutoencoderKL(const AutoencoderKL& rhs) = default;

AutoencoderKL AutoencoderKL::clone() {
    OPENVINO_ASSERT((m_decoder_model != nullptr) ^ static_cast<bool>(m_decoder_request), "AutoencoderKL must have exactly one of m_decoder_model or m_decoder_request initialized");  // encoder is optional

    AutoencoderKL cloned = *this;

    // Required, decoder model
    if (m_decoder_model) {
        cloned.m_decoder_model = m_decoder_model->clone();
    } else {
        cloned.m_decoder_request = m_decoder_request.get_compiled_model().create_infer_request();
    }

    // Optional encoder model
    if (m_encoder_model) {
        cloned.m_encoder_model = m_encoder_model->clone();
    } else {
        // Might not be defined
        if (m_encoder_request) {
            cloned.m_encoder_request = m_encoder_request.get_compiled_model().create_infer_request();
        }
    }

    return cloned;
}

AutoencoderKL& AutoencoderKL::reshape(int batch_size, int height, int width) {
    OPENVINO_ASSERT(m_decoder_model, "Model has been already compiled. Cannot reshape already compiled model");

    const size_t vae_scale_factor = get_vae_scale_factor();

    OPENVINO_ASSERT((height % vae_scale_factor == 0 || height < 0) &&
            (width % vae_scale_factor == 0 || width < 0), "Both 'width' and 'height' must be divisible by ",
            vae_scale_factor);

    if (m_encoder_model) {
        ov::PartialShape input_shape = m_encoder_model->input(0).get_partial_shape();
        std::map<size_t, ov::PartialShape> idx_to_shape{{0, {batch_size, input_shape[1], height, width}}};
        m_encoder_model->reshape(idx_to_shape);
    }

    height /= vae_scale_factor;
    width /= vae_scale_factor;

    ov::PartialShape input_shape = m_decoder_model->input(0).get_partial_shape();
    std::map<size_t, ov::PartialShape> idx_to_shape{{0, {batch_size, input_shape[1], height, width}}};
    m_decoder_model->reshape(idx_to_shape);

    return *this;
}

AutoencoderKL& AutoencoderKL::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_decoder_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::Core core = utils::singleton_core();

    std::optional<AdapterConfig> unused;
    auto filtered_properties = extract_adapters_from_properties(properties, &unused);

    if (m_encoder_model) {
        ov::CompiledModel encoder_compiled_model = core.compile_model(m_encoder_model, device, handle_scale_factor(m_encoder_model, device, *filtered_properties));
        ov::genai::utils::print_compiled_model_properties(encoder_compiled_model, "Auto encoder KL encoder model");
        m_encoder_request = encoder_compiled_model.create_infer_request();
        // release the original model
        m_encoder_model.reset();
    }

    ov::CompiledModel decoder_compiled_model = core.compile_model(m_decoder_model, device, handle_scale_factor(m_decoder_model, device, *filtered_properties));
    ov::genai::utils::print_compiled_model_properties(decoder_compiled_model, "Auto encoder KL decoder model");
    m_decoder_request = decoder_compiled_model.create_infer_request();
    // release the original model
    m_decoder_model.reset();

    return *this;
}

ov::Tensor AutoencoderKL::decode(ov::Tensor latent) {
    OPENVINO_ASSERT(m_decoder_request, "VAE decoder model must be compiled first. Cannot infer non-compiled model");

    m_decoder_request.set_input_tensor(latent);
    m_decoder_request.infer();
    return m_decoder_request.get_output_tensor();
}

ov::Tensor AutoencoderKL::encode(ov::Tensor image, std::shared_ptr<Generator> generator) {
    OPENVINO_ASSERT(m_encoder_request || m_encoder_model, "AutoencoderKL is created without 'VAE encoder' capability. Please, pass extra argument to constructor to create 'VAE encoder'");
    OPENVINO_ASSERT(m_encoder_request, "VAE encoder model must be compiled first. Cannot infer non-compiled model");

    m_encoder_request.set_input_tensor(image);
    m_encoder_request.infer();

    ov::Tensor output = m_encoder_request.get_output_tensor(), latent;

    ov::CompiledModel compiled_model = m_encoder_request.get_compiled_model();
    auto outputs = compiled_model.outputs();
    OPENVINO_ASSERT(outputs.size() == 1, "AutoencoderKL encoder model is expected to have a single output");

    const std::string output_name = outputs[0].get_any_name();
    if (output_name == "latent_sample") {
        latent = output;
    } else if (output_name == "latent_parameters") {
        latent = DiagonalGaussianDistribution(output).sample(generator);
    } else {
        OPENVINO_THROW("Unexpected output name for AutoencoderKL encoder '", output_name, "'");
    }

    // apply shift and scaling factor
    float * latent_data = latent.data<float>();
    for (size_t i = 0; i < latent.get_size(); ++i) {
        latent_data[i] = (latent_data[i] - m_config.shift_factor) * m_config.scaling_factor;
    }

    return latent;
}

const AutoencoderKL::Config& AutoencoderKL::get_config() const {
    return m_config;
}

size_t AutoencoderKL::get_vae_scale_factor() const {
    return std::pow(2, m_config.block_out_channels.size() - 1);
}

void AutoencoderKL::merge_vae_image_post_processing() const {
    ov::preprocess::PrePostProcessor ppp(m_decoder_model);

    // scale and shift input before VAE decoder
    if (m_config.scaling_factor != 1.0f)
        ppp.input().preprocess().scale(m_config.scaling_factor);
    if (m_config.shift_factor != 0.0f)
        ppp.input().preprocess().mean(-m_config.shift_factor);

    // apply VaeImageProcessor normalization steps
    // https://github.com/huggingface/diffusers/blob/v0.30.1/src/diffusers/image_processor.py#L159
    ppp.output().postprocess().custom([](const ov::Output<ov::Node>& port) {
        auto constant_0_5 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.5f);
        auto constant_255 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 255.0f);
        auto scaled_0_5 = std::make_shared<ov::op::v1::Multiply>(port, constant_0_5);
        auto added_0_5 = std::make_shared<ov::op::v1::Add>(scaled_0_5, constant_0_5);
        auto clamped = std::make_shared<ov::op::v0::Clamp>(added_0_5, 0.0f, 1.0f);
        return std::make_shared<ov::op::v1::Multiply>(clamped, constant_255);
    });
    ppp.output().postprocess().convert_element_type(ov::element::u8);
    // layout conversion
    // https://github.com/huggingface/diffusers/blob/v0.30.1/src/diffusers/image_processor.py#L144
    ppp.output().model().set_layout("NCHW");
    ppp.output().tensor().set_layout("NHWC");

    ppp.build();
}

void AutoencoderKL::export_model(const std::filesystem::path& blob_path) {
    OPENVINO_ASSERT(m_decoder_request, "VAE decoder model must be compiled first. Cannot export non-compiled model");
    auto decoder_compiled_model = m_decoder_request.get_compiled_model();
    ov::genai::utils::export_model(decoder_compiled_model, blob_path / "vae_decoder" / "openvino_model.blob");

    if (m_encoder_request) {
        auto encoder_compiled_model = m_encoder_request.get_compiled_model();
        ov::genai::utils::export_model(encoder_compiled_model, blob_path / "vae_encoder" / "openvino_model.blob");
    }
}

void AutoencoderKL::import_model(const std::filesystem::path& blob_path, const std::string& device, const ov::AnyMap& properties) {
    auto decoder_compiled_model = utils::import_model(blob_path / "vae_decoder" / "openvino_model.blob", device, properties);
    ov::genai::utils::print_compiled_model_properties(decoder_compiled_model, "Auto encoder KL decoder model");
    m_decoder_request = decoder_compiled_model.create_infer_request();

    if (std::filesystem::exists(blob_path / "vae_encoder" / "openvino_model.blob")) {
        auto encoder_compiled_model = utils::import_model(blob_path / "vae_encoder" / "openvino_model.blob", device, properties);
        ov::genai::utils::print_compiled_model_properties(encoder_compiled_model, "Auto encoder KL encoder model");
        m_encoder_request = encoder_compiled_model.create_infer_request();
    }
}

} // namespace genai
} // namespace ov
