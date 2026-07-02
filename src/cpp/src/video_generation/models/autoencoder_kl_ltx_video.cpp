// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/autoencoder_kl_ltx_video.hpp"

#include <fstream>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <memory>
#include <map>
#include <numeric>

#include "openvino/runtime/core.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/constant.hpp"

#include "utils.hpp"
#include "json_utils.hpp"
#include "lora/helper.hpp"

using namespace ov::genai;

namespace {

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

std::string find_input_name_by_candidates(const ov::CompiledModel& compiled_model,
                                          const std::vector<std::string>& candidates) {
    // Resolve by candidate priority first (e.g. prefer `timestep` over fallback `temb`).
    for (const auto& candidate : candidates) {
        for (const auto& input : compiled_model.inputs()) {
            const auto& names = input.get_names();
            if (names.find(candidate) != names.end()) {
                return candidate;
            }
            const std::string input_name = input.get_any_name();
            if (input_name == candidate) {
                return input_name;
            }
        }
    }
    return "";
}

std::string find_latent_input_name(const ov::CompiledModel& compiled_model) {
    const std::vector<std::string> preferred_names{"sample", "latent", "latents", "hidden_states", "z"};
    std::string name = find_input_name_by_candidates(compiled_model, preferred_names);
    if (!name.empty()) {
        return name;
    }

    for (const auto& input : compiled_model.inputs()) {
        const ov::PartialShape shape = input.get_partial_shape();
        if (shape.rank().is_dynamic()) {
            continue;
        }
        if (shape.rank().get_length() == 5) {
            return input.get_any_name();
        }
    }

    OPENVINO_THROW("Failed to identify VAE latent input tensor name for LTX decoder IR");
}

ov::Tensor create_scalar_or_batch_tensor(ov::InferRequest& request,
                                         const std::string& input_name,
                                         float value,
                                         size_t batch_size) {
    const ov::Tensor input_tensor = request.get_tensor(input_name);
    const ov::Shape input_shape = input_tensor.get_shape();
    const ov::element::Type input_type = input_tensor.get_element_type();

    auto get_integral_value = [&]() -> int64_t {
        double integral_part = 0.0;
        const double fractional_part = std::modf(static_cast<double>(value), &integral_part);
        if (std::fabs(fractional_part) > 1e-6) {
            OPENVINO_THROW("Conditioning input '", input_name,
                           "' expects an integer value, but got non-integer value ", value);
        }
        return static_cast<int64_t>(integral_part);
    };

    auto write_scalar_value = [&](ov::Tensor& dst) {
        if (input_type == ov::element::f32) {
            dst.data<float>()[0] = value;
        } else if (input_type == ov::element::f16) {
            dst.data<ov::float16>()[0] = static_cast<ov::float16>(value);
        } else if (input_type == ov::element::bf16) {
            dst.data<ov::bfloat16>()[0] = static_cast<ov::bfloat16>(value);
        } else if (input_type == ov::element::f64) {
            dst.data<double>()[0] = static_cast<double>(value);
        } else if (input_type == ov::element::i32) {
            dst.data<int32_t>()[0] = static_cast<int32_t>(get_integral_value());
        } else if (input_type == ov::element::i64) {
            dst.data<int64_t>()[0] = get_integral_value();
        } else if (input_type == ov::element::u32) {
            dst.data<uint32_t>()[0] = static_cast<uint32_t>(get_integral_value());
        } else if (input_type == ov::element::u64) {
            dst.data<uint64_t>()[0] = static_cast<uint64_t>(get_integral_value());
        } else {
            OPENVINO_THROW("Unsupported input element type for conditioning scalar input '", input_name, "'");
        }
    };

    ov::Tensor tensor;
    if (input_shape.empty()) {
        tensor = ov::Tensor(input_type, {});
        write_scalar_value(tensor);
        return tensor;
    }

    ov::Shape normalized_shape = input_shape;
    size_t element_count = 1;
    for (size_t idx = 0; idx < normalized_shape.size(); ++idx) {
        size_t dim = normalized_shape[idx];
        if (dim == 0) {
            // Dynamic dimensions can be materialized as 0 in request tensor metadata.
            // Use batch size for the leading dimension and 1 for trailing dimensions.
            dim = (idx == 0) ? batch_size : 1;
            normalized_shape[idx] = dim;
        }
        element_count *= dim;
    }
    OPENVINO_ASSERT(element_count > 0, "Conditioning input '", input_name, "' has zero element count");

    tensor = ov::Tensor(input_type, normalized_shape);
    if (input_type == ov::element::f32) {
        std::fill_n(tensor.data<float>(), element_count, value);
    } else if (input_type == ov::element::f16) {
        std::fill_n(tensor.data<ov::float16>(), element_count, static_cast<ov::float16>(value));
    } else if (input_type == ov::element::bf16) {
        std::fill_n(tensor.data<ov::bfloat16>(), element_count, static_cast<ov::bfloat16>(value));
    } else if (input_type == ov::element::f64) {
        std::fill_n(tensor.data<double>(), element_count, static_cast<double>(value));
    } else if (input_type == ov::element::i32) {
        std::fill_n(tensor.data<int32_t>(), element_count, static_cast<int32_t>(get_integral_value()));
    } else if (input_type == ov::element::i64) {
        std::fill_n(tensor.data<int64_t>(), element_count, get_integral_value());
    } else if (input_type == ov::element::u32) {
        std::fill_n(tensor.data<uint32_t>(), element_count, static_cast<uint32_t>(get_integral_value()));
    } else if (input_type == ov::element::u64) {
        std::fill_n(tensor.data<uint64_t>(), element_count, static_cast<uint64_t>(get_integral_value()));
    } else {
        OPENVINO_THROW("Unsupported input element type for conditioning input '", input_name, "'");
    }

    const size_t first_dim = normalized_shape.front();
    if (batch_size > 1 && first_dim != 1 && first_dim != batch_size) {
        OPENVINO_THROW("Unexpected first dimension for conditioning input '", input_name,
                       "': expected 1 or batch size ", batch_size, ", got ", first_dim);
    }

    return tensor;
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

    // TODO: for img2video
    // if (m_encoder_model) {...}

    ov::CompiledModel decoder_compiled_model = core.compile_model(m_decoder_model, device, handle_scale_factor(m_decoder_model, device, *filtered_properties));
    ov::genai::utils::print_compiled_model_properties(decoder_compiled_model, "Auto encoder KL LTX video decoder model");
    m_latent_input_name = find_latent_input_name(decoder_compiled_model);
    if (m_config.timestep_conditioning) {
        m_decode_timestep_input_name =
            find_input_name_by_candidates(decoder_compiled_model, {"decode_timestep", "timestep", "temb"});
        m_decode_noise_scale_input_name =
            find_input_name_by_candidates(decoder_compiled_model, {"decode_noise_scale", "noise_scale"});
    }

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

    // TODO: for img2video
    // if (m_encoder_model) {...}

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

    std::map<size_t, ov::PartialShape> idx_to_shape;
    std::optional<size_t> latent_input_idx;
    for (size_t input_idx = 0; input_idx < m_decoder_model->inputs().size(); ++input_idx) {
        const ov::Output<const ov::Node>& input = m_decoder_model->input(input_idx);
        ov::PartialShape input_shape = input.get_partial_shape();
        if (input_shape.rank().is_dynamic() || input_shape.rank().get_length() == 0) {
            continue;
        }
        input_shape[0] = batch_size;
        idx_to_shape[input_idx] = input_shape;

        if (!m_latent_input_name.empty() && input.get_names().count(m_latent_input_name) > 0) {
            latent_input_idx = input_idx;
        }
    }
    OPENVINO_ASSERT(!m_decoder_model->inputs().empty(), "Decoder model has no inputs");
    const size_t resolved_latent_input_idx = latent_input_idx.value_or(0);
    const ov::PartialShape latent_shape = m_decoder_model->input(resolved_latent_input_idx).get_partial_shape();
    idx_to_shape[resolved_latent_input_idx] = {batch_size, latent_shape[1], num_frames, height, width};
    m_decoder_model->reshape(idx_to_shape);

    return *this;
}

ov::Tensor AutoencoderKLLTXVideo::decode(const ov::Tensor& latent) {
    return decode(latent, std::nullopt, std::nullopt);
}

ov::Tensor AutoencoderKLLTXVideo::decode(const ov::Tensor& latent,
                                         const std::optional<float>& decode_timestep,
                                         const std::optional<float>& decode_noise_scale) {
    OPENVINO_ASSERT(m_decoder_request, "VAE decoder model must be compiled first. Cannot infer non-compiled model");

    const size_t batch_size = latent.get_shape().at(0);

    OPENVINO_ASSERT(!m_latent_input_name.empty(),
                    "VAE latent input name is not initialized. Decoder compile stage failed to cache input names.");

    m_decoder_request.set_tensor(m_latent_input_name, latent);

    if (!m_config.timestep_conditioning) {
        m_decoder_request.infer();
        return m_decoder_request.get_output_tensor();
    }

    OPENVINO_ASSERT(!m_decode_timestep_input_name.empty(),
                    "VAE config enables 'timestep_conditioning', but 'decode timestep' input is missing in the IR.");

    const float decode_timestep_value = decode_timestep.value_or(0.0f);
    const float decode_noise_scale_value = decode_noise_scale.value_or(decode_timestep_value);

    m_decoder_request.set_tensor(
        m_decode_timestep_input_name,
        create_scalar_or_batch_tensor(m_decoder_request, m_decode_timestep_input_name, decode_timestep_value, batch_size)
    );

    if (!m_decode_noise_scale_input_name.empty()) {
        m_decoder_request.set_tensor(
            m_decode_noise_scale_input_name,
            create_scalar_or_batch_tensor(
                m_decoder_request,
                m_decode_noise_scale_input_name,
                decode_noise_scale_value,
                batch_size)
        );
    }

    m_decoder_request.infer();
    return m_decoder_request.get_output_tensor();
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

    // (x / 2 + 0.5) -> clamp(0..1) -> *255 -> round() -> u8
    ppp.output().postprocess().custom([](const ov::Output<ov::Node>& port) {
        auto c_0_5  = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.5f);
        auto c_255  = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 255.0f);

        auto scaled = std::make_shared<ov::op::v1::Multiply>(port, c_0_5);
        auto shifted = std::make_shared<ov::op::v1::Add>(scaled, c_0_5);
        auto clamped = std::make_shared<ov::op::v0::Clamp>(shifted, 0.0f, 1.0f);

        auto scaled_255 = std::make_shared<ov::op::v1::Multiply>(clamped, c_255);
        return std::make_shared<ov::op::v5::Round>(scaled_255, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    });

    ppp.output().postprocess().convert_element_type(ov::element::u8);

    // [B, C, F, H, W] -> [B, F, H, W, C]
    ppp.output().model().set_layout("NCDHW");
    ppp.output().tensor().set_layout("NDHWC");

    ppp.build();
}
