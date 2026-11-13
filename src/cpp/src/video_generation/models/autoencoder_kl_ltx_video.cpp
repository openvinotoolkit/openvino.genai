// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/autoencoder_kl_ltx_video.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
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

class DiagonalGaussianDistribution {
public:
    explicit DiagonalGaussianDistribution(ov::Tensor parameters) : m_params(std::move(parameters)) {
        OPENVINO_ASSERT(m_params.get_element_type() == ov::element::f32,
            "DiagonalGaussianDistribution requires f32 encoder output, got ",
            m_params.get_element_type());

        const ov::Shape& full_shape = m_params.get_shape();
        OPENVINO_ASSERT(full_shape.size() >= 2, "Parameters tensor rank must be at least 2");
        OPENVINO_ASSERT(full_shape[1] % 2 == 0, "Channel dimension must be even to split mean and logvar");

        m_channels = full_shape[1] / 2;
        m_spatial = 1;
        for (size_t i = 2; i < full_shape.size(); ++i)
            m_spatial *= full_shape[i];

        ov::Shape std_shape = full_shape;
        std_shape[1] = m_channels;
        m_std = ov::Tensor(m_params.get_element_type(), std_shape);

        const float* src = m_params.data<float>();
        float* std_data = m_std.data<float>();
        const size_t batch = full_shape[0];

        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < m_channels; ++c) {
                const size_t lvar_off = (b * full_shape[1] + m_channels + c) * m_spatial;
                const size_t dst_off  = (b * m_channels + c) * m_spatial;
                for (size_t s = 0; s < m_spatial; ++s) {
                    const float logvar = std::min(std::max(src[lvar_off + s], -30.0f), 20.0f);
                    std_data[dst_off + s] = std::exp(0.5f * logvar);
                }
            }
        }
    }

    ov::Tensor sample(std::shared_ptr<ov::genai::Generator> generator) const {
        OPENVINO_ASSERT(generator, "Generator must not be nullptr");

        ov::Shape sample_shape = m_params.get_shape();
        sample_shape[1] = m_channels;
        ov::Tensor result = generator->randn_tensor(sample_shape);
        OPENVINO_ASSERT(result.get_element_type() == ov::element::f32,
            "Generator::randn_tensor() must return an f32 tensor, got ",
            result.get_element_type());

        const float* params_data = m_params.data<float>();
        const float* std_data = m_std.data<float>();
        float* result_data = result.data<float>();
        const size_t batch = m_params.get_shape()[0];
        const size_t full_channels = m_params.get_shape()[1];

        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < m_channels; ++c) {
                const size_t mean_off = (b * full_channels + c) * m_spatial;
                const size_t dst_off  = (b * m_channels + c) * m_spatial;
                for (size_t s = 0; s < m_spatial; ++s) {
                    result_data[dst_off + s] = params_data[mean_off + s] + std_data[dst_off + s] * result_data[dst_off + s];
                }
            }
        }

        return result;
    }

private:
    ov::Tensor m_params, m_std;
    size_t m_channels, m_spatial;
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
        OPENVINO_ASSERT(encoder_compiled_model.outputs().size() == 1, "AutoencoderKLLTXVideo encoder model is expected to have a single output");
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

AutoencoderKLLTXVideo AutoencoderKLLTXVideo::clone() {
    OPENVINO_ASSERT((m_decoder_model != nullptr) ^ static_cast<bool>(m_decoder_request),
                    "AutoencoderKLLTXVideo must have exactly one of m_decoder_model or m_decoder_request initialized");

    AutoencoderKLLTXVideo cloned = *this;

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

AutoencoderKLLTXVideo& AutoencoderKLLTXVideo::reshape(int64_t batch_size,
                                                      int64_t num_frames,
                                                      int64_t height,
                                                      int64_t width) {
    OPENVINO_ASSERT(m_decoder_model, "Model has been already compiled. Cannot reshape already compiled model");
    OPENVINO_ASSERT(height > 0, "Height must be positive");
    OPENVINO_ASSERT(height % 32 == 0, "Height must be divisible by 32 but got ", height);
    OPENVINO_ASSERT(width > 0, "Width must be positive");
    OPENVINO_ASSERT(width % 32 == 0, "Width must be divisible by 32 but got ", width);

    if (m_encoder_model) {
        ov::PartialShape input_shape = m_encoder_model->input(0).get_partial_shape();
        OPENVINO_ASSERT(input_shape.rank().is_static() && input_shape.rank().get_length() == 5,
            "AutoencoderKLLTXVideo encoder input must be rank 5 [B, C, F, H, W], got rank ",
            input_shape.rank());
        // The encoder receives a single conditioning image, not batch_size videos, so its batch
        // dimension is left unchanged.
        std::map<size_t, ov::PartialShape> idx_to_shape{{0, {input_shape[0], input_shape[1], 1, height, width}}};
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

    int64_t latent_num_frames = ((num_frames - 1) / temporal_compression_ratio + 1) / m_transformer_patch_size_t;
    int64_t latent_height = height / (spatial_compression_ratio * m_transformer_patch_size);
    int64_t latent_width  = width  / (spatial_compression_ratio * m_transformer_patch_size);

    ov::PartialShape input_shape = m_decoder_model->input(0).get_partial_shape();
    std::map<size_t, ov::PartialShape> idx_to_shape{{0, {batch_size, input_shape[1], latent_num_frames, latent_height, latent_width}}};
    m_decoder_model->reshape(idx_to_shape);

    return *this;
}

ov::Tensor AutoencoderKLLTXVideo::decode(const ov::Tensor& latent) {
    OPENVINO_ASSERT(m_decoder_request, "VAE decoder model must be compiled first. Cannot infer non-compiled model");

    m_decoder_request.set_input_tensor(latent);
    m_decoder_request.infer();
    return m_decoder_request.get_output_tensor();
}

ov::Tensor AutoencoderKLLTXVideo::decode(const ov::Tensor& latent,
                                         size_t tile_sample_min_num_frames,
                                         size_t tile_sample_stride_num_frames) {
    if (tile_sample_min_num_frames == 0) {
        return decode(latent);
    }
    OPENVINO_ASSERT(m_decoder_request, "VAE decoder model must be compiled first. Cannot infer non-compiled model");

    const size_t temporal_compression_ratio = static_cast<size_t>(
        m_config.patch_size_t *
        std::pow(2,
                 std::accumulate(m_config.spatio_temporal_scaling.begin(), m_config.spatio_temporal_scaling.end(), 0)));
    OPENVINO_ASSERT(tile_sample_min_num_frames % temporal_compression_ratio == 0 &&
                        tile_sample_stride_num_frames % temporal_compression_ratio == 0,
                    "'tile_sample_min_num_frames' and 'tile_sample_stride_num_frames' must be divisible by ",
                    temporal_compression_ratio, " but got ", tile_sample_min_num_frames, " and ",
                    tile_sample_stride_num_frames);
    OPENVINO_ASSERT(tile_sample_stride_num_frames > 0 && tile_sample_stride_num_frames <= tile_sample_min_num_frames,
                    "'tile_sample_stride_num_frames' must be in (0, 'tile_sample_min_num_frames'] but got ",
                    tile_sample_stride_num_frames);

    const ov::Shape latent_shape = latent.get_shape();
    OPENVINO_ASSERT(latent_shape.size() == 5, "Latent video expected to be [B, C, F, H, W]");
    const size_t batch = latent_shape[0], latent_frames = latent_shape[2];
    // Sizes in latent frames. A chunk reads one latent frame more: its first latent frame gives one video frame only.
    const size_t chunk = tile_sample_min_num_frames / temporal_compression_ratio;
    const size_t step = tile_sample_stride_num_frames / temporal_compression_ratio;
    if (latent_frames <= chunk) {
        return decode(latent);
    }
    OPENVINO_ASSERT(m_decoder_request.get_compiled_model().input().get_partial_shape()[2].is_dynamic(),
                    "Temporal VAE tiling requires a decoder with a dynamic number of frames, but it was reshaped "
                    "to a static shape. Disable tiling or don't call reshape().");

    const size_t num_frames = (latent_frames - 1) * temporal_compression_ratio + 1;
    const size_t keep = tile_sample_stride_num_frames;  // video frames a chunk contributes
    const size_t overlap = tile_sample_min_num_frames - tile_sample_stride_num_frames;
    // one channel of one latent frame in bytes
    const size_t latent_plane_bytes = latent_shape[3] * latent_shape[4] * latent.get_element_type().size();
    const uint8_t* latent_data = static_cast<const uint8_t*>(latent.data());

    ov::Tensor video;
    // The last frames of the previous chunk [B, tail_frames, H, W, C]. The infer request reuses its output
    // buffer, so they are copied. Nothing else of the previous chunk is needed for blending.
    std::vector<uint8_t> tail;
    size_t tail_frames = 0, frame_size = 0, position = 0;

    for (size_t begin = 0; begin < latent_frames; begin += step) {
        const size_t chunk_latent_frames = std::min(chunk + 1, latent_frames - begin);
        if (begin > 0 && chunk_latent_frames == 1) {
            // A single latent frame decodes to the frame which is dropped below
            break;
        }

        ov::Shape chunk_shape = latent_shape;
        chunk_shape[2] = chunk_latent_frames;
        ov::Tensor latent_chunk(latent.get_element_type(), chunk_shape);
        uint8_t* chunk_data = static_cast<uint8_t*>(latent_chunk.data());
        for (size_t bc = 0; bc < batch * latent_shape[1]; ++bc) {
            std::memcpy(chunk_data + bc * chunk_latent_frames * latent_plane_bytes,
                        latent_data + (bc * latent_frames + begin) * latent_plane_bytes,
                        chunk_latent_frames * latent_plane_bytes);
        }

        m_decoder_request.set_input_tensor(latent_chunk);
        m_decoder_request.infer();
        ov::Tensor decoded = m_decoder_request.get_output_tensor();
        const ov::Shape decoded_shape = decoded.get_shape();
        OPENVINO_ASSERT(decoded.get_element_type() == ov::element::u8 && decoded_shape.size() == 5 &&
                            decoded_shape[1] == (chunk_latent_frames - 1) * temporal_compression_ratio + 1,
                        "Unexpected VAE decoder output ", decoded.get_element_type(), " ", decoded_shape);

        // Only the first chunk starts with the first frame of the video. In other chunks the frame decoded from
        // the first latent frame repeats the last frame the previous chunk contributed, so it's dropped.
        const size_t skip = begin == 0 ? 0 : 1;
        const size_t chunk_frames = decoded_shape[1] - skip;
        const size_t blend_frames = std::min({tail_frames, chunk_frames, overlap});
        const size_t write_frames = std::min(keep + 1 - skip, std::min(chunk_frames, num_frames - position));
        const size_t next_tail_frames = std::min(chunk_frames, overlap);
        if (begin == 0) {
            frame_size = decoded_shape[2] * decoded_shape[3] * decoded_shape[4];
            video = ov::Tensor(ov::element::u8,
                               {batch, num_frames, decoded_shape[2], decoded_shape[3], decoded_shape[4]});
        }
        std::vector<uint8_t> next_tail(batch * next_tail_frames * frame_size);

        for (size_t b = 0; b < batch; ++b) {
            uint8_t* current = decoded.data<uint8_t>() + (b * decoded_shape[1] + skip) * frame_size;
            // Linear cross-fade: frame k is previous * (1 - k / blend_frames) + current * (k / blend_frames)
            const uint8_t* previous = tail.data() + (b * tail_frames + tail_frames - blend_frames) * frame_size;
            for (size_t k = 0; k < blend_frames; ++k) {
                const double weight = static_cast<double>(k) / blend_frames;
                const float previous_weight = static_cast<float>(1.0 - weight);
                const float current_weight = static_cast<float>(weight);
                uint8_t* current_frame = current + k * frame_size;
                const uint8_t* previous_frame = previous + k * frame_size;
                for (size_t i = 0; i < frame_size; ++i) {
                    current_frame[i] = static_cast<uint8_t>(
                        std::nearbyint(previous_frame[i] * previous_weight + current_frame[i] * current_weight));
                }
            }
            std::memcpy(video.data<uint8_t>() + (b * num_frames + position) * frame_size,
                        current,
                        write_frames * frame_size);
            std::memcpy(next_tail.data() + b * next_tail_frames * frame_size,
                        current + (chunk_frames - next_tail_frames) * frame_size,
                        next_tail_frames * frame_size);
        }
        position += write_frames;
        tail = std::move(next_tail);
        tail_frames = next_tail_frames;
    }
    OPENVINO_ASSERT(position == num_frames, "Temporal VAE tiling produced ", position, " frames instead of ", num_frames);

    return video;
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
    const std::string output_name = m_encoder_request.get_compiled_model().outputs()[0].get_any_name();

    if (output_name == "latent_sample") {
        // Copy to an owned tensor — get_output_tensor() aliases the infer request's
        // internal buffer, which would be overwritten on the next encode() call.
        latent = ov::Tensor(output.get_element_type(), output.get_shape());
        output.copy_to(latent);
    } else if (output_name == "latent_parameters") {
        OPENVINO_ASSERT(generator,
            "AutoencoderKLLTXVideo::encode requires a non-null generator when encoder output is "
            "'latent_parameters', because latent sampling is performed from distribution parameters. "
            "A generator is not required when encoder output is 'latent_sample'.");
        latent = DiagonalGaussianDistribution(output).sample(generator);
    } else {
        OPENVINO_ASSERT(false, "Unexpected output name for AutoencoderKLLTXVideo encoder '", output_name, "'");
    }

    // inverse of denormalize_latents used in the decode path
    const ov::Shape shape = latent.get_shape();
    OPENVINO_ASSERT(shape.size() == 5, "Encoder output expected to be [B, C, F, H, W]");
    OPENVINO_ASSERT(latent.get_element_type() == ov::element::f32,
        "Latent normalization requires f32, got ", latent.get_element_type());
    const size_t B = shape[0], C = shape[1], spatial = shape[2] * shape[3] * shape[4];

    const auto& mean = m_config.latents_mean_data;
    const auto& std_data = m_config.latents_std_data;
    OPENVINO_ASSERT(mean.size() == C && std_data.size() == C,
        "Config latents_mean/std size (", mean.size(), ") does not match latent channels (", C, ")");

    float* latent_data = latent.data<float>();
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
