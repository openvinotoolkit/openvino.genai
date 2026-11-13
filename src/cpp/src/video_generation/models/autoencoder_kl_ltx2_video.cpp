// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/autoencoder_kl_ltx2_video.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>

#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/round.hpp"

#include "json_utils.hpp"
#include "utils.hpp"

using namespace ov::genai;

namespace {

// How many video frames to drop from the start of a decoded chunk of latent frames which starts at latent frame
// 'chunk_begin'. The first latent frame of any chunk decodes to one video frame. In the first chunk it's the first
// frame of the video. In a later chunk it's video frame 8 * chunk_begin, which the previous chunk has already
// produced, so it's dropped and the chunk starts at video frame 8 * chunk_begin + 1.
size_t leading_frames_to_drop(size_t chunk_begin) {
    return chunk_begin == 0 ? 0 : 1;
}

}  // namespace

AutoencoderKLLTX2Video::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "latent_channels", latent_channels);
    read_json_param(data, "scaling_factor", scaling_factor);
    read_json_param(data, "timestep_conditioning", timestep_conditioning);
    read_json_param(data, "latents_mean_data", latents_mean_data);
    read_json_param(data, "latents_std_data", latents_std_data);

    int64_t spatial = 0, temporal = 0;
    read_json_param(data, "spatial_compression_ratio", spatial);
    read_json_param(data, "temporal_compression_ratio", temporal);
    if (spatial == 0 || temporal == 0) {
        std::vector<bool> spatio_temporal_scaling;
        int64_t patch_size = 0, patch_size_t = 0;
        read_json_param(data, "spatio_temporal_scaling", spatio_temporal_scaling);
        read_json_param(data, "patch_size", patch_size);
        read_json_param(data, "patch_size_t", patch_size_t);
        if (patch_size > 0 && patch_size_t > 0) {
            const auto compression_factor =
                std::pow(2, std::accumulate(spatio_temporal_scaling.begin(), spatio_temporal_scaling.end(), 0));
            spatial = patch_size * compression_factor;
            temporal = patch_size_t * compression_factor;
        }
    }
    // Keep the member defaults when the config carries neither the ratios nor the patch keys
    if (spatial > 0) {
        spatial_compression_ratio = spatial;
    }
    if (temporal > 0) {
        temporal_compression_ratio = temporal;
    }

    if (latents_mean_data.empty()) {
        latents_mean_data.assign(latent_channels, 0.0f);
    }
    if (latents_std_data.empty()) {
        latents_std_data.assign(latent_channels, 1.0f);
    }
}

AutoencoderKLLTX2Video::AutoencoderKLLTX2Video(const std::filesystem::path& vae_decoder_path)
    : m_config(vae_decoder_path / "config.json") {
    m_decoder_model = utils::singleton_core().read_model(vae_decoder_path / "openvino_model.xml");
    merge_vae_video_post_processing();
}

AutoencoderKLLTX2Video::AutoencoderKLLTX2Video(const std::filesystem::path& vae_decoder_path,
                                               const std::string& device,
                                               const ov::AnyMap& properties)
    : AutoencoderKLLTX2Video(vae_decoder_path) {
    compile(device, properties);
}

AutoencoderKLLTX2Video::AutoencoderKLLTX2Video(const AutoencoderKLLTX2Video&) = default;

AutoencoderKLLTX2Video AutoencoderKLLTX2Video::clone() {
    OPENVINO_ASSERT((m_decoder_model != nullptr) ^ static_cast<bool>(m_decoder_request),
                    "AutoencoderKLLTX2Video must have exactly one of m_decoder_model or m_decoder_request initialized");

    AutoencoderKLLTX2Video cloned = *this;

    if (m_decoder_model) {
        cloned.m_decoder_model = m_decoder_model->clone();
    } else {
        cloned.m_decoder_request = m_decoder_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

const AutoencoderKLLTX2Video::Config& AutoencoderKLLTX2Video::get_config() const {
    return m_config;
}

AutoencoderKLLTX2Video& AutoencoderKLLTX2Video::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_decoder_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_decoder_model, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Auto encoder KL LTX2 video decoder model");
    m_decoder_request = compiled_model.create_infer_request();
    m_decoder_model.reset();

    return *this;
}

AutoencoderKLLTX2Video& AutoencoderKLLTX2Video::reshape(int64_t batch_size,
                                                        int64_t num_frames,
                                                        int64_t height,
                                                        int64_t width) {
    OPENVINO_ASSERT(m_decoder_model, "Model has been already compiled. Cannot reshape already compiled model");

    const int64_t latent_num_frames = (num_frames - 1) / m_config.temporal_compression_ratio + 1;
    const int64_t latent_height = height / m_config.spatial_compression_ratio;
    const int64_t latent_width = width / m_config.spatial_compression_ratio;

    ov::PartialShape input_shape = m_decoder_model->input(0).get_partial_shape();
    std::map<size_t, ov::PartialShape> idx_to_shape{
        {0, {batch_size, input_shape[1], latent_num_frames, latent_height, latent_width}}};
    m_decoder_model->reshape(idx_to_shape);

    return *this;
}

ov::Tensor AutoencoderKLLTX2Video::decode(const ov::Tensor& latent) {
    OPENVINO_ASSERT(m_decoder_request, "VAE decoder model must be compiled first. Cannot infer non-compiled model");

    // Write into an owned tensor - the request's internal buffer would be overwritten on the next decode() call
    const auto& output = m_decoder_request.get_compiled_model().output(0);
    ov::Shape video_shape;
    if (output.get_partial_shape().is_static()) {
        video_shape = output.get_shape();
    } else {
        const ov::Shape latent_shape = latent.get_shape();  // [B, C, F, H, W]
        video_shape = {latent_shape[0],
                       (latent_shape[2] - 1) * static_cast<size_t>(m_config.temporal_compression_ratio) + 1,
                       latent_shape[3] * static_cast<size_t>(m_config.spatial_compression_ratio),
                       latent_shape[4] * static_cast<size_t>(m_config.spatial_compression_ratio),
                       3};
    }
    ov::Tensor video(output.get_element_type(), video_shape);
    m_decoder_request.set_input_tensor(latent);
    m_decoder_request.set_output_tensor(video);
    m_decoder_request.infer();
    return video;
}

ov::Tensor AutoencoderKLLTX2Video::decode(const ov::Tensor& latent,
                                          size_t tile_sample_min_num_frames,
                                          size_t tile_sample_stride_num_frames) {
    if (tile_sample_min_num_frames == 0) {
        return decode(latent);
    }
    OPENVINO_ASSERT(m_decoder_request, "VAE decoder model must be compiled first. Cannot infer non-compiled model");

    const size_t temporal_compression_ratio = static_cast<size_t>(m_config.temporal_compression_ratio);
    OPENVINO_ASSERT(tile_sample_min_num_frames % temporal_compression_ratio == 0 &&
                        tile_sample_stride_num_frames % temporal_compression_ratio == 0,
                    "'tile_sample_min_num_frames' and 'tile_sample_stride_num_frames' must be divisible by ",
                    temporal_compression_ratio,
                    " but got ",
                    tile_sample_min_num_frames,
                    " and ",
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
    // The last frames of the previous chunk [B, tail_frames, H, W, C]. Nothing else of it is needed for blending,
    // so the decoded chunk is released before the next one is decoded.
    std::vector<uint8_t> tail;
    size_t tail_frames = 0, frame_size = 0, position = 0;

    for (size_t begin = 0; begin < latent_frames; begin += step) {
        const size_t chunk_latent_frames = std::min(chunk + 1, latent_frames - begin);
        const size_t skip = leading_frames_to_drop(begin);
        if (chunk_latent_frames == 1 && skip == 1) {
            // A single latent frame decodes to the frame which is dropped
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

        ov::Tensor decoded = decode(latent_chunk);
        const ov::Shape decoded_shape = decoded.get_shape();
        OPENVINO_ASSERT(decoded.get_element_type() == ov::element::u8 && decoded_shape.size() == 5 &&
                            decoded_shape[1] == (chunk_latent_frames - 1) * temporal_compression_ratio + 1,
                        "Unexpected VAE decoder output ",
                        decoded.get_element_type(),
                        " ",
                        decoded_shape);

        const size_t chunk_frames = decoded_shape[1] - skip;
        const size_t blend_frames = std::min({tail_frames, chunk_frames, overlap});
        const size_t write_frames = std::min({keep + 1 - skip, chunk_frames, num_frames - position});
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
            // Like diffusers, the next chunk blends with this chunk after its own blend
            std::memcpy(next_tail.data() + b * next_tail_frames * frame_size,
                        current + (chunk_frames - next_tail_frames) * frame_size,
                        next_tail_frames * frame_size);
        }
        position += write_frames;
        tail = std::move(next_tail);
        tail_frames = next_tail_frames;
    }
    OPENVINO_ASSERT(position == num_frames,
                    "Temporal VAE tiling produced ",
                    position,
                    " frames instead of ",
                    num_frames);

    return video;
}

void AutoencoderKLLTX2Video::merge_vae_video_post_processing() const {
    ov::preprocess::PrePostProcessor ppp(m_decoder_model);

    if (m_config.scaling_factor != 1.0f)
        ppp.input().preprocess().scale(m_config.scaling_factor);

    // (x / 2 + 0.5) -> clamp(0..1) -> *255 -> round() -> u8
    ppp.output().postprocess().custom([](const ov::Output<ov::Node>& port) {
        auto c_0_5 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.5f);
        auto c_255 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 255.0f);

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
