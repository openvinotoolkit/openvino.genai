// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/ltx2_video_transformer_3d_model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

#include "json_utils.hpp"
#include "utils.hpp"
#include "video_generation/video_generation_utils.hpp"

using namespace ov::genai;

namespace {

// One rotary module of diffusers' LTX2VideoTransformer3DModel and the transformer inputs it feeds when the
// export takes RoPE as precomputed (cos, sin), see optimum-intel's LTX2_ROPE_INPUTS
enum class RopeModule { Video, Audio, VideoCrossAttn, AudioCrossAttn };

struct RopeInput {
    RopeModule module;
    const char* cos_name;
    const char* sin_name;
};

constexpr RopeInput kRopeInputs[] = {
    {RopeModule::Video, "video_rotary_cos", "video_rotary_sin"},
    {RopeModule::Audio, "audio_rotary_cos", "audio_rotary_sin"},
    {RopeModule::VideoCrossAttn, "video_cross_attn_rotary_cos", "video_cross_attn_rotary_sin"},
    {RopeModule::AudioCrossAttn, "audio_cross_attn_rotary_cos", "audio_cross_attn_rotary_sin"},
};

bool is_video_rope(RopeModule module) {
    return module == RopeModule::Video || module == RopeModule::VideoCrossAttn;
}

// Port of LTX2AudioVideoRotaryPosEmbed.forward (diffusers v0.40.0, transformer_ltx2.py). The cross-attention
// modules see only the first (temporal) coordinate, as the transformer passes them `coords[:, 0:1]`. The
// angle arithmetic follows the reference in fp32 (fp64 for the frequency grid when 'rope_double_precision'),
// so the only difference from PyTorch is the last ulp of cos/sin.
std::pair<ov::Tensor, ov::Tensor> compute_rope(const LTX2VideoTransformer3DModel::Config& config,
                                               RopeModule module,
                                               const ov::Tensor& coords) {
    const ov::Shape& coords_shape = coords.get_shape();
    OPENVINO_ASSERT(coords_shape.size() == 4 && coords_shape[3] == 2,
                    "RoPE coords must be [B, num_pos_dims, num_patches, 2], got ",
                    coords_shape);
    OPENVINO_ASSERT(coords.get_element_type() == ov::element::f32, "RoPE coords must be f32");

    const bool cross_attn = module == RopeModule::VideoCrossAttn || module == RopeModule::AudioCrossAttn;
    const bool video_modality = is_video_rope(module);
    const size_t batch = coords_shape[0];
    const size_t coords_dims = coords_shape[1];
    const size_t num_patches = coords_shape[2];
    const size_t num_pos_dims = cross_attn ? 1 : coords_dims;
    OPENVINO_ASSERT(num_pos_dims == (video_modality && !cross_attn ? 3u : 1u),
                    "Unexpected number of RoPE coordinate dimensions: ",
                    coords_dims);

    size_t dim = 0;
    size_t num_heads = 0;
    int64_t base_num_frames = 0;
    switch (module) {
    case RopeModule::Video:
        dim = config.num_attention_heads * config.attention_head_dim;
        num_heads = config.num_attention_heads;
        base_num_frames = config.pos_embed_max_pos;
        break;
    case RopeModule::Audio:
        dim = config.audio_num_attention_heads * config.audio_attention_head_dim;
        num_heads = config.audio_num_attention_heads;
        base_num_frames = config.audio_pos_embed_max_pos;
        break;
    case RopeModule::VideoCrossAttn:
        dim = config.audio_cross_attention_dim;
        num_heads = config.num_attention_heads;
        base_num_frames = std::max(config.pos_embed_max_pos, config.audio_pos_embed_max_pos);
        break;
    case RopeModule::AudioCrossAttn:
        dim = config.audio_cross_attention_dim;
        num_heads = config.audio_num_attention_heads;
        base_num_frames = std::max(config.pos_embed_max_pos, config.audio_pos_embed_max_pos);
        break;
    }
    const float max_positions[3] = {static_cast<float>(base_num_frames),
                                    static_cast<float>(config.base_height),
                                    static_cast<float>(config.base_width)};

    // 1D frequency grid: theta ** linspace(0, 1, steps) * pi / 2, cast to fp32
    const size_t num_rope_elems = num_pos_dims * 2;
    const size_t steps = dim / num_rope_elems;
    std::vector<float> freqs(steps);
    for (size_t k = 0; k < steps; ++k) {
        const double position = steps > 1 ? static_cast<double>(k) / static_cast<double>(steps - 1) : 0.0;
        if (config.rope_double_precision) {
            freqs[k] = static_cast<float>(std::pow(config.rope_theta, position) * video_generation_utils::PI / 2.0);
        } else {
            const float pow_index = std::pow(static_cast<float>(config.rope_theta), static_cast<float>(position));
            freqs[k] = pow_index * static_cast<float>(video_generation_utils::PI) / 2.0f;
        }
    }

    // Angles laid out as the reference's transpose(-1, -2).flatten(2): index k * num_pos_dims + i
    const size_t num_freqs = steps * num_pos_dims;
    const float* coords_data = coords.data<const float>();
    auto angle = [&](size_t b, size_t t, size_t freq_index) {
        const size_t k = freq_index / num_pos_dims;
        const size_t i = freq_index % num_pos_dims;
        const float* bounds = coords_data + ((b * coords_dims + i) * num_patches + t) * 2;
        const float midpoint = (bounds[0] + bounds[1]) / 2.0f;
        const float grid = midpoint / max_positions[i];
        return (grid * 2.0f - 1.0f) * freqs[k];
    };

    ov::Tensor cos_tensor, sin_tensor;
    if (config.rope_type == "split") {
        // [B, H, T, D / 2 / H], front-padded with cos = 1, sin = 0 up to D / 2
        const size_t half_dim = dim / 2;
        OPENVINO_ASSERT(half_dim % num_heads == 0, "RoPE dim ", dim, " does not split over ", num_heads, " heads");
        OPENVINO_ASSERT(half_dim >= num_freqs, "RoPE dim ", dim, " is too small for ", num_freqs, " frequencies");
        const size_t pad = half_dim - num_freqs;
        const size_t head_dim = half_dim / num_heads;
        cos_tensor = ov::Tensor(ov::element::f32, {batch, num_heads, num_patches, head_dim});
        sin_tensor = ov::Tensor(ov::element::f32, {batch, num_heads, num_patches, head_dim});
        float* cos_data = cos_tensor.data<float>();
        float* sin_data = sin_tensor.data<float>();
        for (size_t b = 0; b < batch; ++b) {
            for (size_t t = 0; t < num_patches; ++t) {
                for (size_t j = 0; j < half_dim; ++j) {
                    const size_t h = j / head_dim;
                    const size_t d = j % head_dim;
                    const size_t out = ((b * num_heads + h) * num_patches + t) * head_dim + d;
                    if (j < pad) {
                        cos_data[out] = 1.0f;
                        sin_data[out] = 0.0f;
                    } else {
                        const double a = angle(b, t, j - pad);
                        cos_data[out] = static_cast<float>(std::cos(a));
                        sin_data[out] = static_cast<float>(std::sin(a));
                    }
                }
            }
        }
    } else {
        OPENVINO_ASSERT(config.rope_type == "interleaved", "Unsupported 'rope_type': ", config.rope_type);
        // [B, T, D], each frequency repeated twice, front-padded with cos = 1, sin = 0
        const size_t pad = dim % num_rope_elems;
        cos_tensor = ov::Tensor(ov::element::f32, {batch, num_patches, dim});
        sin_tensor = ov::Tensor(ov::element::f32, {batch, num_patches, dim});
        float* cos_data = cos_tensor.data<float>();
        float* sin_data = sin_tensor.data<float>();
        for (size_t b = 0; b < batch; ++b) {
            for (size_t t = 0; t < num_patches; ++t) {
                float* cos_row = cos_data + (b * num_patches + t) * dim;
                float* sin_row = sin_data + (b * num_patches + t) * dim;
                std::fill_n(cos_row, pad, 1.0f);
                std::fill_n(sin_row, pad, 0.0f);
                for (size_t f = 0; f < num_freqs; ++f) {
                    const double a = angle(b, t, f);
                    cos_row[pad + 2 * f] = cos_row[pad + 2 * f + 1] = static_cast<float>(std::cos(a));
                    sin_row[pad + 2 * f] = sin_row[pad + 2 * f + 1] = static_cast<float>(std::sin(a));
                }
            }
        }
    }
    return {cos_tensor, sin_tensor};
}

}  // namespace

LTX2VideoTransformer3DModel::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "audio_in_channels", audio_in_channels);
    read_json_param(data, "patch_size", patch_size);
    read_json_param(data, "patch_size_t", patch_size_t);
    read_json_param(data, "vae_scale_factors", vae_scale_factors);
    read_json_param(data, "audio_scale_factor", audio_scale_factor);
    read_json_param(data, "causal_offset", causal_offset);
    read_json_param(data, "audio_sampling_rate", audio_sampling_rate);
    read_json_param(data, "audio_hop_length", audio_hop_length);
    read_json_param(data, "num_layers", num_layers);
    read_json_param(data, "perturbed_attn", perturbed_attn);
    read_json_param(data, "num_attention_heads", num_attention_heads);
    read_json_param(data, "attention_head_dim", attention_head_dim);
    read_json_param(data, "audio_num_attention_heads", audio_num_attention_heads);
    read_json_param(data, "audio_attention_head_dim", audio_attention_head_dim);
    read_json_param(data, "audio_cross_attention_dim", audio_cross_attention_dim);
    read_json_param(data, "pos_embed_max_pos", pos_embed_max_pos);
    read_json_param(data, "audio_pos_embed_max_pos", audio_pos_embed_max_pos);
    read_json_param(data, "base_height", base_height);
    read_json_param(data, "base_width", base_width);
    read_json_param(data, "rope_theta", rope_theta);
    read_json_param(data, "rope_double_precision", rope_double_precision);
    read_json_param(data, "rope_type", rope_type);
}

LTX2VideoTransformer3DModel::LTX2VideoTransformer3DModel(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json") {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

LTX2VideoTransformer3DModel::LTX2VideoTransformer3DModel(const std::filesystem::path& root_dir,
                                                         const std::string& device,
                                                         const ov::AnyMap& properties)
    : LTX2VideoTransformer3DModel(root_dir) {
    compile(device, properties);
}

LTX2VideoTransformer3DModel::LTX2VideoTransformer3DModel(const LTX2VideoTransformer3DModel&) = default;

LTX2VideoTransformer3DModel LTX2VideoTransformer3DModel::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "LTX2VideoTransformer3DModel must have exactly one of m_model or m_request initialized");

    LTX2VideoTransformer3DModel cloned = *this;

    if (m_model) {
        cloned.m_model = m_model->clone();
    } else {
        cloned.m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

const LTX2VideoTransformer3DModel::Config& LTX2VideoTransformer3DModel::get_config() const {
    return m_config;
}

LTX2VideoTransformer3DModel& LTX2VideoTransformer3DModel::compile(const std::string& device,
                                                                  const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    size_t rotary_inputs = 0;
    for (const auto& input : kRopeInputs) {
        rotary_inputs += has_input(input.cos_name);
        rotary_inputs += has_input(input.sin_name);
    }
    const bool video_coords = has_input("video_coords");
    const bool audio_coords = has_input("audio_coords");
    OPENVINO_ASSERT((rotary_inputs == 8 && !video_coords && !audio_coords) ||
                        (rotary_inputs == 0 && video_coords && audio_coords),
                    "LTX2 transformer requires all eight external rotary inputs or both legacy coordinate inputs");
    m_external_rope = rotary_inputs != 0;
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "LTX2 Video Transformer 3D model");
    m_request = compiled_model.create_infer_request();
    const auto& input_shape = compiled_model.input("hidden_states").get_partial_shape();
    m_expected_batch_size = input_shape[0].is_static() ? input_shape[0].get_length() : 0;
    m_timestep_rank = compiled_model.input("timestep").get_partial_shape().rank().get_length();
    m_has_audio_timestep = has_input("audio_timestep");
    m_has_cross_modality_gate = has_input("cross_modality_gate");
    m_has_stg_perturbation_mask = has_input("stg_perturbation_mask");
    if (m_has_stg_perturbation_mask) {
        // One mask entry per transformer block; fall back to the config if the export leaves it dynamic
        const auto& mask_shape = compiled_model.input("stg_perturbation_mask").get_partial_shape();
        m_num_stg_blocks = mask_shape[0].is_static() ? mask_shape[0].get_length() : m_config.num_layers;
        OPENVINO_ASSERT(m_num_stg_blocks > 0,
                        "'stg_perturbation_mask' has a dynamic length and 'num_layers' is missing from the "
                        "transformer config, so the mask cannot be sized");
    }
    m_model.reset();

    return *this;
}

void LTX2VideoTransformer3DModel::set_hidden_states(const std::string& tensor_name, const ov::Tensor& tensor) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    const auto input_type = m_request.get_compiled_model().input(tensor_name).get_element_type();
    m_request.set_tensor(tensor_name, video_generation_utils::convert_tensor(tensor, input_type));
}

size_t LTX2VideoTransformer3DModel::get_expected_batch_size() const {
    return m_expected_batch_size;
}

bool LTX2VideoTransformer3DModel::has_input(const std::string& name) {
    if (m_model) {
        return utils::has_input(m_model, name);
    }
    OPENVINO_ASSERT(m_request, "Transformer model must be loaded first");
    for (const auto& input : m_request.get_compiled_model().inputs()) {
        if (input.get_names().count(name)) {
            return true;
        }
    }
    return false;
}

void LTX2VideoTransformer3DModel::set_rope_coords(const ov::Tensor& video_coords, const ov::Tensor& audio_coords) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    if (!m_external_rope) {
        set_hidden_states("video_coords", video_coords);
        set_hidden_states("audio_coords", audio_coords);
        return;
    }
    for (const auto& input : kRopeInputs) {
        auto [cos, sin] =
            compute_rope(m_config, input.module, is_video_rope(input.module) ? video_coords : audio_coords);
        m_request.set_tensor(input.cos_name, cos);
        m_request.set_tensor(input.sin_name, sin);
    }
}

size_t LTX2VideoTransformer3DModel::get_timestep_rank() {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first. Cannot query non-compiled model");
    return m_timestep_rank;
}

bool LTX2VideoTransformer3DModel::has_cross_modality_gate() const {
    return m_has_cross_modality_gate;
}

bool LTX2VideoTransformer3DModel::has_stg_perturbation_mask() const {
    return m_has_stg_perturbation_mask;
}

std::pair<ov::Tensor, ov::Tensor> LTX2VideoTransformer3DModel::infer(const ov::Tensor& video_latent,
                                                                     const ov::Tensor& audio_latent,
                                                                     float timestep,
                                                                     bool isolate_modalities,
                                                                     const std::vector<int64_t>& stg_blocks) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first. Cannot infer non-compiled model");

    m_request.set_tensor("hidden_states", video_latent);
    m_request.set_tensor("audio_hidden_states", audio_latent);

    const ov::Shape& latent_shape = video_latent.get_shape();
    OPENVINO_ASSERT(latent_shape.size() == 3, "Packed latents must be rank-3 [B, S, C], got rank ", latent_shape.size());

    // External-RoPE exports broadcast [B, 1] modulation across tokens. Internal-RoPE exports retain their
    // original [B, S] contract, since their graph may reshape the modulation to the full sequence length.
    OPENVINO_ASSERT(m_timestep_rank == 1 || m_timestep_rank == 2,
                    "LTX2 transformer expects a rank-1 or rank-2 'timestep' input, got rank ", m_timestep_rank);
    ov::Shape timestep_shape{latent_shape[0]};
    if (m_timestep_rank == 2) {
        timestep_shape = {latent_shape[0], m_external_rope ? 1 : latent_shape[1]};
    }
    ov::Tensor timestep_tensor(ov::element::f32, timestep_shape);
    std::fill_n(timestep_tensor.data<float>(), timestep_tensor.get_size(), timestep);
    m_request.set_tensor("timestep", timestep_tensor);

    if (m_has_audio_timestep) {
        ov::Tensor audio_timestep(ov::element::f32, {audio_latent.get_shape()[0]});
        std::fill_n(audio_timestep.data<float>(), audio_timestep.get_size(), timestep);
        m_request.set_tensor("audio_timestep", audio_timestep);
    }

    // Re-bound on every call, so one pass's perturbation cannot leak into the next
    if (m_has_cross_modality_gate) {
        // 1.0 lets audio and video attend to each other, as LTX-2.0 always does; 0.0 isolates them
        ov::Tensor cross_modality_gate(ov::element::f32, ov::Shape{});
        cross_modality_gate.data<float>()[0] = isolate_modalities ? 0.0f : 1.0f;
        m_request.set_tensor("cross_modality_gate", cross_modality_gate);
    } else {
        OPENVINO_ASSERT(!isolate_modalities,
                        "Modality isolation guidance was requested but the transformer has no "
                        "'cross_modality_gate' input");
    }

    if (m_has_stg_perturbation_mask) {
        // All-ones leaves every block unperturbed; a perturbed block is weighted by 0.0
        ov::Tensor stg_mask(ov::element::f32, ov::Shape{m_num_stg_blocks});
        std::fill_n(stg_mask.data<float>(), stg_mask.get_size(), 1.0f);
        for (int64_t block_idx : stg_blocks) {
            // Out-of-range indices are ignored rather than rejected, matching the reference implementation
            if (block_idx >= 0 && static_cast<size_t>(block_idx) < m_num_stg_blocks) {
                stg_mask.data<float>()[block_idx] = 0.0f;
            }
        }
        m_request.set_tensor("stg_perturbation_mask", stg_mask);
    } else {
        OPENVINO_ASSERT(stg_blocks.empty(),
                        "Spatio-Temporal Guidance was requested but the transformer has no "
                        "'stg_perturbation_mask' input");
    }

    m_request.infer();

    return {m_request.get_tensor("out_sample"), m_request.get_tensor("audio_out_sample")};
}

LTX2VideoTransformer3DModel& LTX2VideoTransformer3DModel::reshape(int64_t batch_size,
                                                                  int64_t num_frames,
                                                                  int64_t height,
                                                                  int64_t width,
                                                                  int64_t audio_num_frames) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    const int64_t patch_size = m_config.patch_size;
    const int64_t patch_size_t = m_config.patch_size_t;

    OPENVINO_ASSERT(m_config.vae_scale_factors.size() == 3,
                    "'vae_scale_factors' must contain [temporal, height, width] ratios");
    const int64_t temporal_compression_ratio = m_config.vae_scale_factors[0];
    const int64_t spatial_compression_ratio = m_config.vae_scale_factors[1];

    const int64_t latent_num_frames = ((num_frames - 1) / temporal_compression_ratio + 1) / patch_size_t;
    const int64_t latent_height = height / (spatial_compression_ratio * patch_size);
    const int64_t latent_width = width / (spatial_compression_ratio * patch_size);
    const int64_t video_sequence_length = latent_num_frames * latent_height * latent_width;

    std::map<std::string, ov::PartialShape> name_to_shape;

    for (auto&& input : m_model->inputs()) {
        std::string input_name = input.get_any_name();
        name_to_shape[input_name] = input.get_partial_shape();
        if (input_name == "hidden_states") {
            name_to_shape[input_name] = {batch_size, video_sequence_length, name_to_shape[input_name][2]};
        } else if (input_name == "audio_hidden_states") {
            name_to_shape[input_name] = {batch_size, audio_num_frames, name_to_shape[input_name][2]};
        } else if (input_name == "timestep") {
            const auto timestep_rank = name_to_shape[input_name].rank().get_length();
            OPENVINO_ASSERT(timestep_rank == 1 || timestep_rank == 2,
                            "LTX2 transformer expects a rank-1 or rank-2 'timestep' input, got rank ", timestep_rank);
            if (timestep_rank == 2) {
                name_to_shape[input_name] = {
                    batch_size, utils::has_input(m_model, "video_rotary_cos") ? 1 : video_sequence_length};
            } else {
                name_to_shape[input_name] = {batch_size};
            }
        } else if (input_name == "audio_timestep") {
            name_to_shape[input_name] = {batch_size};
        } else if (input_name == "video_coords") {
            name_to_shape[input_name] = {batch_size, 3, video_sequence_length, 2};
        } else if (input_name == "audio_coords") {
            name_to_shape[input_name] = {batch_size, 1, audio_num_frames, 2};
        } else if (const auto rope_input = std::find_if(std::begin(kRopeInputs),
                                                        std::end(kRopeInputs),
                                                        [&](const RopeInput& rope) {
                                                            return input_name == rope.cos_name ||
                                                                   input_name == rope.sin_name;
                                                        });
                   rope_input != std::end(kRopeInputs)) {
            // "split" is [B, H, T, D'] and "interleaved" [B, T, D]; the head and rotary dims stay as exported
            auto& shape = name_to_shape[input_name];
            const size_t sequence_axis = shape.rank().get_length() == 4 ? 2 : 1;
            shape[0] = batch_size;
            shape[sequence_axis] = is_video_rope(rope_input->module) ? video_sequence_length : audio_num_frames;
        } else if (input_name == "encoder_hidden_states" || input_name == "audio_encoder_hidden_states") {
            // The connector output sequence length is model-internal, keep it dynamic
            name_to_shape[input_name] = {batch_size, -1, name_to_shape[input_name][2]};
        } else if (input_name == "encoder_attention_mask" || input_name == "audio_encoder_attention_mask") {
            name_to_shape[input_name] = {batch_size, -1};
        }
    }

    m_model->reshape(name_to_shape);

    return *this;
}
