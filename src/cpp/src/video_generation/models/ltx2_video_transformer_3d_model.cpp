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

bool has_exact_input(const ov::CompiledModel& compiled_model, const std::string& name) {
    for (const auto& input : compiled_model.inputs()) {
        if (input.get_names().count(name)) {
            return true;
        }
    }
    return false;
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
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "LTX2 Video Transformer 3D model");
    m_request = compiled_model.create_infer_request();
    const auto& input_shape = compiled_model.input("hidden_states").get_partial_shape();
    m_expected_batch_size = input_shape[0].is_static() ? input_shape[0].get_length() : 0;
    m_timestep_rank = compiled_model.input("timestep").get_partial_shape().rank().get_length();
    m_has_audio_timestep = has_exact_input(compiled_model, "audio_timestep");
    m_has_cross_modality_gate = has_exact_input(compiled_model, "cross_modality_gate");
    m_has_stg_perturbation_mask = has_exact_input(compiled_model, "stg_perturbation_mask");
    if (m_has_stg_perturbation_mask) {
        // The mask is one entry per transformer block. The export usually pins the dimension; fall back to
        // the config when it does not.
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

    // Legacy exports take a rank-1 [B] timestep, current ones a rank-2 [B, S] per-token timestep
    OPENVINO_ASSERT(m_timestep_rank == 1 || m_timestep_rank == 2,
                    "LTX2 transformer expects a rank-1 or rank-2 'timestep' input, got rank ", m_timestep_rank);
    ov::Shape timestep_shape{latent_shape[0]};
    if (m_timestep_rank == 2) {
        timestep_shape = {latent_shape[0], latent_shape[1]};
    }
    ov::Tensor timestep_tensor(ov::element::f32, timestep_shape);
    std::fill_n(timestep_tensor.data<float>(), timestep_tensor.get_size(), timestep);
    m_request.set_tensor("timestep", timestep_tensor);

    if (m_has_audio_timestep) {
        ov::Tensor audio_timestep(ov::element::f32, {audio_latent.get_shape()[0]});
        std::fill_n(audio_timestep.data<float>(), audio_timestep.get_size(), timestep);
        m_request.set_tensor("audio_timestep", audio_timestep);
    }

    // Both LTX-2.3 guidance inputs are re-bound on every call, including at their neutral values: the infer
    // request keeps the tensors of the previous pass otherwise, which would leak one pass's perturbation
    // into the next.
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
                                                                  int64_t audio_num_frames,
                                                                  bool dynamic_batch) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    // LTX-2.3's extra guidance passes run at batch 1 while classifier-free guidance runs at batch 2, so one
    // statically shaped model cannot serve both and compiling twice would double the transformer's weights.
    if (dynamic_batch) {
        batch_size = -1;
    }

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
                name_to_shape[input_name] = {batch_size, video_sequence_length};
            } else {
                name_to_shape[input_name] = {batch_size};
            }
        } else if (input_name == "audio_timestep") {
            name_to_shape[input_name] = {batch_size};
        } else if (input_name == "video_coords") {
            name_to_shape[input_name] = {batch_size, 3, video_sequence_length, 2};
        } else if (input_name == "audio_coords") {
            name_to_shape[input_name] = {batch_size, 1, audio_num_frames, 2};
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
