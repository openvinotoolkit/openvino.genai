// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>
#include <nlohmann/json.hpp>

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct Qwen3_5RopeConfig {
    bool mrope_interleaved = false;
    std::vector<int32_t> mrope_section = {11, 11, 10};
    std::string rope_type = "default";
};

struct Qwen3_5TextConfig {
    std::string model_type = "qwen3_5_text";
    int32_t vocab_size = 0;
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
    int32_t max_position_embeddings = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    std::string hidden_act = "silu";
    bool attention_bias = false;
    float attention_dropout = 0.0f;
    bool tie_word_embeddings = false;
    std::string dtype = "float16";
    float partial_rotary_factor = 0.25f;
    int32_t full_attention_interval = 4;
    std::vector<std::string> layer_types;
    int32_t linear_conv_kernel_dim = 4;
    int32_t linear_key_head_dim = 128;
    int32_t linear_value_head_dim = 128;
    int32_t linear_num_key_heads = 16;
    int32_t linear_num_value_heads = 32;
    int32_t moe_intermediate_size = 0;
    int32_t shared_expert_intermediate_size = 0;
    int32_t num_experts = 0;
    int32_t num_experts_per_tok = 0;
    bool norm_topk_prob = true;
    bool output_router_logits = false;
    float router_aux_loss_coef = 0.0f;
    int64_t eos_token_id = 0;
    Qwen3_5RopeConfig rope;

    int32_t kv_heads() const;
    int32_t resolved_head_dim() const;
    bool is_moe_enabled() const;
    void finalize();
    void validate() const;
};

struct Qwen3_5VisionConfig {
    std::string model_type = "qwen3_5";
    int32_t depth = 0;
    int32_t hidden_size = 0;
    std::string hidden_act = "gelu_pytorch_tanh";
    int32_t intermediate_size = 0;
    int32_t num_heads = 0;
    int32_t in_channels = 3;
    int32_t patch_size = 16;
    int32_t spatial_merge_size = 2;
    int32_t temporal_patch_size = 2;
    int32_t out_hidden_size = 0;
    int32_t num_position_embeddings = 0;
    std::vector<int32_t> deepstack_visual_indexes;
    float initializer_range = 0.02f;

    int32_t head_dim() const;
    void finalize();
    void validate() const;
};

struct Qwen3_5Config {
    std::string model_type = "qwen3_5";
    std::vector<std::string> architectures;
    Qwen3_5TextConfig text;
    Qwen3_5VisionConfig vision;
    int32_t image_token_id = 248056;
    int32_t video_token_id = 248057;
    int32_t vision_start_token_id = 248053;
    int32_t vision_end_token_id = 248054;
    bool tie_word_embeddings = false;

    void finalize();
    void validate() const;

    static Qwen3_5Config from_json(const nlohmann::json& data);
    static Qwen3_5Config from_json_file(const std::filesystem::path& config_path);
    static Qwen3_5Config make_dummy_dense9b_config();
    static Qwen3_5Config make_dummy_moe35b_config();
};

struct Qwen3_5ModuleNames {
    static constexpr const char* kRoot = "model";
    static constexpr const char* kVision = "visual";
    static constexpr const char* kText = "language_model";
    static constexpr const char* kLmHead = "lm_head";
    static std::string vision_block(int32_t index);
    static std::string deepstack_merger(int32_t index);
    static std::string text_layer(int32_t index);
};

struct Qwen3_5VisionIO {
    static constexpr const char* kPixelValues = "pixel_values";
    static constexpr const char* kGridThw = "grid_thw";
    static constexpr const char* kPosEmbeds = "pos_embeds";
    static constexpr const char* kRotaryCos = "rotary_cos";
    static constexpr const char* kRotarySin = "rotary_sin";
    static constexpr const char* kVisualEmbeds = "visual_embeds";
    static constexpr const char* kDeepstackEmbedsPrefix = "deepstack_embeds";
};

struct Qwen3_5TextIO {
    static constexpr const char* kInputIds = "input_ids";
    static constexpr const char* kInputsEmbeds = "inputs_embeds";
    static constexpr const char* kAttentionMask = "attention_mask";
    static constexpr const char* kPositionIds = "position_ids";
    static constexpr const char* kBeamIdx = "beam_idx";
    static constexpr const char* kVisualEmbeds = "visual_embeds";
    static constexpr const char* kVisualPosMask = "visual_pos_mask";
    static constexpr const char* kDeepstackEmbedsPrefix = "deepstack_embeds";
    static constexpr const char* kLogits = "logits";
};

struct Qwen3_5GraphSpec {
    static std::vector<std::string> vision_required_inputs(bool use_external_pos_embeds);
    static std::vector<std::string> vision_outputs(const Qwen3_5VisionConfig& cfg);
    static std::vector<std::string> text_required_inputs(bool use_inputs_embeds);
    static std::vector<std::string> text_optional_inputs();
};

struct Qwen3_5InputPlan {
    ov::Tensor position_ids;
    ov::Tensor visual_pos_mask;
    ov::Tensor rope_deltas;
};

class Qwen3_5InputPlanner {
public:
    explicit Qwen3_5InputPlanner(const Qwen3_5Config& cfg);

    Qwen3_5InputPlan build_plan(const ov::Tensor& input_ids,
                                const ov::Tensor* attention_mask = nullptr,
                                const ov::Tensor* image_grid_thw = nullptr,
                                const ov::Tensor* video_grid_thw = nullptr) const;

    ov::Tensor build_visual_pos_mask(const ov::Tensor& input_ids,
                                     const ov::Tensor* attention_mask = nullptr) const;

    static ov::Tensor scatter_visual_embeds(const ov::Tensor& visual_embeds,
                                            const ov::Tensor& visual_pos_mask);

    static std::vector<ov::Tensor> scatter_deepstack_embeds(const std::vector<ov::Tensor>& deepstack_embeds,
                                                            const ov::Tensor& visual_pos_mask);

    static ov::Tensor build_decode_position_ids(const ov::Tensor& rope_deltas,
                                                int64_t past_length,
                                                int64_t seq_len);

private:
    int64_t image_token_id_ = 0;
    int64_t video_token_id_ = 0;
    int64_t vision_start_token_id_ = 0;
    int32_t spatial_merge_size_ = 1;
};

struct Qwen3_5VisionPreprocessConfig {
    int64_t min_pixels = 56 * 56;
    int64_t max_pixels = 28 * 28 * 1280;
    int32_t patch_size = 16;
    int32_t temporal_patch_size = 2;
    int32_t merge_size = 2;
    std::array<float, 3> image_mean = {0.5f, 0.5f, 0.5f};
    std::array<float, 3> image_std = {0.5f, 0.5f, 0.5f};
    bool do_resize = true;

    static Qwen3_5VisionPreprocessConfig from_json_file(const std::filesystem::path& path);
};

struct Qwen3_5VisionInputs {
    ov::Tensor pixel_values;
    ov::Tensor grid_thw;
    ov::Tensor pos_embeds;
    ov::Tensor rotary_cos;
    ov::Tensor rotary_sin;
};

class Qwen3_5VisionPreprocessor {
public:
    Qwen3_5VisionPreprocessor(const Qwen3_5VisionConfig& vision_cfg,
                              const Qwen3_5VisionPreprocessConfig& preprocess_cfg);

    Qwen3_5VisionInputs preprocess(const ov::Tensor& images,
                                   const ov::Tensor& pos_embed_weight) const;

    static int64_t count_visual_tokens(const ov::Tensor& grid_thw,
                                       int32_t spatial_merge_size);

private:
    Qwen3_5VisionConfig vision_cfg_;
    Qwen3_5VisionPreprocessConfig preprocess_cfg_;
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

