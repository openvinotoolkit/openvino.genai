// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen2_5_vl/classes.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace qwen2_5_vl_utils {

std::pair<ov::Tensor, std::vector<int32_t>> get_window_index(
    const std::vector<std::array<size_t, 3>>& grids_thw,
    const ProcessorConfig& processor_config,
    const VLMConfig& vlm_config
) {
    std::vector<int64_t> window_indices;
    std::vector<int32_t> cu_window_seqlens = {0};
    size_t window_index_id = 0;

    const size_t spatial_merge_size = processor_config.merge_size;
    const size_t spatial_merge_unit = spatial_merge_size * spatial_merge_size;
    const size_t vit_merger_window_size = vlm_config.vision_config_window_size / spatial_merge_size / vlm_config.vision_config_patch_size;

    for (const auto& grid_thw : grids_thw) {
        size_t grid_t = grid_thw.at(0);
        size_t grid_h = grid_thw.at(1);
        size_t grid_w = grid_thw.at(2);

        size_t llm_grid_h = grid_h / spatial_merge_size;
        size_t llm_grid_w = grid_w / spatial_merge_size;

        size_t pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size) % vit_merger_window_size;
        size_t pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size) % vit_merger_window_size;

        size_t num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
        size_t num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;

        for (size_t t = 0; t < grid_t; ++t) {
            for (size_t wh = 0; wh < num_windows_h; ++wh) {
                for (size_t ww = 0; ww < num_windows_w; ++ww) {
                    int32_t valid_count = 0;
                    for (size_t h = 0; h < vit_merger_window_size; ++h) {
                        size_t gh = wh * vit_merger_window_size + h;
                        if (gh >= llm_grid_h + pad_h) break;
                        for (size_t w = 0; w < vit_merger_window_size; ++w) {
                            size_t gw = ww * vit_merger_window_size + w;
                            if (gw >= llm_grid_w + pad_w) break;
                            if (gh < llm_grid_h && gw < llm_grid_w) {
                                int32_t idx = static_cast<int32_t>(t * llm_grid_h * llm_grid_w + gh * llm_grid_w + gw);
                                window_indices.push_back(idx + window_index_id);
                                valid_count++;
                            }
                        }
                    }
                    cu_window_seqlens.push_back(cu_window_seqlens.back() + valid_count * spatial_merge_unit);
                }
            }
        }
        window_index_id += static_cast<size_t>(grid_t * llm_grid_h * llm_grid_w);
    }

    ov::Tensor window_index_tensor{ov::element::i64, {window_indices.size()}};
    std::memcpy(window_index_tensor.data<int64_t>(), window_indices.data(), window_indices.size() * sizeof(int64_t));
    return {window_index_tensor, cu_window_seqlens};
}

ov::Tensor get_window_attention_mask(const size_t hidden_states_size, const std::vector<int32_t>& cu_window_seqlens) {
    ov::Tensor window_attention_mask{ov::element::f32, {1, hidden_states_size, hidden_states_size}};
    float* window_mask_data = window_attention_mask.data<float>();
    std::fill_n(window_mask_data, window_attention_mask.get_size(), -std::numeric_limits<float>::infinity());

    for (size_t i = 1; i < cu_window_seqlens.size(); ++i) {
        size_t start = cu_window_seqlens[i-1];
        size_t end = cu_window_seqlens[i];
        for (size_t row = start; row < end; ++row) {
            for (size_t col = start; col < end; ++col) {
                window_mask_data[row * hidden_states_size + col] = 0.0f;
            }
        }
    }
    return window_attention_mask;
}

ov::Tensor get_cu_window_seqlens(const std::vector<int32_t>& cu_window_seqlens) {
    // Convert cumulative window sequence lengths to ov Tensor
    ov::Tensor t_cu_seqlens(ov::element::i32, {cu_window_seqlens.size()});
    std::memcpy(t_cu_seqlens.data<int32_t>(), cu_window_seqlens.data(), cu_window_seqlens.size() * sizeof(int32_t));
    return t_cu_seqlens;
}

} // namespace qwen2_5_vl_utils

InputsEmbedderQwen2_5_VL::InputsEmbedderQwen2_5_VL(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    InputsEmbedderQwen2VL(vlm_config, model_dir, device, device_config) {}

InputsEmbedderQwen2_5_VL::InputsEmbedderQwen2_5_VL(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer, 
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    InputsEmbedderQwen2VL(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

ov::Tensor InputsEmbedderQwen2_5_VL::run_image_embeddings_merger(
    const std::vector<EncodedImage>& images, 
    const std::vector<size_t>& images_sequence
) {
    auto [reordered_image_embeds, reordered_images_grid_thw] = qwen2_vl_utils::reorder_image_embeds_and_grid_thw(images, images_sequence);

    ov::Tensor concatenated_embeds = qwen2_vl_utils::concatenate_image_embeds(reordered_image_embeds);
    ov::Tensor rotary_pos_emb = get_rotary_pos_emb(reordered_images_grid_thw);

    auto [window_index, cu_window_seqlens] = qwen2_5_vl_utils::get_window_index(
        reordered_images_grid_thw,
        m_vision_encoder->get_processor_config(),
        m_vlm_config
    );

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& vision_embeddings_merger = infer_request_guard.get();
    vision_embeddings_merger.set_tensor("hidden_states", concatenated_embeds);
    if (m_with_cu_seqlens_input) {
        ov::Tensor cu_seq_lens = qwen2_vl_utils::get_cu_seqlens(reordered_images_grid_thw);
        ov::Tensor t_cu_window_seqlens = qwen2_5_vl_utils::get_cu_window_seqlens(cu_window_seqlens);
        vision_embeddings_merger.set_tensor("cu_seq_lens", cu_seq_lens);
        vision_embeddings_merger.set_tensor("cu_window_seqlens", t_cu_window_seqlens);
    }
    else {
        ov::Tensor attention_mask = qwen2_vl_utils::get_attention_mask(reordered_images_grid_thw);
        size_t hidden_states_size = attention_mask.get_shape().at(1);
        ov::Tensor window_attention_mask = qwen2_5_vl_utils::get_window_attention_mask(hidden_states_size, cu_window_seqlens);
        vision_embeddings_merger.set_tensor("attention_mask", attention_mask);
        vision_embeddings_merger.set_tensor("window_attention_mask", window_attention_mask);
    }
    vision_embeddings_merger.set_tensor("rotary_pos_emb", rotary_pos_emb);
    vision_embeddings_merger.set_tensor("window_index", window_index);
    vision_embeddings_merger.infer();
    ov::Tensor processed_vision_embeds = vision_embeddings_merger.get_output_tensor();

    ov::Tensor res = ov::Tensor(processed_vision_embeds.get_element_type(), processed_vision_embeds.get_shape());
    std::memcpy(res.data(), processed_vision_embeds.data(), processed_vision_embeds.get_byte_size());
    return res;
}

} // namespace ov::genai
