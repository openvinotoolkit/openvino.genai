// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen2_5_vl/classes.hpp"

#include "utils.hpp"

namespace ov::genai {

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
    const std::vector<size_t>& images_sequence, 
    size_t image_id, 
    const VLMConfig& vlm_config) {
    
    std::vector<ov::Tensor> image_embeds;
    std::vector<std::array<size_t, 3>> images_grid_thw;
    image_embeds.reserve(images.size());
    images_grid_thw.reserve(images.size());
    
    for (const auto& encoded_image : images) {
        ov::Tensor single_image_embeds = encoded_image.resized_source;
        image_embeds.push_back(std::move(single_image_embeds));

        size_t grid_t = 1;
        size_t grid_h = encoded_image.resized_source_size.height;
        size_t grid_w = encoded_image.resized_source_size.width;
        images_grid_thw.push_back({grid_t, grid_h, grid_w});
    }

    std::vector<ov::Tensor> reordered_image_embeds;
    std::vector<std::array<size_t, 3>> reordered_images_grid_thw;
    for (size_t new_image_id : images_sequence) {
        reordered_image_embeds.push_back(image_embeds.at(new_image_id - image_id));
        reordered_images_grid_thw.push_back(images_grid_thw.at(new_image_id - image_id));
    }

    // Calculate cumulative sequence lengths for attention mask
    std::vector<int32_t> cu_seqlens;
    cu_seqlens.push_back(0);
    int32_t cumsum = 0;
    for (const auto& grid_thw : reordered_images_grid_thw) {
        size_t slice_len = grid_thw.at(1) * grid_thw.at(2);
        for (size_t t = 0; t < grid_thw.at(0); ++t) {
            cumsum += slice_len;
            cu_seqlens.push_back(cumsum);
        }
    }

    // Create attention mask for vision embeddings merger model
    size_t hidden_states_size = cumsum;
    ov::Tensor attention_mask{ov::element::f32, {1, hidden_states_size, hidden_states_size}};
    float* attention_mask_data = attention_mask.data<float>();
    std::fill_n(attention_mask_data, attention_mask.get_size(), -std::numeric_limits<float>::infinity());

    for (size_t i = 1; i < cu_seqlens.size(); ++i) {
        size_t start = cu_seqlens[i-1];
        size_t end = cu_seqlens[i];
        for (size_t row = start; row < end; ++row) {
            for (size_t col = start; col < end; ++col) {
                attention_mask_data[row * hidden_states_size + col] = 0.0f;
            }
        }
    }

    // Get window index and create window attention mask
    auto [window_index, cu_window_seqlens] = get_window_index(reordered_images_grid_thw);
    
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

    // Concatenate image embeddings
    ov::Tensor concatenated_images;
    if (reordered_image_embeds.size() == 1) {
        concatenated_images = reordered_image_embeds.at(0);
    } else {
        size_t total_length = 0;
        for (const auto& embed : reordered_image_embeds) {
            total_length += embed.get_shape().at(0);
        }
        size_t hidden_dim = reordered_image_embeds.at(0).get_shape().at(1);
        
        concatenated_images = ov::Tensor(reordered_image_embeds.at(0).get_element_type(), {total_length, hidden_dim});
        float* concat_data = concatenated_images.data<float>();
        
        size_t offset = 0;
        for (const auto& embed : reordered_image_embeds) {
            size_t embed_size = embed.get_shape().at(0) * embed.get_shape().at(1);
            std::memcpy(concat_data + offset, embed.data(), embed.get_byte_size());
            offset += embed_size;
        }
    }

    ov::Tensor rotary_pos_emb = get_rotary_pos_emb(reordered_images_grid_thw);
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& vision_embeddings_merger = infer_request_guard.get();

    vision_embeddings_merger.set_tensor("hidden_states", concatenated_images);
    vision_embeddings_merger.set_tensor("attention_mask", attention_mask);
    vision_embeddings_merger.set_tensor("rotary_pos_emb", rotary_pos_emb);
    vision_embeddings_merger.set_tensor("window_attention_mask", window_attention_mask);
    vision_embeddings_merger.set_tensor("window_index", window_index);
    vision_embeddings_merger.infer();
    ov::Tensor processed_vision_embeds = vision_embeddings_merger.get_output_tensor();

    ov::Tensor res = ov::Tensor(processed_vision_embeds.get_element_type(), processed_vision_embeds.get_shape());
    std::memcpy(res.data(), processed_vision_embeds.data(), processed_vision_embeds.get_byte_size());
    return res;
}

std::pair<ov::Tensor, std::vector<int32_t>> InputsEmbedderQwen2_5_VL::get_window_index(const std::vector<std::array<size_t, 3>>& grids_thw) {
    std::vector<size_t> window_indices;
    std::vector<int32_t> cu_window_seqlens = {0};
    size_t window_index_id = 0;
    
    const size_t spatial_merge_size = m_vision_encoder->get_processor_config().merge_size;
    const size_t spatial_merge_unit = spatial_merge_size * spatial_merge_size;
    const size_t vit_merger_window_size = m_vlm_config.vision_config_window_size / spatial_merge_size / m_vlm_config.vision_config_patch_size;
    
    for (const auto& grid_thw : grids_thw) {
        size_t grid_t = grid_thw.at(0);
        size_t grid_h = grid_thw.at(1);
        size_t grid_w = grid_thw.at(2);
        
        // Calculate merged grid dimensions
        size_t llm_grid_h = grid_h / spatial_merge_size;
        size_t llm_grid_w = grid_w / spatial_merge_size;
        
        // Calculate padding for making dimensions divisible by window size
        size_t pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size) % vit_merger_window_size;
        size_t pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size) % vit_merger_window_size;
        
        size_t num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
        size_t num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;
        
        // Create and fill indices matrix with sequential indices
        std::vector<std::vector<std::vector<int32_t>>> index_3d(grid_t, 
            std::vector<std::vector<int32_t>>(llm_grid_h, 
                std::vector<int32_t>(llm_grid_w, 0)));
        
        size_t index = 0;
        for (size_t t = 0; t < grid_t; ++t) {
            for (size_t h = 0; h < llm_grid_h; ++h) {
                for (size_t w = 0; w < llm_grid_w; ++w) {
                    index_3d[t][h][w] = index++;
                }
            }
        }
        
        // Pad the 3D array
        for (size_t t = 0; t < grid_t; ++t) {
            for (size_t h = 0; h < llm_grid_h; ++h) {
                index_3d[t][h].resize(llm_grid_w + pad_w, -100);
            }
            index_3d[t].resize(llm_grid_h + pad_h, std::vector<int32_t>(llm_grid_w + pad_w, -100));
        }
        
        // Process windows
        for (size_t t = 0; t < grid_t; ++t) {
            for (size_t wh = 0; wh < num_windows_h; ++wh) {
                for (size_t ww = 0; ww < num_windows_w; ++ww) {
                    // Count valid positions in this window
                    size_t valid_count = 0;
                    
                    for (size_t h = 0; h < vit_merger_window_size; ++h) {
                        for (size_t w = 0; w < vit_merger_window_size; ++w) {
                            size_t gh = wh * vit_merger_window_size + h;
                            size_t gw = ww * vit_merger_window_size + w;
                            
                            if (gh < llm_grid_h && gw < llm_grid_w) {
                                int32_t idx = index_3d[t][gh][gw];
                                if (idx != -100) {
                                    window_indices.push_back(idx + window_index_id);
                                    valid_count++;
                                }
                            }
                        }
                    }
                    
                    cu_window_seqlens.push_back(cu_window_seqlens.back() + valid_count * spatial_merge_unit);
                }
            }
        }
        
        window_index_id += grid_t * llm_grid_h * llm_grid_w;
    }
    
    ov::Tensor window_index_tensor{ov::element::i64, {window_indices.size()}};
    int64_t* window_index_data = window_index_tensor.data<int64_t>();
    
    for (size_t i = 0; i < window_indices.size(); ++i) {
        window_index_data[i] = static_cast<int64_t>(window_indices[i]);
    }
    
    return {window_index_tensor, cu_window_seqlens};
}

} // namespace ov::genai
