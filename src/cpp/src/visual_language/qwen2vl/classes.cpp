// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen2vl/classes.hpp"

#include "visual_language/clip.hpp"
#include "visual_language/embedding_model.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

// Chat template hardcodes char sequence instead of referring to tag values, so NATIVE_TAG is hardcoded as well.
std::string NATIVE_TAG = "<|vision_start|><|image_pad|><|vision_end|>";

} // namespace

namespace qwen2_vl_utils {

ImageSize smart_resize(size_t height, size_t width, size_t factor, size_t min_pixels, size_t max_pixels) {
    if (height < factor || width < factor) {
        OPENVINO_THROW("Height (" + std::to_string(height) + ") and width (" + std::to_string(width) + ") must be greater than factor (" + std::to_string(factor) + ")");
    }
    if (std::max(height, width) / std::min(height, width) > 200) {
        OPENVINO_THROW("Absolute aspect ratio must be smaller than 200");
    }

    size_t h_bar = std::round(static_cast<float>(height) / factor) * factor;
    size_t w_bar = std::round(static_cast<float>(width) / factor) * factor; 

    if (h_bar * w_bar > max_pixels) {
        double beta = std::sqrt((height * width) / static_cast<double>(max_pixels));
        h_bar = std::floor(height / beta / factor) * factor;
        w_bar = std::floor(width / beta / factor) * factor;
    } else if (h_bar * w_bar < min_pixels) {
        double beta = std::sqrt(min_pixels / static_cast<double>(height * width));
        h_bar = std::ceil(height * beta / factor) * factor;
        w_bar = std::ceil(width * beta / factor) * factor;
    }
    
    return ImageSize{h_bar, w_bar};
}

ov::Tensor reshape_image_patches(
    const ov::Tensor& patches,
    const size_t grid_t,
    const size_t grid_h,
    const size_t grid_w,
    const size_t channel,
    const size_t temporal_patch_size,
    const size_t patch_size,
    const size_t spatial_merge_size
) {
    ov::Shape output_shape{
        grid_t,                      
        temporal_patch_size,         
        channel,                     
        grid_h / spatial_merge_size, 
        spatial_merge_size,          
        patch_size,                  
        grid_w / spatial_merge_size, 
        spatial_merge_size,          
        patch_size                   
    };
    
    ov::Tensor reshaped_patches(patches.get_element_type(), output_shape);

    const float* input_data = patches.data<float>();
    float* output_data = reshaped_patches.data<float>();

    size_t input_idx = 0;
    
    for (size_t gt = 0; gt < output_shape.at(0); ++gt) {
        for (size_t tp = 0; tp < output_shape.at(1); ++tp) {
            for (size_t c = 0; c < output_shape.at(2); ++c) {
                for (size_t gh = 0; gh < output_shape.at(3); ++gh) {
                    for (size_t ms1 = 0; ms1 < output_shape.at(4); ++ms1) {
                        for (size_t p1 = 0; p1 < output_shape.at(5); ++p1) {
                            for (size_t gw = 0; gw < output_shape.at(6); ++gw) {
                                for (size_t ms2 = 0; ms2 < output_shape.at(7); ++ms2) {
                                    for (size_t p2 = 0; p2 < output_shape.at(8); ++p2) {
                                        size_t output_idx = gt;
                                        output_idx = output_idx * output_shape.at(1) + tp;
                                        output_idx = output_idx * output_shape.at(2) + c;
                                        output_idx = output_idx * output_shape.at(3) + gh;
                                        output_idx = output_idx * output_shape.at(4) + ms1;
                                        output_idx = output_idx * output_shape.at(5) + p1;
                                        output_idx = output_idx * output_shape.at(6) + gw;
                                        output_idx = output_idx * output_shape.at(7) + ms2;
                                        output_idx = output_idx * output_shape.at(8) + p2;

                                        output_data[output_idx] = input_data[input_idx];
                                        input_idx++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return reshaped_patches;
}
    
ov::Tensor transpose_image_patches(const ov::Tensor& reshaped_patches) {
    // Input dimensions order:  [0,1,2,3,4,5,6,7,8]
    // Output dimensions order: [0,3,6,4,7,2,1,5,8]
    auto input_shape = reshaped_patches.get_shape();
    
    ov::Shape output_shape = {
        input_shape.at(0), // grid_t
        input_shape.at(3), // grid_h / spatial_merge_size
        input_shape.at(6), // grid_w / spatial_merge_size
        input_shape.at(4), // spatial_merge_size
        input_shape.at(7), // spatial_merge_size
        input_shape.at(2), // channel
        input_shape.at(1), // temporal_patch_size
        input_shape.at(5), // patch_size
        input_shape.at(8)  // patch_size
    };

    ov::Tensor transposed_patches(reshaped_patches.get_element_type(), output_shape);
    
    const float* src = reshaped_patches.data<float>();
    float* dst = transposed_patches.data<float>();
    
    size_t shape_size = input_shape.size();
    std::vector<size_t> input_strides(shape_size);
    std::vector<size_t> output_strides(shape_size);
    
    input_strides[shape_size - 1] = 1;
    output_strides[shape_size - 1] = 1;
    for(int i = 7; i >= 0; i--) {
        input_strides[i] = input_strides[i+1] * input_shape[i+1];
        output_strides[i] = output_strides[i+1] * output_shape[i+1];
    }

    size_t total_elements = reshaped_patches.get_size();
    for(size_t idx = 0; idx < total_elements; idx++) {
        size_t remaining = idx;
        std::vector<size_t> input_indices(shape_size);
        for(int i = 0; i < shape_size; i++) {
            input_indices[i] = remaining / input_strides[i];
            remaining %= input_strides[i];
        }
        
        std::vector<size_t> output_indices = {
            input_indices.at(0),
            input_indices.at(3),
            input_indices.at(6),
            input_indices.at(4),
            input_indices.at(7),
            input_indices.at(2),
            input_indices.at(1),
            input_indices.at(5),
            input_indices.at(8)
        };
        
        size_t dst_idx = 0;
        for(int i = 0; i < shape_size; i++) {
            dst_idx += output_indices[i] * output_strides[i];
        }
        
        dst[dst_idx] = src[idx];
    }
    
    return transposed_patches;
}

std::pair<std::vector<ov::Tensor>, std::vector<std::array<size_t, 3>>> reorder_image_embeds_and_grid_thw(
    const std::vector<EncodedImage>& encoded_images,
    const std::vector<size_t>& images_sequence
) {
    std::vector<ov::Tensor> image_embeds;
    std::vector<std::array<size_t, 3>> images_grid_thw;
    image_embeds.reserve(encoded_images.size());
    images_grid_thw.reserve(encoded_images.size());
    
    for (const auto& encoded_image : encoded_images) {
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
        reordered_image_embeds.push_back(image_embeds.at(new_image_id));
        reordered_images_grid_thw.push_back(images_grid_thw.at(new_image_id));
    }

    return {reordered_image_embeds, reordered_images_grid_thw};
}
    
ov::Tensor get_attention_mask(const std::vector<std::array<size_t, 3>>& reordered_images_grid_thw) {
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
    return attention_mask;
}

ov::Tensor concatenate_image_embeds(const std::vector<ov::Tensor>& reordered_image_embeds) {
    ov::Tensor concatenated_embeds;
    if (reordered_image_embeds.size() == 1) {
        concatenated_embeds = reordered_image_embeds.at(0);
    } else {
        size_t total_length = 0;
        for (const auto& embed : reordered_image_embeds) {
            total_length += embed.get_shape().at(0);
        }
        size_t hidden_dim = reordered_image_embeds.at(0).get_shape().at(1);
        
        concatenated_embeds = ov::Tensor(reordered_image_embeds.at(0).get_element_type(), {total_length, hidden_dim});
        float* concat_data = concatenated_embeds.data<float>();
        
        size_t offset = 0;
        for (const auto& embed : reordered_image_embeds) {
            size_t embed_size = embed.get_shape().at(0) * embed.get_shape().at(1);
            std::memcpy(concat_data + offset, embed.data(), embed.get_byte_size());
            offset += embed_size;
        }
    }
    return concatenated_embeds;
}

ov::Tensor merge_text_and_image_embeddings(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds, 
    const ov::Tensor& processed_vision_embeds,
    const int64_t image_pad_token_id
) {
    ov::Tensor merged_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(merged_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());

    auto text_embeds_shape = text_embeds.get_shape();
    size_t batch_size = text_embeds_shape.at(0);
    size_t seq_length = text_embeds_shape.at(1);
    size_t hidden_size = text_embeds_shape.at(2);

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    float* merged_embeds_data = merged_embeds.data<float>();
    const float* vision_embeds_data = processed_vision_embeds.data<const float>();

    size_t vision_embed_idx = 0;
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (size_t seq_idx = 0; seq_idx < seq_length; ++seq_idx) {
            size_t flat_idx = batch_idx * seq_length + seq_idx;
            if (input_ids_data[flat_idx] == image_pad_token_id) {
                std::copy_n(
                    vision_embeds_data + vision_embed_idx * hidden_size,
                    hidden_size,
                    merged_embeds_data + flat_idx * hidden_size
                );
                ++vision_embed_idx;
            }
        }
    }
    return merged_embeds;
}
    
} // namespace qwen2vl_utils

EncodedImage VisionEncoderQwen2VL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ov::Shape image_shape = image.get_shape();
    auto original_height = image_shape.at(1);
    auto original_width = image_shape.at(2);

    ImageSize target_image_size = qwen2_vl_utils::smart_resize(
        original_height, 
        original_width, 
        config.patch_size * config.merge_size,
        config.min_pixels,
        config.max_pixels
    );

    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_u8 resized_image;
    bicubic_resize(input_image, resized_image, target_image_size.width, target_image_size.height);

    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
    clip_image_f32 normalized_image = clip_image_preprocess(ctx, resized_image);

    ov::Tensor patches = clip_image_f32_to_tensor(normalized_image);

    // For single patch tile it to match temporal_patch_size
    if (patches.get_shape().at(0) == 1) {
        auto orig_shape = patches.get_shape();
        ov::Tensor tiled_patches(patches.get_element_type(),
                                    {config.temporal_patch_size, orig_shape.at(1), orig_shape.at(2), orig_shape.at(3)});
        
        for (size_t i = 0; i < config.temporal_patch_size; i++) {
            std::memcpy(
                tiled_patches.data<float>() + i * patches.get_byte_size() / sizeof(float),
                patches.data<float>(),
                patches.get_byte_size()
            );
        }
        patches = std::move(tiled_patches);
    }

    auto patches_shape = patches.get_shape();
    size_t channel = patches_shape.at(1);
    
    size_t grid_t = patches_shape.at(0) / config.temporal_patch_size;
    size_t grid_h = target_image_size.height / config.patch_size;
    size_t grid_w = target_image_size.width / config.patch_size;

    ov::Tensor reshaped_patches = qwen2_vl_utils::reshape_image_patches(
        patches, grid_t, grid_h, grid_w, channel, config.temporal_patch_size, config.patch_size, config.merge_size
    );
    ov::Tensor transposed_patches = qwen2_vl_utils::transpose_image_patches(reshaped_patches);

    ov::Shape flattened_patches_shape{
        grid_t * grid_h * grid_w,
        channel * config.temporal_patch_size * config.patch_size * config.patch_size
    };
    ov::Tensor flattened_patches(transposed_patches.get_element_type(), flattened_patches_shape);
    std::memcpy(flattened_patches.data(), transposed_patches.data(), transposed_patches.get_byte_size());

    encoder.set_tensor("hidden_states", flattened_patches);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    ImageSize resized_source_size{grid_h, grid_w};

    return {std::move(image_features), resized_source_size};
}

InputsEmbedderQwen2VL::InputsEmbedderQwen2VL(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) {
    auto compiled_model = utils::singleton_core().compile_model(model_dir / "openvino_vision_embeddings_merger_model.xml", device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings merger model");
    m_ireq_queue_vision_embeddings_merger = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    
    // [CDPruner] Initialize CDPruner with hardcoded configuration
    ov::genai::cdpruner::Config cdpruner_config;
    cdpruner_config.num_visual_tokens = 300;  // Hardcoded 40% retention rate for Qwen2.5-VL
    cdpruner_config.relevance_weight = 0.5f;  // Balance between relevance and diversity
    cdpruner_config.enable_pruning = true;    // Enable pruning functionality
    cdpruner_config.device = device;          // Use same device as the model
    cdpruner_config.debug_mode = false;       // Disable debug output for production
    m_cdpruner = std::make_unique<ov::genai::cdpruner::CDPruner>(cdpruner_config);
}

InputsEmbedderQwen2VL::InputsEmbedderQwen2VL(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer, 
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    auto compiled_model = utils::singleton_core().compile_model(
        utils::get_model_weights_pair(models_map, "vision_embeddings_merger").first,
        utils::get_model_weights_pair(models_map, "vision_embeddings_merger").second,
        device,
        device_config
    );
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings merger model");
    m_ireq_queue_vision_embeddings_merger = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    
    // [CDPruner] Initialize CDPruner with hardcoded configuration
    ov::genai::cdpruner::Config cdpruner_config;
    cdpruner_config.num_visual_tokens = 550;  // Hardcoded 40% retention rate for Qwen2.5-VL (from ~1376 to 550)
    cdpruner_config.relevance_weight = 0.5f;  // Balance between relevance and diversity
    cdpruner_config.enable_pruning = true;    // Enable pruning functionality
    cdpruner_config.device = device;          // Use same device as the model
    cdpruner_config.debug_mode = false;       // Disable debug output for production
    m_cdpruner = std::make_unique<ov::genai::cdpruner::CDPruner>(cdpruner_config);
}

std::pair<std::string, std::vector<size_t>> InputsEmbedderQwen2VL::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG, base_id, images.size());
        std::vector<std::array<size_t, 3>> images_grid_thw;
    images_grid_thw.reserve(images.size());
    
    for (const auto& encoded_image : images) {
        size_t grid_t = 1;
        size_t grid_h = encoded_image.resized_source_size.height;
        size_t grid_w = encoded_image.resized_source_size.width;
        images_grid_thw.push_back({grid_t, grid_h, grid_w});
    }

    for (size_t new_image_id : images_sequence) {
        auto [grid_t, grid_h, grid_w] = images_grid_thw.at(new_image_id - base_id);
        size_t merge_length = std::pow(m_vision_encoder->get_processor_config().merge_size, 2);
        size_t num_image_pad_tokens = grid_t * grid_h * grid_w / merge_length;

        std::string expanded_tag = m_vlm_config.vision_start_token;
        for (int i = 0; i < num_image_pad_tokens; i++) {
            expanded_tag += m_vlm_config.image_pad_token;
        }
        expanded_tag += m_vlm_config.vision_end_token;
        unified_prompt.replace(unified_prompt.find(NATIVE_TAG), NATIVE_TAG.length(), expanded_tag);
    }
    return {std::move(unified_prompt), std::move(images_sequence)};
}

ov::Tensor InputsEmbedderQwen2VL::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    std::vector<std::array<size_t, 3>> images_grid_thw;
    images_grid_thw.reserve(images.size());
    for (const auto& encoded_image : images) {
        size_t grid_t = 1;
        size_t grid_h = encoded_image.resized_source_size.height;
        size_t grid_w = encoded_image.resized_source_size.width;
        images_grid_thw.push_back({grid_t, grid_h, grid_w});
    }

    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    ov::Tensor text_embeds;
    {
        // Acquire request, run inference, then copy the result to safeguard against later reuse
        CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
        EmbeddingsRequest& req = embeddings_request_guard.get();
        ov::Tensor tmp_embeds = m_embedding->infer(req, input_ids);

        // Deep-copy to ensure the data remains valid after the request is released
        text_embeds = ov::Tensor(tmp_embeds.get_element_type(), tmp_embeds.get_shape());
        std::memcpy(text_embeds.data(), tmp_embeds.data(), tmp_embeds.get_byte_size());
    } // Request released here

    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor encoded_vision_start_token = m_tokenizer.encode(m_vlm_config.vision_start_token, ov::genai::add_special_tokens(false)).input_ids;
    ov::Tensor encoded_image_pad_token = m_tokenizer.encode(m_vlm_config.image_pad_token, ov::genai::add_special_tokens(false)).input_ids;
    ov::Tensor encoded_vision_end_token = m_tokenizer.encode(m_vlm_config.vision_end_token, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t vision_start_token_id = encoded_vision_start_token.data<int64_t>()[encoded_vision_start_token.get_size() - 1];
    int64_t image_pad_token_id = encoded_image_pad_token.data<int64_t>()[encoded_image_pad_token.get_size() - 1];
    int64_t vision_end_token_id = encoded_vision_end_token.data<int64_t>()[encoded_vision_end_token.get_size() - 1];

    m_position_ids = create_position_ids(input_ids, images_grid_thw, images_sequence, 0, vision_start_token_id);

    int64_t position_ids_max_element = *std::max_element(m_position_ids.data<int64_t>(), m_position_ids.data<int64_t>() + m_position_ids.get_size());
    m_rope_delta = position_ids_max_element + 1 - static_cast<int64_t>(input_ids.get_shape().at(1));

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }
    ov::Tensor merged_image_embeddings_tensor;
    if (recalculate_merged_embeddings) {
        m_merged_image_embeddings = run_image_embeddings_merger(images, images_sequence);
    }
    merged_image_embeddings_tensor = m_merged_image_embeddings;

    // [CDPruner] Apply token pruning if CDPruner is enabled and images are present
    size_t original_visual_tokens = 0;
    size_t pruned_visual_tokens = 0;

    auto pruner_config = m_cdpruner->get_config();
    bool pruner_enabled = pruner_config.enable_pruning;
    
    if (m_cdpruner && pruner_enabled && !images.empty()) {
        // Store original visual token count for position adjustment
        original_visual_tokens = merged_image_embeddings_tensor.get_shape()[0];
        
        // Extract text features for CDPruner using the implemented function
        ov::Tensor text_features = extract_text_features_for_cdpruner(input_ids, 
                                                                     image_pad_token_id,
                                                                     vision_start_token_id,
                                                                     vision_end_token_id);
        
        // Convert visual features for CDPruner using the implemented function
        ov::Tensor visual_features = convert_visual_features_for_cdpruner(merged_image_embeddings_tensor);
        
        // Apply CDPruner to get pruned visual tokens
        ov::Tensor pruned_visual_features = m_cdpruner->apply_pruning(visual_features, text_features);
        
        // [CDPruner] Convert back from 3D [1, num_tokens, hidden_size] to 2D [num_tokens, hidden_size]
        // to match the expected input format for merge_text_and_image_embeddings
        ov::Shape pruned_shape = pruned_visual_features.get_shape();
        pruned_visual_tokens = pruned_shape[1];  // num_tokens dimension
        size_t hidden_size = pruned_shape[2];           // hidden_size dimension
        
        // Create 2D tensor with shape [num_tokens, hidden_size]
        ov::Tensor pruned_2d_tensor(pruned_visual_features.get_element_type(), {pruned_visual_tokens, hidden_size});
        
        // Copy data from 3D [1, num_tokens, hidden_size] to 2D [num_tokens, hidden_size]
        const float* src_data = pruned_visual_features.data<const float>();
        float* dst_data = pruned_2d_tensor.data<float>();
        size_t total_elements = pruned_visual_tokens * hidden_size;
        std::memcpy(dst_data, src_data, total_elements * sizeof(float));
        
        merged_image_embeddings_tensor = pruned_2d_tensor;
        
        // Adjust position_ids if pruning occurred
        if (original_visual_tokens != pruned_visual_tokens) {
            m_position_ids = adjust_position_ids_for_pruning(m_position_ids, input_ids,
                                                           original_visual_tokens, pruned_visual_tokens,
                                                           vision_start_token_id, image_pad_token_id);
            
            // Recalculate rope_delta based on adjusted position_ids
            int64_t position_ids_max_element = *std::max_element(
                m_position_ids.data<int64_t>(), 
                m_position_ids.data<int64_t>() + m_position_ids.get_size());
            
            // Calculate new sequence length after pruning
            size_t tokens_removed = original_visual_tokens - pruned_visual_tokens;
            size_t new_sequence_length = input_ids.get_shape().at(1) - tokens_removed;
            
            m_rope_delta = position_ids_max_element + 1 - static_cast<int64_t>(new_sequence_length);
        }
    }

    // [CDPruner] Handle pruned visual tokens case
    if (m_cdpruner && pruner_enabled  && !images.empty() && original_visual_tokens != pruned_visual_tokens) {
        // Visual tokens have been pruned, need to create new merged embeddings with correct dimensions
        return merge_text_and_image_embeddings_with_pruning(input_ids, text_embeds, merged_image_embeddings_tensor, image_pad_token_id, original_visual_tokens);
    } else {
        // No pruning or no images, use original function
        return qwen2_vl_utils::merge_text_and_image_embeddings(input_ids, text_embeds, merged_image_embeddings_tensor, image_pad_token_id);
    }
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderQwen2VL::get_position_ids(const size_t inputs_embeds_size, const size_t history_size) {
    if (history_size != 0) {
        ov::Tensor position_ids{ov::element::i64, {3, 1, inputs_embeds_size}};
        int64_t new_pos_id = static_cast<int64_t>(history_size + m_rope_delta);
        for (size_t dim = 0; dim < 3; ++dim) {
            int64_t* pos_data = position_ids.data<int64_t>() + dim * inputs_embeds_size;
            std::iota(pos_data, pos_data + inputs_embeds_size, new_pos_id);
        }
        return {position_ids, m_rope_delta};
    }
    return {m_position_ids, m_rope_delta};
}

void InputsEmbedderQwen2VL::start_chat(const std::string& system_message) {
    IInputsEmbedder::start_chat(system_message);
    m_position_ids = ov::Tensor();
    m_rope_delta = 0;
}

void InputsEmbedderQwen2VL::finish_chat() {
    IInputsEmbedder::finish_chat();
    m_position_ids = ov::Tensor();
    m_rope_delta = 0;
    m_merged_image_embeddings = ov::Tensor();
    // [CDPruner] CDPruner is stateless, no cleanup needed
}

ov::Tensor InputsEmbedderQwen2VL::run_image_embeddings_merger(
    const std::vector<EncodedImage>& images,
    const std::vector<size_t>& images_sequence
) {
    auto [reordered_image_embeds, reordered_images_grid_thw] = qwen2_vl_utils::reorder_image_embeds_and_grid_thw(images, images_sequence);

    ov::Tensor concatenated_embeds = qwen2_vl_utils::concatenate_image_embeds(reordered_image_embeds);
    ov::Tensor attention_mask = qwen2_vl_utils::get_attention_mask(reordered_images_grid_thw);
    ov::Tensor rotary_pos_emb = get_rotary_pos_emb(reordered_images_grid_thw);

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& vision_embeddings_merger = infer_request_guard.get();
    vision_embeddings_merger.set_tensor("hidden_states", concatenated_embeds);
    vision_embeddings_merger.set_tensor("attention_mask", attention_mask);
    vision_embeddings_merger.set_tensor("rotary_pos_emb", rotary_pos_emb);
    vision_embeddings_merger.infer();
    ov::Tensor processed_vision_embeds = vision_embeddings_merger.get_output_tensor();

    ov::Tensor res = ov::Tensor(processed_vision_embeds.get_element_type(), processed_vision_embeds.get_shape());
    std::memcpy(res.data(), processed_vision_embeds.data(), processed_vision_embeds.get_byte_size());
    return res;
}

ov::Tensor InputsEmbedderQwen2VL::get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw) {
    const size_t spatial_merge_size = m_vision_encoder->get_processor_config().merge_size;

    std::vector<std::vector<size_t>> all_pos_ids;
    size_t total_positions = 0;
    size_t max_grid_size = 0;

    for (const auto& grid_thw : grids_thw) {
        size_t t = grid_thw.at(0);
        size_t h = grid_thw.at(1);
        size_t w = grid_thw.at(2);

        total_positions += t * h * w;
        max_grid_size = std::max({max_grid_size, h, w});
        
        // Create height position IDs
        std::vector<size_t> hpos_ids(h * w);
        for (size_t hi = 0; hi < h; ++hi) {
            for (size_t wi = 0; wi < w; ++wi) {
                size_t idx = hi * w + wi;
                hpos_ids[idx] = hi;
            }
        }

        // Reshape hpos_ids according to spatial merge size
        std::vector<size_t> reshaped_hpos;
        size_t h_blocks = h / spatial_merge_size;
        size_t w_blocks = w / spatial_merge_size;
        reshaped_hpos.reserve(h * w);

        for (size_t hb = 0; hb < h_blocks; ++hb) {
            for (size_t wb = 0; wb < w_blocks; ++wb) {
                for (size_t hs = 0; hs < spatial_merge_size; ++hs) {
                    for (size_t ws = 0; ws < spatial_merge_size; ++ws) {
                        reshaped_hpos.push_back(hb * spatial_merge_size + hs);
                    }
                }
            }
        }

        // Create width position IDs
        std::vector<size_t> wpos_ids(h * w);
        for (size_t hi = 0; hi < h; ++hi) {
            for (size_t wi = 0; wi < w; ++wi) {
                size_t idx = hi * w + wi;
                wpos_ids[idx] = wi;
            }
        }

        // Reshape wpos_ids according to spatial merge size
        std::vector<size_t> reshaped_wpos;
        reshaped_wpos.reserve(h * w);

        for (size_t hb = 0; hb < h_blocks; ++hb) {
            for (size_t wb = 0; wb < w_blocks; ++wb) {
                for (size_t hs = 0; hs < spatial_merge_size; ++hs) {
                    for (size_t ws = 0; ws < spatial_merge_size; ++ws) {
                        reshaped_wpos.push_back(wb * spatial_merge_size + ws);
                    }
                }
            }
        }

        // Stack and repeat for each t
        for (size_t i = 0; i < t; ++i) {
            for (size_t j = 0; j < reshaped_hpos.size(); ++j) {
                all_pos_ids.push_back({reshaped_hpos[j], reshaped_wpos[j]});
            }
        }
    }

    // Calculate rotary embeddings for max_grid_size
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& vision_embeddings_merger = infer_request_guard.get();
    const size_t dim = vision_embeddings_merger.get_tensor("rotary_pos_emb").get_shape().at(1);
    const float theta = 10000.0f;
    
    std::vector<float> inv_freq(dim / 2);
    for (size_t i = 0; i < dim / 2; ++i) {
        inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(i) / static_cast<float>(dim / 2));
    }

    std::vector<std::vector<float>> freqs(max_grid_size);
    for (size_t i = 0; i < max_grid_size; ++i) {
        freqs[i].resize(dim / 2);
        for (size_t j = 0; j < dim / 2; ++j) {
            freqs[i][j] = static_cast<float>(i) * inv_freq[j];
        }
    }

    ov::Tensor rotary_pos_emb(ov::element::f32, {all_pos_ids.size(), dim});
    float* output_data = rotary_pos_emb.data<float>();

    for (size_t i = 0; i < all_pos_ids.size(); ++i) {
        const auto& pos = all_pos_ids.at(i);
        size_t h_idx = pos.at(0);
        size_t w_idx = pos.at(1);
        std::copy_n(freqs[h_idx].begin(), dim / 2, output_data + i * dim);
        std::copy_n(freqs[w_idx].begin(), dim / 2, output_data + i * dim + dim / 2);
    }

    return rotary_pos_emb;
}

ov::Tensor InputsEmbedderQwen2VL::create_position_ids(
    const ov::Tensor& input_ids_tensor,
    const std::vector<std::array<size_t, 3>>& images_grid_thw,
    const std::vector<size_t>& images_sequence,
    const size_t image_id,
    const int64_t vision_start_token_id) {
    const size_t spatial_merge_size = m_vision_encoder->get_processor_config().merge_size;

    std::vector<std::array<size_t, 3>> reordered_images_grid_thw;
    for (size_t new_image_id : images_sequence) {
        reordered_images_grid_thw.push_back(images_grid_thw.at(new_image_id - image_id));
    }
    
    const int64_t* input_ids = input_ids_tensor.data<int64_t>();
    size_t batch_size = input_ids_tensor.get_shape().at(0);
    size_t seq_len = input_ids_tensor.get_shape().at(1);

    std::vector<size_t> vision_start_indices;
    for (size_t i = 0; i < seq_len; ++i) {
        if (input_ids[i] == vision_start_token_id) {
            vision_start_indices.push_back(i);
        }
    }

    ov::Tensor position_ids{ov::element::i64, {3, batch_size, seq_len}};
    int64_t* pos_data = position_ids.data<int64_t>();
    
    size_t st = 0;
    int64_t next_pos = 0;
    size_t grid_idx = 0;

    for (size_t i = 0; i < vision_start_indices.size(); ++i) {
        size_t ed = vision_start_indices.at(i);

        // Process text tokens before image
        if (st < ed) {
            for (size_t pos = st; pos < ed; ++pos) {
                pos_data[pos] = next_pos;               // temporal
                pos_data[seq_len + pos] = next_pos;     // height
                pos_data[2 * seq_len + pos] = next_pos; // width
                next_pos++;
            }
        }

        // Process image start token
        pos_data[ed] = next_pos;               // temporal
        pos_data[seq_len + ed] = next_pos;     // height
        pos_data[2 * seq_len + ed] = next_pos; // width
        next_pos++;
        ed++;

        // Process image token with grid
        if (grid_idx < reordered_images_grid_thw.size()) {
            const auto& grid = reordered_images_grid_thw.at(grid_idx);
            size_t llm_grid_h = grid.at(1) / spatial_merge_size;
            size_t llm_grid_w = grid.at(2) / spatial_merge_size;
            size_t ed_image = ed + llm_grid_h * llm_grid_w;

            // Fill temporal dimension
            std::fill_n(pos_data + ed, llm_grid_h * llm_grid_w, next_pos);

            // Fill height and width dimensions
            int64_t* height_data = pos_data + seq_len + ed;
            int64_t* width_data = pos_data + 2 * seq_len + ed;
            for (size_t h = 0; h < llm_grid_h; ++h) {
                std::fill_n(height_data + h * llm_grid_w, llm_grid_w, next_pos + h);
                for (size_t w = 0; w < llm_grid_w; ++w) {
                    width_data[h * llm_grid_w + w] = next_pos + w;
                }
            }

            next_pos += std::max(llm_grid_h, llm_grid_w);
            st = ed_image;
            grid_idx++;
        }
    }

    // Process remaining text tokens
    if (st < seq_len) {
        for (size_t pos = st; pos < seq_len; ++pos) {
            pos_data[pos] = next_pos;               // temporal
            pos_data[seq_len + pos] = next_pos;     // height
            pos_data[2 * seq_len + pos] = next_pos; // width
            next_pos++;
        }
    }

    return position_ids;
}

// [CDPruner] Text feature extraction functions implementation

std::vector<int64_t> InputsEmbedderQwen2VL::extract_instruction_tokens(const ov::Tensor& input_ids,
                                                                      int64_t image_pad_token_id,
                                                                      int64_t vision_start_token_id, 
                                                                      int64_t vision_end_token_id) {
    std::vector<int64_t> instruction_tokens;
    
    // Get input_ids data
    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    size_t seq_len = input_ids.get_shape()[1]; // [batch_size, seq_len]
    
    bool inside_vision_region = false;
    
    // Iterate through the sequence to extract instruction tokens
    for (size_t i = 0; i < seq_len; ++i) {
        int64_t current_token = input_ids_data[i];
        
        // Check for vision start token
        if (current_token == vision_start_token_id) {
            inside_vision_region = true;
            continue;
        }
        
        // Check for vision end token
        if (current_token == vision_end_token_id) {
            inside_vision_region = false;
            continue;
        }
        
        // Skip if inside vision region or if it's an image pad token
        if (inside_vision_region || current_token == image_pad_token_id) {
            continue;
        }
        
        // This is a valid instruction token
        instruction_tokens.push_back(current_token);
    }
    
    return instruction_tokens;
}

ov::Tensor InputsEmbedderQwen2VL::extract_text_features_for_cdpruner(const ov::Tensor& input_ids, 
                                                                 int64_t image_pad_token_id,
                                                                 int64_t vision_start_token_id,
                                                                 int64_t vision_end_token_id) {
    // Extract instruction tokens
    std::vector<int64_t> instruction_tokens = extract_instruction_tokens(
        input_ids, image_pad_token_id, vision_start_token_id, vision_end_token_id);
    
    if (instruction_tokens.empty()) {
        // If no instruction tokens found, create a zero tensor with proper dimensions
        ov::Tensor zero_embedding(ov::element::f32, {1, m_vlm_config.hidden_size});
        std::memset(zero_embedding.data<float>(), 0, zero_embedding.get_byte_size());
        return zero_embedding;
    }
    
    // [OPTIMIZED] Create batch tensor for all instruction tokens
    size_t num_tokens = instruction_tokens.size();
    ov::Tensor instruction_tokens_tensor(ov::element::i64, {1, num_tokens});
    int64_t* tokens_data = instruction_tokens_tensor.data<int64_t>();
    
    // Copy instruction tokens to tensor
    for (size_t i = 0; i < num_tokens; ++i) {
        tokens_data[i] = instruction_tokens[i];
    }
    
    // Acquire a fresh EmbeddingsRequest for this independent inference
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& local_req = embeddings_request_guard.get();

    ov::Tensor batch_embeddings = m_embedding->infer(local_req, instruction_tokens_tensor, false);
    
    // [OPTIMIZED] Calculate average pooling across sequence dimension
    ov::Tensor avg_embedding(ov::element::f32, {1, m_vlm_config.hidden_size});
    float* avg_data = avg_embedding.data<float>();
    const float* batch_data = batch_embeddings.data<const float>();
    
    // Initialize to zero
    std::memset(avg_data, 0, avg_embedding.get_byte_size());
    
    // Sum across all tokens: Σ_{j=1 to N} e_j
    for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
        const float* token_embed = batch_data + token_idx * m_vlm_config.hidden_size;
        for (size_t dim = 0; dim < m_vlm_config.hidden_size; ++dim) {
            avg_data[dim] += token_embed[dim];
        }
    }
    
    // Calculate average: H_q = (1/N) * Σ_{j=1 to N} e_j
    float num_tokens_f = static_cast<float>(num_tokens);
    for (size_t dim = 0; dim < m_vlm_config.hidden_size; ++dim) {
        avg_data[dim] /= num_tokens_f;
    }
    
    return avg_embedding;
}

// [CDPruner] Position encoding adjustment function for pruning
ov::Tensor InputsEmbedderQwen2VL::adjust_position_ids_for_pruning(const ov::Tensor& original_position_ids,
                                                                 const ov::Tensor& input_ids,
                                                                 size_t original_visual_tokens,
                                                                 size_t pruned_visual_tokens,
                                                                 int64_t vision_start_token_id,
                                                                 int64_t image_pad_token_id) {
    // If no pruning occurred, return original position_ids
    if (original_visual_tokens == pruned_visual_tokens) {
        return original_position_ids;
    }
    
    // Calculate the reduction in visual tokens
    size_t tokens_removed = original_visual_tokens - pruned_visual_tokens;
    
    // Get input_ids data and dimensions
    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    size_t batch_size = input_ids.get_shape().at(0);
    size_t seq_len = input_ids.get_shape().at(1);
    
    // Get original position_ids data and create new position_ids
    const int64_t* orig_pos_data = original_position_ids.data<const int64_t>();
    ov::Tensor adjusted_position_ids(original_position_ids.get_element_type(), original_position_ids.get_shape());
    int64_t* adj_pos_data = adjusted_position_ids.data<int64_t>();
    
    // Process each batch
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        // Find vision regions in this batch (image_pad_token sequences)
        std::vector<std::pair<size_t, size_t>> vision_regions; // [start_idx, end_idx)
        bool in_vision_region = false;
        size_t vision_start_idx = 0;
        
        for (size_t i = 0; i < seq_len; ++i) {
            size_t input_idx = batch_idx * seq_len + i;
            
            if (input_ids_data[input_idx] == vision_start_token_id) {
                in_vision_region = true;
                vision_start_idx = i + 1; // Start after vision_start_token
            } else if (in_vision_region && input_ids_data[input_idx] != image_pad_token_id) {
                // End of vision region (found non-image_pad token)
                in_vision_region = false;
                vision_regions.push_back({vision_start_idx, i});
            }
        }
        
        // Handle case where vision region extends to end of sequence
        if (in_vision_region) {
            vision_regions.push_back({vision_start_idx, seq_len});
        }
        
        // Copy and adjust position_ids for each dimension (temporal, height, width)
        for (size_t dim = 0; dim < 3; ++dim) {
            size_t dim_offset = dim * batch_size * seq_len + batch_idx * seq_len;
            
            for (size_t i = 0; i < seq_len; ++i) {
                size_t pos_idx = dim_offset + i;
                int64_t original_pos = orig_pos_data[pos_idx];
                
                // Check if this position is after any vision region
                bool is_after_vision = false;
                for (const auto& region : vision_regions) {
                    if (i >= region.second) { // Position is after this vision region
                        is_after_vision = true;
                        break;
                    }
                }
                
                // Adjust position if it's after vision regions
                // For text tokens after vision, subtract the number of removed visual tokens
                if (is_after_vision) {
                    adj_pos_data[pos_idx] = original_pos - static_cast<int64_t>(tokens_removed);
                } else {
                    // For tokens before or within vision regions, keep original position
                    adj_pos_data[pos_idx] = original_pos;
                }
            }
        }
    }
    
    return adjusted_position_ids;
}

ov::Tensor InputsEmbedderQwen2VL::convert_visual_features_for_cdpruner(const ov::Tensor& merged_image_embeddings) {
    // Convert from [num_patches, embedding_dim] to [1, num_patches, embedding_dim]
    ov::Shape original_shape = merged_image_embeddings.get_shape();
    size_t num_patches = original_shape[0];
    size_t embedding_dim = original_shape[1];
    
    // Create new tensor with batch dimension
    ov::Shape new_shape = {1, num_patches, embedding_dim};
    ov::Tensor visual_features(merged_image_embeddings.get_element_type(), new_shape);
    
    // Copy data
    const float* src_data = merged_image_embeddings.data<const float>();
    float* dst_data = visual_features.data<float>();
    
    size_t total_elements = num_patches * embedding_dim;
    std::memcpy(dst_data, src_data, total_elements * sizeof(float));
    
    return visual_features;
}

// [CDPruner] Create merged embeddings for pruned visual tokens
ov::Tensor InputsEmbedderQwen2VL::merge_text_and_image_embeddings_with_pruning(const ov::Tensor& input_ids,
                                                                 const ov::Tensor& text_embeds,
                                                                 const ov::Tensor& pruned_vision_embeds,
                                                                 int64_t image_pad_token_id,
                                                                 size_t original_visual_tokens) {
    auto text_embeds_shape = text_embeds.get_shape();
    size_t batch_size = text_embeds_shape.at(0);
    size_t original_seq_length = text_embeds_shape.at(1);  // original sequence length (text + original visual)
    size_t hidden_size = text_embeds_shape.at(2);
    
    // Calculate new sequence length after pruning
    size_t pruned_visual_tokens = pruned_vision_embeds.get_shape()[0];  // dynamic pruned count
    size_t tokens_removed = original_visual_tokens - pruned_visual_tokens;
    size_t new_seq_length = original_seq_length - tokens_removed;
    
    // Create new merged embeddings tensor with correct dimensions
    ov::Tensor merged_embeds(text_embeds.get_element_type(), {batch_size, new_seq_length, hidden_size});
    
    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    const float* text_embeds_data = text_embeds.data<const float>();
    const float* vision_embeds_data = pruned_vision_embeds.data<const float>();
    float* merged_embeds_data = merged_embeds.data<float>();
    
    size_t vision_embed_idx = 0;
    size_t output_idx = 0;
    
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (size_t seq_idx = 0; seq_idx < original_seq_length; ++seq_idx) {
            size_t input_flat_idx = batch_idx * original_seq_length + seq_idx;
            
            if (input_ids_data[input_flat_idx] == image_pad_token_id) {
                // This is a visual token position
                if (vision_embed_idx < pruned_visual_tokens) {
                    // Copy from pruned visual embeddings
                    std::copy_n(
                        vision_embeds_data + vision_embed_idx * hidden_size,
                        hidden_size,
                        merged_embeds_data + output_idx * hidden_size
                    );
                    output_idx++;
                }
                vision_embed_idx++;
            } else {
                // This is a text token, copy from text_embeds
                std::copy_n(
                    text_embeds_data + input_flat_idx * hidden_size,
                    hidden_size,
                    merged_embeds_data + output_idx * hidden_size
                );
                output_idx++;
            }
        }
    }
    
    return merged_embeds;
}

} // namespace ov::genai
