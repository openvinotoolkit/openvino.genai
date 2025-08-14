// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/llava/classes.hpp"

#include <algorithm>
#include <cmath>
#include "visual_language/clip.hpp"
#include "visual_language/embedding_model.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "circular_buffer_queue.hpp"
#include "utils.hpp"

namespace ov::genai {

clip_image_f32 preprocess_clip_image_llava(const clip_image_u8& image, const ProcessorConfig& config) {
    // Resize
    clip_image_u8 resized_image;
    int target_size = config.size_shortest_edge;
    float scale = static_cast<float>(target_size) / std::min(image.nx, image.ny);
    int new_width = static_cast<int>(image.nx * scale);
    int new_height = static_cast<int>(image.ny * scale);
    bicubic_resize(image, resized_image, new_width, new_height);

    // Center crop
    clip_image_u8 cropped_image;
    int crop_height = config.crop_size_height;
    int crop_width = config.crop_size_width;
    int start_x = (resized_image.nx - crop_width) / 2;
    int start_y = (resized_image.ny - crop_height) / 2;

    cropped_image.nx = crop_width;
    cropped_image.ny = crop_height;
    cropped_image.buf.resize(3 * crop_width * crop_height);

    for (int y = 0; y < crop_height; ++y) {
        for (int x = 0; x < crop_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                cropped_image.buf[(y * crop_width + x) * 3 + c] =
                    resized_image.buf[((start_y + y) * resized_image.nx + (start_x + x)) * 3 + c];
            }
        }
    }

    // Normalize
    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    clip_image_f32 normalized_image = clip_image_preprocess(ctx, cropped_image);
    return normalized_image;
}

namespace {

ov::Tensor get_pixel_values_llava(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_f32 preprocessed_image = preprocess_clip_image_llava(input_image, config);
    return clip_image_f32_to_tensor(preprocessed_image);
}

} // namespace

// VisionEncoderLLaVA constructors and methods
VisionEncoderLLaVA::VisionEncoderLLaVA(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap properties) : VisionEncoder(model_dir, device, properties) {
    
    // Load VLM config
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
    
    // Initialize text processing components
    initialize_text_components(model_dir, device, properties);
}

VisionEncoderLLaVA::VisionEncoderLLaVA(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap properties) : VisionEncoder(models_map, config_dir_path, device, properties) {
    
    // Load VLM config
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
    
    // Initialize text processing components
    initialize_text_components(models_map, config_dir_path, device, properties);
}

void VisionEncoderLLaVA::initialize_text_components(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap& properties) {
    
    try {
        // Initialize tokenizer
        m_tokenizer = Tokenizer(model_dir, properties);
        
        // Initialize text embedding model
        // First try to find dedicated text embeddings model
        auto embeddings_model_path = model_dir / "openvino_text_embeddings_model.xml";
        if (std::filesystem::exists(embeddings_model_path)) {
            try {
                m_text_embedding_model = EmbeddingsModel::create(model_dir, 1.0f, device, properties);
            } catch (const std::exception& e) {
                // If text embeddings model fails, try language model
                auto language_model_path = model_dir / "openvino_language_model.xml";
                if (std::filesystem::exists(language_model_path)) {
                    try {
                        m_text_embedding_model = EmbeddingsModel::create(model_dir, 1.0f, device, properties);
                    } catch (const std::exception& e2) {
                        // Both failed, disable text processing
                        m_text_embedding_model.reset();
                    }
                }
            }
        }
        
        // Validate that both components are available
        if (!m_tokenizer.has_value() || !m_text_embedding_model) {
            throw std::runtime_error("Text processing components not available");
        }
        
    } catch (const std::exception& e) {
        // Text processing components are optional for backward compatibility
        // If they fail to load, CDPruner will be disabled
        m_tokenizer.reset();
        m_text_embedding_model.reset();
    }
}

void VisionEncoderLLaVA::initialize_text_components(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap& properties) {
    
    try {
        // Initialize tokenizer
        m_tokenizer = Tokenizer(config_dir_path, properties);
        
        // Initialize text embedding model from models map
        bool embedding_model_initialized = false;
        
        // First try text_embeddings model
        if (models_map.find("text_embeddings") != models_map.end()) {
            try {
                const auto& text_model_pair = utils::get_model_weights_pair(models_map, "text_embeddings");
                m_text_embedding_model = EmbeddingsModel::create(
                    text_model_pair.first,
                    text_model_pair.second,
                    1.0f, // scale_emb
                    device,
                    properties
                );
                embedding_model_initialized = true;
            } catch (const std::exception& e) {
                // Continue to try language model
            }
        }
        
        // Fallback to language model for embeddings
        if (!embedding_model_initialized && models_map.find("language") != models_map.end()) {
            try {
                const auto& lang_model_pair = utils::get_model_weights_pair(models_map, "language");
                m_text_embedding_model = EmbeddingsModel::create(
                    lang_model_pair.first,
                    lang_model_pair.second,
                    1.0f, // scale_emb
                    device,
                    properties
                );
                embedding_model_initialized = true;
            } catch (const std::exception& e) {
                // Both failed
            }
        }
        
        // Validate that both components are available
        if (!m_tokenizer.has_value() || !embedding_model_initialized) {
            throw std::runtime_error("Text processing components not available");
        }
        
    } catch (const std::exception& e) {
        // Text processing components are optional for backward compatibility
        m_tokenizer.reset();
        m_text_embedding_model.reset();
    }
}

ov::Tensor VisionEncoderLLaVA::extract_text_features(const std::string& text_prompt) {
    if (!m_tokenizer.has_value() || !m_text_embedding_model) {
        // Return empty tensor if text processing is not available
        // Use a more generic shape that works with CDPruner
        return ov::Tensor(ov::element::f32, {1, 1, 768}); // Default hidden size
    }
    
    try {
        // Tokenize text prompt
        auto encoded_result = m_tokenizer.value().encode(text_prompt, ov::genai::add_special_tokens(true));
        ov::Tensor input_ids = encoded_result.input_ids;
        
        // Validate input_ids shape
        if (input_ids.get_shape().size() != 2) {
            throw std::runtime_error("Invalid input_ids shape from tokenizer");
        }
        
        // Get text embeddings
        CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(
            m_text_embedding_model->get_request_queue().get());
        EmbeddingsRequest& req = embeddings_request_guard.get();
        ov::Tensor text_features = m_text_embedding_model->infer(req, input_ids);
        
        // Validate output shape
        auto output_shape = text_features.get_shape();
        if (output_shape.size() != 3) {
            throw std::runtime_error("Invalid text features shape from embedding model");
        }
        
        // CDPruner expects text features to be aggregated across sequence length
        // If we have [batch, seq_len, hidden_size], we need to aggregate to [batch, hidden_size]
        if (output_shape[1] > 1) {
            // Mean pooling across sequence length
            ov::Tensor aggregated_features(text_features.get_element_type(), {output_shape[0], output_shape[2]});
            const float* input_data = text_features.data<const float>();
            float* output_data = aggregated_features.data<float>();
            
            for (size_t b = 0; b < output_shape[0]; ++b) {
                for (size_t h = 0; h < output_shape[2]; ++h) {
                    float sum = 0.0f;
                    for (size_t s = 0; s < output_shape[1]; ++s) {
                        sum += input_data[b * output_shape[1] * output_shape[2] + s * output_shape[2] + h];
                    }
                    output_data[b * output_shape[2] + h] = sum / static_cast<float>(output_shape[1]);
                }
            }
            return aggregated_features;
        }
        
        return text_features;
        
    } catch (const std::exception& e) {
        // Return dummy tensor on error with proper shape
        // Get the expected hidden size from VLM config if available
        size_t hidden_size = m_vlm_config.hidden_size > 0 ? m_vlm_config.hidden_size : 768;
        return ov::Tensor(ov::element::f32, {1, hidden_size});
    }
}

EncodedImage VisionEncoderLLaVA::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values = get_pixel_values_llava(image, config);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    ImageSize resized_source_size{config.crop_size_height / config.patch_size, config.crop_size_width / config.patch_size};

    return {std::move(image_features), resized_source_size};
}

EncodedImage VisionEncoderLLaVA::encode_with_pruning(
    const ov::Tensor& image,
    const std::string& text_prompt,
    const size_t visual_tokens_percentage,
    const ov::AnyMap& config_map) {
    
    // First, get the full visual features using the standard encode method
    EncodedImage full_encoded_image = encode(image, config_map);

    // If CDPruner is not available or text processing failed, return full features
    if (!is_pruning_available() || !m_tokenizer.has_value() || !m_text_embedding_model) {
        return full_encoded_image;
    }

    // Validate visual_tokens_percentage parameter
    auto visual_shape = full_encoded_image.resized_source.get_shape();
    if (visual_shape.size() != 3) {
        throw std::invalid_argument("Invalid visual features shape for pruning");
    }
    
    size_t total_visual_tokens = visual_shape[1];
    if (visual_tokens_percentage == 0 || visual_tokens_percentage >= 100) {
        // If invalid percentage, return full features
        return full_encoded_image;
    }
    
    // Calculate actual token count from percentage
    size_t num_visual_tokens = static_cast<size_t>(std::round(total_visual_tokens * visual_tokens_percentage / 100.0));
    
    // If requested tokens equals total tokens, no pruning needed
    if (num_visual_tokens >= total_visual_tokens) {
        return full_encoded_image;
    }
    
    try {
        // Extract text features
        ov::Tensor text_features = extract_text_features(text_prompt);
        
        // Validate text features shape
        auto text_shape = text_features.get_shape();
        if (text_shape.size() != 2) {
            throw std::runtime_error("Invalid text features shape for CDPruner");
        }

        // Update CDPruner configuration with the requested percentage
        auto current_config = get_pruning_config();
        if (current_config.has_value() && current_config->visual_tokens_percentage != visual_tokens_percentage) {
            current_config->visual_tokens_percentage = visual_tokens_percentage;
            set_pruning_config(current_config.value());
        }

        // Apply CDPruner to get pruned visual features
        ov::Tensor pruned_visual_features = apply_pruning(full_encoded_image.resized_source, text_features);

        // Validate pruned features shape
        auto pruned_shape = pruned_visual_features.get_shape();
        if (pruned_shape.size() != 3 || pruned_shape[1] != num_visual_tokens) {
            throw std::runtime_error("CDPruner returned invalid pruned features shape");
        }
        
        // Create new EncodedImage with pruned features
        EncodedImage pruned_encoded_image = full_encoded_image;
        pruned_encoded_image.resized_source = std::move(pruned_visual_features);
        
        // Update the resized_source_size to reflect the pruned token count
        // For LLaVA, we need to maintain the patch grid structure
        size_t original_tokens = full_encoded_image.resized_source_size.height * 
                                full_encoded_image.resized_source_size.width;
        
        if (original_tokens > 0 && num_visual_tokens != original_tokens) {
            // Calculate new grid dimensions that best approximate the pruned token count
            // Try to maintain aspect ratio as much as possible
            float aspect_ratio = static_cast<float>(full_encoded_image.resized_source_size.width) / 
                                static_cast<float>(full_encoded_image.resized_source_size.height);
            
            // Calculate new dimensions
            float new_height_f = std::sqrt(static_cast<float>(num_visual_tokens) / aspect_ratio);
            float new_width_f = new_height_f * aspect_ratio;
            
            size_t new_height = std::max(static_cast<size_t>(1), static_cast<size_t>(std::round(new_height_f)));
            size_t new_width = std::max(static_cast<size_t>(1), static_cast<size_t>(std::round(new_width_f)));
            
            // Adjust to ensure we don't exceed the target token count
            while (new_height * new_width > num_visual_tokens && (new_height > 1 || new_width > 1)) {
                if (new_height > new_width && new_height > 1) {
                    new_height--;
                } else if (new_width > 1) {
                    new_width--;
                } else {
                    break;
                }
            }
            
            // Ensure we have at least some reasonable grid
            new_height = std::max(static_cast<size_t>(1), new_height);
            new_width = std::max(static_cast<size_t>(1), new_width);
            
            pruned_encoded_image.resized_source_size.height = new_height;
            pruned_encoded_image.resized_source_size.width = new_width;
        }
        
        return pruned_encoded_image;
        
    } catch (const std::exception& e) {
        // Log error for debugging purposes
        std::cerr << "CDPruner error in encode_with_pruning: " << e.what() << std::endl;
        
        // On any error, fallback to full features
        return full_encoded_image;
    }
}

InputsEmbedderLLaVA::InputsEmbedderLLaVA(const VLMConfig& vlm_config,
                                         const std::filesystem::path& model_dir,
                                         const std::string& device,
                                         const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, device, device_config) {
    auto current_config = m_vision_encoder->get_pruning_config();
    if (current_config.has_value() && current_config->use_negative_relevance) {
        current_config->use_negative_relevance = true; // Keep negative relevance for LLaVA
        m_vision_encoder->set_pruning_config(current_config.value());
    }
}

InputsEmbedderLLaVA::InputsEmbedderLLaVA(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    auto current_config = m_vision_encoder->get_pruning_config();
    if (current_config.has_value() && current_config->use_negative_relevance) {
        current_config->use_negative_relevance = true;  // Keep negative relevance for LLaVA
        m_vision_encoder->set_pruning_config(current_config.value());
    }
}

std::vector<ov::genai::EncodedImage> InputsEmbedderLLaVA::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
    }
    return embeds;
}

std::vector<ov::genai::EncodedImage> InputsEmbedderLLaVA::encode_images(const std::vector<ov::Tensor>& images, const ov::AnyMap& config_map) {
    std::vector<EncodedImage> embeds;
    // Merge the default vision config with the provided config_map
    ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};
    // Add CDPruner configuration from config_map
    for (const auto& item : config_map) {
        vision_config[item.first] = item.second;
    }
    
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    
    // Check if CDPruner is enabled and text prompt is available
    bool use_pruning = false;
    std::string text_prompt;
    size_t visual_tokens_percentage = 30; // default percentage value
    
    try {
        auto enable_it = vision_config.find("enable_pruning");
        auto prompt_it = vision_config.find("text_prompt");
        auto tokens_it = vision_config.find("visual_tokens_percentage");
        
        if (enable_it != vision_config.end()) {
            use_pruning = enable_it->second.as<bool>();
        }
        if (prompt_it != vision_config.end()) {
            text_prompt = prompt_it->second.as<std::string>();
        }
        if (tokens_it != vision_config.end()) {
            // Extract percentage and validate range
            size_t percentage = tokens_it->second.as<size_t>();
            if (percentage >= 1 && percentage <= 100) {
                visual_tokens_percentage = percentage;
            }
        }
        
        // Only use pruning if explicitly enabled and text prompt is provided
        use_pruning = use_pruning && !text_prompt.empty();
        
    } catch (const std::exception& e) {
        // If there's any error in configuration parsing, disable pruning
        use_pruning = false;
    }
    
    // Process each image
    for (const ov::Tensor& image : single_images) {
        if (use_pruning) {
            // Try to use CDPruner if conditions are met
            try {
                // Cast to VisionEncoderLLaVA to access encode_with_pruning
                auto llava_encoder = dynamic_cast<VisionEncoderLLaVA*>(m_vision_encoder.get());
                if (llava_encoder) {
                    EncodedImage pruned_image = llava_encoder->encode_with_pruning(
                        image, text_prompt, visual_tokens_percentage, vision_config
                    );
                    embeds.emplace_back(std::move(pruned_image));
                } else {
                    // Fallback to standard encoding if cast fails
                    embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
                }
            } catch (const std::exception& e) {
                // If pruning fails, fallback to standard encoding
                std::cerr << "CDPruner failed, using standard encoding: " << e.what() << std::endl;
                embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
            }
        } else {
            // Standard encoding without pruning
            embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
        }
    }
    return embeds;
}

std::pair<std::string, std::vector<size_t>> InputsEmbedderLLaVA::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    std::string image_token = m_vlm_config.im_start;
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());

    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id - base_id).resized_source);
        std::string expanded_tag;
        for (size_t idx = 0; idx < image_embeds.back().get_shape().at(1); ++idx) {
            expanded_tag += image_token;
        }
        expanded_tag += '\n';
        OPENVINO_ASSERT(searched_pos < unified_prompt.length());
        searched_pos = unified_prompt.find(image_token, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos);
        unified_prompt.replace(searched_pos, image_token.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }
    return {std::move(unified_prompt), std::move(images_sequence)};
}

std::pair<std::string, std::vector<size_t>> InputsEmbedderLLaVA::normalize_prompt_with_pruning_support(
    const std::string& prompt,
    size_t base_id,
    const std::vector<EncodedImage>& images) const {
    
    std::string image_token = m_vlm_config.im_start;
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());

    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    size_t searched_pos = 0;
    
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& encoded_image = images.at(new_image_id - base_id);
        image_embeds.push_back(encoded_image.resized_source);
        
        // Get the actual number of visual tokens from the tensor
        // This handles both pruned and non-pruned features correctly
        size_t actual_token_count = get_actual_visual_token_count(encoded_image.resized_source);
        
        std::string expanded_tag;
        for (size_t idx = 0; idx < actual_token_count; ++idx) {
            expanded_tag += image_token;
        }
        expanded_tag += '\n';
        
        OPENVINO_ASSERT(searched_pos < unified_prompt.length());
        searched_pos = unified_prompt.find(image_token, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos);
        unified_prompt.replace(searched_pos, image_token.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }
    return {std::move(unified_prompt), std::move(images_sequence)};
}

bool InputsEmbedderLLaVA::is_image_features_pruned(
    const ov::Tensor& image_features,
    const ImageSize& resized_source_size) const {
    
    if (image_features.get_shape().size() != 3) {
        return false; // Invalid shape, assume not pruned
    }
    
    size_t actual_tokens = image_features.get_shape()[1];
    size_t expected_tokens = resized_source_size.height * resized_source_size.width;
    
    // If actual tokens is significantly less than expected, it's likely pruned
    // We use a threshold to account for possible rounding differences
    return (actual_tokens < expected_tokens * 0.95f);
}

size_t InputsEmbedderLLaVA::get_actual_visual_token_count(const ov::Tensor& image_features) const {
    if (image_features.get_shape().size() != 3) {
        return 0; // Invalid shape
    }
    
    // For LLaVA, the visual token count is the second dimension of the tensor
    // Shape is typically [batch_size, num_tokens, hidden_size]
    return image_features.get_shape()[1];
}

ov::Tensor InputsEmbedderLLaVA::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    
    // Check if any of the images have pruned features
    bool has_pruned_features = false;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& encoded_image = images.at(new_image_id);
        image_embeds.push_back(encoded_image.resized_source);
        
        // Check if this image has pruned features
        if (is_image_features_pruned(encoded_image.resized_source, encoded_image.resized_source_size)) {
            has_pruned_features = true;
        }
    }

    ov::Tensor input_ids;
    
    if (has_pruned_features) {
        // If we have pruned features, we need to re-normalize the prompt
        // to match the actual token counts
        auto [corrected_prompt, corrected_sequence] = normalize_prompt_with_pruning_support(
            unified_prompt, 0, images);
        
        // Get input_ids for the corrected prompt
        input_ids = get_encoded_input_ids(corrected_prompt, metrics);
    } else {
        // Use the original prompt as-is for non-pruned features
        input_ids = get_encoded_input_ids(unified_prompt, metrics);
    }
    
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }
    
    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor encoded_image_token = m_tokenizer.encode(m_vlm_config.im_start, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];
    return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
}

bool VisionEncoderLLaVA::is_pruning_available() {
    return (m_cdpruner != nullptr && m_tokenizer.has_value() && m_text_embedding_model != nullptr);
}

} // namespace ov::genai
