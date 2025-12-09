
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/videochat_flash/classes.hpp"

#include "openvino/opsets/opset13.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

const std::regex NATIVE_PATTERN{R"(<\|image_(\d+)\|>)"};

void write_native(std::ostream& os, size_t idx) {
    os << "<|image_" << idx + 1 << "|>\n";
}
} // namespace

namespace videochat_flash_utils {

std::string normalize_prompt(
    const std::string& prompt, size_t base_id, size_t n_images, const std::regex& native_pattern, void(*write_native)(std::ostream& os, size_t idx)
) {
    std::smatch match;
    std::regex_search(prompt, match, native_pattern);
    auto [image_prompt, image_sequence] = universal_to_native(prompt, write_native);
    if (!image_sequence.empty()) {
        OPENVINO_ASSERT(match.empty(), "Prompt can contain only one type of image tags.");
        verify_ids(image_sequence, base_id, n_images);
        return image_prompt;
    }
    // Restore ids from native tags
    if (!match.empty()) {
        size_t image_id = std::stoul(match.str(1));
        OPENVINO_ASSERT(image_id != 0, "Image tags must be greater than 0");
        image_sequence.push_back(image_id - 1);
        constexpr int submatch_id_to_return = 1;
        for (std::sregex_token_iterator iter{
            match.suffix().first,
            prompt.end(),
            native_pattern,
            submatch_id_to_return
        }; iter != std::sregex_token_iterator{}; ++iter) {
            size_t image_id = std::stoul(*iter);
            OPENVINO_ASSERT(image_id != 0, "Image tags must be greater than 0");
            image_sequence.push_back(image_id - 1);
        }
        if (!image_sequence.empty()) {
            verify_ids(image_sequence, base_id, n_images);
            return image_prompt;
        }
    }
    // Prepend native tags
    std::stringstream stream;
    for (size_t relative_id = 0; relative_id < n_images; relative_id++) {
        image_sequence.push_back(base_id + relative_id);
        write_native(stream, image_sequence.back());
    }
    stream << prompt;
    return stream.str();
}

/// @brief ov::Tensor is tokenized text, size_t is image tag
std::vector<std::variant<ov::Tensor, size_t>> split_tokenize(const std::string& text, ov::genai::Tokenizer& tokenizer, const std::regex& native_pattern) {
    std::vector<std::variant<ov::Tensor, size_t>> tokenized;
    auto prefix_begin = text.begin();
    bool is_submatch = false;
    for (std::sregex_token_iterator iter{
        prefix_begin,
        text.end(),
        native_pattern,
        {0, 1}  // Every match emits two values: whole match and submatch
    }; iter != std::sregex_token_iterator{}; ++iter) {
        if (is_submatch) {
            size_t idx = std::stoul(iter->str());
            OPENVINO_ASSERT(idx != 0);
            tokenized.push_back(idx - 1);
        } else {
            std::string regular_text{prefix_begin, iter->first};
            if (!regular_text.empty()) {
                tokenized.push_back(tokenizer.encode(regular_text, {ov::genai::add_special_tokens(true)}).input_ids);
            }
            prefix_begin = iter->second;
        }
        is_submatch = !is_submatch;
    }
    std::string regular_text{prefix_begin, text.end()};
    if (!regular_text.empty()) {
        tokenized.push_back(tokenizer.encode(regular_text, {ov::genai::add_special_tokens(true)}).input_ids);
    }
    return tokenized;
}

ov::Tensor insert_image_placeholders(
    const std::vector<std::variant<ov::Tensor, size_t>>& chunks,
    const std::vector<size_t>& tokens_per_images
) {
    size_t merged_length = 0;
    for (const std::variant<ov::Tensor, size_t>& chunk : chunks) {
        merged_length += std::visit(utils::overloaded{
            [](const ov::Tensor& chunk) {
                return chunk.get_shape().at(1);
            },
            [&](size_t image_id) {
                return tokens_per_images.at(image_id);
            }
        }, chunk);
    }
    ov::Tensor merged{ov::element::i64, {1, merged_length}};
    size_t offset = 0;
    for (const std::variant<ov::Tensor, size_t>& chunk : chunks) {
        offset += std::visit(utils::overloaded{
            [&](const ov::Tensor& chunk) {
                size_t length = chunk.get_shape().at(1);
                std::copy_n(
                    chunk.data<int64_t>(),
                    length,
                    merged.data<int64_t>() + offset
                );
                return length;
            },
            [&](size_t image_id) {
                int64_t fill_value = -(static_cast<int64_t>(image_id)) - 1;
                std::fill_n(
                    merged.data<int64_t>() + offset,
                    tokens_per_images.at(image_id),
                    fill_value  // -1 to distinguish 0 token and 0 image id.
                );
                return tokens_per_images.at(image_id);
            }
        }, chunk);
    }
    return merged;
}

std::vector<std::variant<ov::Tensor, size_t>> drop_image_placeholders(const ov::Tensor& tokens) {
    std::vector<std::variant<ov::Tensor, size_t>> chunks;
    int64_t last_token = tokens.data<int64_t>()[0];
    size_t text_start = 0;
    for (size_t offset = 1; offset < tokens.get_shape().at(1); ++offset) {
        // If last_token and next_token are not negative, it's continuation of the current chunk text - skip
        // If last_token is negative and next_token is not negative, it's a start of text - save the offset, add image placeholder
        // If last token is not negative and next_token is negative, it's an end of text - push_back a chunk
        // If last_token and next_token are negative, it's continuation of an image placeholder - skip
        // if last_token and next_token are negative but different, it's a start of a new image placeholder - save the previous image placeholder
        int64_t next_token = tokens.data<int64_t>()[offset];
        if (last_token < 0 && next_token >= 0) {
            text_start = offset;
            chunks.push_back(size_t(-(last_token + 1)));
        } else if (last_token >= 0 && next_token < 0) {
            chunks.emplace_back(
                std::in_place_type<ov::Tensor>,
                ov::element::i64,
                ov::Shape{1, offset - text_start},
                tokens.data<int64_t>() + text_start
            );
        } else if (last_token < 0 && next_token < 0 && last_token != next_token) {
            chunks.push_back(size_t(-(last_token + 1)));
        }
        last_token = next_token;
    }
    // Add the last chunk
    size_t full_length = tokens.get_shape().at(1);
    if (last_token >= 0) {
        chunks.emplace_back(
            std::in_place_type<ov::Tensor>,
            ov::element::i64,
            ov::Shape{1, full_length - text_start},
            tokens.data<int64_t>() + text_start
        );
    } else {
        chunks.push_back(size_t(-(last_token + 1)));
    }
    return chunks;
}

ov::Tensor transpose_video_features(const ov::Tensor& src_tensor, const size_t mm_local_num_frames) {
    // Input feature:  [N, C, H, W]
    // Output feature with reshape & transpose: [N//mm_local_num_frames, C, mm_local_num_frames, H, W]
    const ov::Shape S0 = src_tensor.get_shape();
    if (S0.size() != 4 || S0[0] % 4 != 0) {
        throw std::runtime_error("Input tensor must be 4D (NCHW) and Batch size N must be divisible by 4.");
    }

    const size_t N = S0[0];
    const size_t C = S0[1];
    const size_t H = S0[2];
    const size_t W = S0[3];

    const ov::Shape S2 = {N / mm_local_num_frames, C, mm_local_num_frames, H, W};
    const size_t N_prime = N / mm_local_num_frames;

    ov::Tensor dst_tensor(src_tensor.get_element_type(), S2);

    if (src_tensor.get_element_type() != ov::element::f32) {
        throw std::runtime_error("Only f32 element type is supported in this manual implementation.");
    }

    const float* src_data = src_tensor.data<const float>();
    float* dst_data = dst_tensor.data<float>();

    const size_t MCHW = mm_local_num_frames * C * H * W;
    const size_t CHW = C * H * W;
    const size_t HW = H * W;
    const size_t MHW = mm_local_num_frames * H * W;

    for (size_t n_prime = 0; n_prime < N_prime; ++n_prime) { // N/4
        for (size_t c_prime = 0; c_prime < C; ++c_prime) {   // C
            for (size_t d_prime = 0; d_prime < mm_local_num_frames; ++d_prime) { // 4
                for (size_t h_prime = 0; h_prime < H; ++h_prime) { // H
                    for (size_t w_prime = 0; w_prime < W; ++w_prime) { // W
                        // dst shape [N/4, C, 4, H, W])
                        size_t dst_idx = n_prime * MCHW + 
                                         c_prime * MHW + 
                                         d_prime * HW + 
                                         h_prime * W + 
                                         w_prime;
                        // src shape [N, C, H, W]
                        size_t src_idx = (n_prime * mm_local_num_frames + d_prime) * CHW + 
                                         c_prime * HW + 
                                         h_prime * W + 
                                         w_prime;
                        dst_data[dst_idx] = src_data[src_idx];
                    }
                }
            }
        }
    }

    return dst_tensor;
}

// TODO: temporary function and will need to be updated later.
ov::Tensor load_pos_emb_bin_to_ov_tensor(const std::string& bin_file_path, size_t num_patches, size_t emb_dim) {
    // Load internvideo_pos_embed.bin to ov Tensor
    const ov::Shape EXPECTED_SHAPE = {1, num_patches+1, emb_dim};
    const ov::element::Type EXPECTED_TYPE = ov::element::f32;

    size_t total_elements = std::accumulate(
        EXPECTED_SHAPE.begin(), 
        EXPECTED_SHAPE.end(), 
        (size_t)1, 
        std::multiplies<size_t>()
    );
    size_t element_size = EXPECTED_TYPE.size(); // f32 占用 4 字节
    size_t total_bytes = total_elements * element_size;

    std::ifstream file(bin_file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        OPENVINO_THROW("Error opening file: ", bin_file_path);
    }

    size_t file_size = file.tellg();
    if (file_size != total_bytes) {
        OPENVINO_THROW("File size mismatch. Expected ", total_bytes, " bytes, but found ", file_size, " bytes.");
    }

    file.seekg(0, std::ios::beg);

    ov::Tensor pos_emd_tensor(EXPECTED_TYPE, EXPECTED_SHAPE);

    void* tensor_data_ptr = pos_emd_tensor.data(); 

    file.read(static_cast<char*>(tensor_data_ptr), total_bytes);

    if (file.fail()) {
        OPENVINO_THROW("Error reading data from file: ", bin_file_path);
    }

    std::cout << "[DEBUG] Successfully loaded " << total_bytes << " bytes into ov::Tensor." << std::endl;
    return pos_emd_tensor;
}

ov::Tensor remove_second_dim_first_element(const ov::Tensor& input) {
    const ov::Shape& input_shape = input.get_shape();
    if (input_shape.size() < 2) {
        throw std::invalid_argument("Input tensor must have at least 2 dimensions");
    }
    if (input_shape[1] < 1) {
        throw std::invalid_argument("Second dimension of input tensor must be at least 1");
    }
    auto input_data = input.data<float>();
    ov::Shape output_shape = input_shape;
    const size_t org_seq_len = output_shape[1];
    output_shape[1] -= 1;
    const size_t seq_len = output_shape[1];
    const size_t head_elements = input_shape[2];
    std::cout << "DEBUG with input shape" << input_shape << " output shape " << output_shape << std::endl;
    ov::Tensor output(input.get_element_type(), output_shape);
    auto output_data = output.data<float>();

    for(int i=0; i < input_shape[0]; i++) {
        std::copy(
            input_data + i * org_seq_len * head_elements + head_elements,
            input_data + (i + 1) * org_seq_len * head_elements,
            output_data + i * seq_len * head_elements
        );
    }
    return output;
}

ov::Tensor merge_tokens(const ov::Tensor& input, ov::InferRequest& merge_embeddings, const size_t target_num_token = 64) {

    const ov::Shape& x_shape = input.get_shape();
    if (x_shape.size() != 3) {
        throw std::invalid_argument("x must be 3D tensor [batch, tokens, channels], got "
            + std::to_string(x_shape.size()) + "D");
    }
    size_t b = x_shape[0];
    size_t p = x_shape[1];
    size_t c = x_shape[2];

    if (p <= target_num_token) {
        throw std::invalid_argument("Current tokens (" + std::to_string(p) +
            ") must be greater than target (" + std::to_string(target_num_token) + ")");
    }

    std::vector<size_t> r_merge_list;
    size_t tmp_p = p;
    while (tmp_p > target_num_token) {
        size_t next_p = std::max(target_num_token, tmp_p / 2);
        r_merge_list.push_back(tmp_p - next_p);
        tmp_p = next_p;
    }

    const ov::Shape size_shape = {b, p, 1};
    ov::Tensor size_tensor(input.get_element_type(), size_shape);
    float* size_data = size_tensor.data<float>();
    size_t num_elements = size_tensor.get_size();
    std::fill(size_data, size_data + num_elements, 1.0f);

    ov::Tensor current_x = input;

    // 多轮合并
    for (int64_t r : r_merge_list) {
        int64_t current_p = current_x.get_shape()[1];
        merge_embeddings.set_tensor("hidden_states", current_x);
        merge_embeddings.set_tensor("size", size_tensor);
        merge_embeddings.infer();
        current_x = merge_embeddings.get_output_tensor(0);
        size_tensor = merge_embeddings.get_output_tensor(1);
    }

    // 验证输出形状
    const ov::Shape& final_shape = current_x.get_shape();
    ov::Shape expected_shape = { b, target_num_token, c };
    if (final_shape != expected_shape) {
        throw std::runtime_error("Merge failed: expected shape " + expected_shape.to_string() +
            ", got " + final_shape.to_string());
    }

    return current_x;
}

ov::Tensor efficient_flatten(ov::Tensor& original_tensor) {
    // flatten 3D tensor [N,C,W] to 3D tensor [1, N*C, W]
    const ov::Shape& original_shape = original_tensor.get_shape();
    const ov::element::Type& dtype = original_tensor.get_element_type();

    // 2. 计算新形状
    ov::Shape new_shape = {
        1,
        original_shape[0] * original_shape[1], // N*C
        original_shape[2]                      // W
    };

    // 3. 创建一个新的 Tensor 并分配内存 (目标 Tensor)
    ov::Tensor new_tensor(dtype, new_shape);

    // 4. 复制数据 (深拷贝)
    
    // 确保数据大小一致
    if (original_tensor.get_size() != new_tensor.get_size()) {
         OPENVINO_THROW("Flatten error: Element count mismatch during reshape.");
    }
    
    // 获取源和目标 Tensor 的原始数据指针
    const void* src_data = original_tensor.data();
    void* dst_data = new_tensor.data();
    
    // 使用 std::memcpy 进行高效的内存复制
    std::memcpy(dst_data, src_data, original_tensor.get_byte_size());

    return new_tensor;
}

// Temp for debug
void write_ov_tensor_to_bin(const ov::Tensor& tensor, const std::string& bin_file_path) {
    if (!tensor) {
        throw std::runtime_error("Input tensor is invalid or empty.");
    }
    
    // 1. 计算总字节大小
    size_t total_bytes = tensor.get_byte_size();

    // 2. 打开文件流，使用二进制模式和截断模式（如果文件存在则清空）
    std::ofstream file(bin_file_path, std::ios::out | std::ios::binary | std::ios::trunc);
    
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file for writing: " + bin_file_path);
    }
    
    // 3. 获取 Tensor 数据的原始指针
    // Tensor::data() 返回 void* 指针，需要将其转换为 char* 才能用于文件写入。
    const char* tensor_data_ptr = static_cast<const char*>(tensor.data());

    // 4. 将整个数据缓冲区写入文件
    file.write(tensor_data_ptr, total_bytes);

    if (file.fail()) {
        throw std::runtime_error("Error writing data to file: " + bin_file_path);
    }
    
    // 5. 关闭文件
    file.close();
    
    std::cout << "✅ Successfully wrote " << total_bytes << " bytes to " << bin_file_path << std::endl;
    std::cout << "   Shape: " << tensor.get_shape() << std::endl;
    std::cout << "   Type: " << tensor.get_element_type().get_type_name() << std::endl;
}

}  // namespace videochat_flash_utils


VisionEncoderVideoChat_Flash::VisionEncoderVideoChat_Flash(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap properties) : VisionEncoder(model_dir, device, properties) {
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
    auto compiled_model = utils::singleton_core().compile_model(model_dir / "openvino_vision_projection_model.xml", device, {});
    m_ireq_queue_vision_projection = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    
    // TODO: temp load
    m_pos_emb_path = model_dir / "internvideo_pos_embed.bin";
    // TODO: load merge token IR first here
    compiled_model = utils::singleton_core().compile_model(model_dir / "bipartite_soft_matching_merge_ov_model.xml", "CPU", {});
    m_ireq_queue_merge_model = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

VisionEncoderVideoChat_Flash::VisionEncoderVideoChat_Flash(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap properties) : VisionEncoder(models_map, config_dir_path, device, properties) {
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
    const auto& vision_encoder_model = utils::get_model_weights_pair(models_map, "vision_projection").first;
    const auto& vision_encoder_weights = utils::get_model_weights_pair(models_map, "vision_projection").second;
    auto compiled_model = utils::singleton_core().compile_model(vision_encoder_model, vision_encoder_weights, device, properties);
    m_ireq_queue_vision_projection = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    // TODO: load merge token IR first here
}

EncodedImage VisionEncoderVideoChat_Flash::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    EncodedImage encoded_feature;
    size_t frame_num = image.get_shape().at(0);
    size_t mm_local_num_frames = m_vlm_config.mm_local_num_frames;
    size_t mm_hidden_size = m_vlm_config.mm_hidden_size;
    // TODO: how to calculate this value for num_patches
    size_t num_patches = 1024;
    auto input_shape = image.get_shape();
    OPENVINO_ASSERT(input_shape.size() == 4, "Input video features must be 4D.");

    // TODO: here suppose passed in frames have been preprocessed, and shape is [N,3,224,224]
    // Consider if we add preprocess in encode_frames in next step

    // transpose video features
    auto transpose_features = videochat_flash_utils::transpose_video_features(image, mm_local_num_frames);
    // TODO: split each frame group and use static shape vit for better performance

    // TODO: obtain rotary_pos_emb from cpp code
    // now obtain rotary_pos_emd from bin file
    ov::Tensor rotary_pos_emb = videochat_flash_utils::load_pos_emb_bin_to_ov_tensor(m_pos_emb_path.string(), num_patches, mm_hidden_size);

    // video embedding
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& vision_embeddings = infer_request_guard.get();
    vision_embeddings.set_tensor("hidden_states", transpose_features);
    vision_embeddings.set_tensor("rotary_pos_emb", rotary_pos_emb);
    vision_embeddings.infer();
    ov::Tensor processed_vision_embeds = vision_embeddings.get_output_tensor();

    ov::Tensor clipped_vision_embeds = videochat_flash_utils::remove_second_dim_first_element(processed_vision_embeds);

    // merge tokens
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard_merge(this->m_ireq_queue_merge_model.get());
    ov::InferRequest& merge_embeddings = infer_request_guard_merge.get();
    ov::Tensor merged_vision_features = videochat_flash_utils::merge_tokens(clipped_vision_embeds, merge_embeddings);

    // vision projection
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard_proj(this->m_ireq_queue_vision_projection.get());
    ov::InferRequest& vision_projection = infer_request_guard_proj.get();
    vision_projection.set_tensor("hidden_states", merged_vision_features);
    vision_projection.infer();
    // here proj features shape is [N_frames // 4, 4 * 16, 3584]
    ov::Tensor proj_features = vision_projection.get_output_tensor();

    // flatten vision features
    auto final_features = videochat_flash_utils::efficient_flatten(proj_features);
    encoded_feature.images_features_projection = final_features;
    return encoded_feature;
}


std::vector<ov::genai::EncodedImage> InputsEmbedderVideoChat_Flash::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    for (const ov::Tensor& single_video : images) {
        auto encoded_video = m_vision_encoder->encode(single_video);
        embeds.emplace_back(encoded_video);
    }
    return embeds;
}

InputsEmbedderVideoChat_Flash::InputsEmbedderVideoChat_Flash(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config
) : IInputsEmbedder(vlm_config, model_dir, device, device_config) {}


InputsEmbedderVideoChat_Flash::InputsEmbedderVideoChat_Flash(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}


std::pair<std::string, std::vector<size_t>> InputsEmbedderVideoChat_Flash::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    return {videochat_flash_utils::normalize_prompt(prompt, base_id, images.size(), NATIVE_PATTERN, write_native), {}};
}

ov::Tensor InputsEmbedderVideoChat_Flash::get_inputs_embeds(const std::string& image_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& image_sequence) {
    size_t base_id = m_tokens_per_images.size();
    std::vector<ov::Tensor> images_features_proj;
    for (const ov::genai::EncodedImage& encoded_image : images) {
        images_features_proj.push_back(encoded_image.images_features_projection);
        m_tokens_per_images.push_back(images_features_proj.back().get_shape().at(1));
    }
    std::vector<std::variant<ov::Tensor, size_t>> new_chat_tokens;
    if (m_is_chat_conversation) {
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        new_chat_tokens = videochat_flash_utils::split_tokenize(image_prompt, m_tokenizer, NATIVE_PATTERN);
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        std::cout << "[DEBUG] new_chat_tokens size of m_is_chat_conversation is " << new_chat_tokens.size() << std::endl;
    } else {
        std::string templated_prompt;
        if (m_apply_chat_template) {
            ChatHistory history({{{"role", "user"}, {"content", std::move(image_prompt)}}});
            constexpr bool add_generation_prompt = true;
            templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
        } else {
            templated_prompt = std::move(image_prompt);
        }
        std::cout << "m_apply_chat_template is " << m_apply_chat_template << std::endl;
        std::cout << templated_prompt << std::endl;
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        new_chat_tokens = videochat_flash_utils::split_tokenize(templated_prompt, m_tokenizer, NATIVE_PATTERN);
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        std::cout << "[DEBUG] new_chat_tokens size of non m_is_chat_conversation is " << new_chat_tokens.size() << std::endl;
    }
    ov::Tensor new_merged_tokens = videochat_flash_utils::insert_image_placeholders(new_chat_tokens, m_tokens_per_images);
    std::cout << "[DEBUG] new_merged_tokens shape is " << new_merged_tokens.get_shape() << std::endl;
    ov::Tensor new_tokens = update_history(new_merged_tokens);
    std::cout << "[DEBUG] new_tokens shape is " << new_tokens.get_shape() << std::endl;
    m_prev_hist_length = m_kv_cache_state.get_state().size();
    m_kv_cache_state.add_inputs(new_tokens);

    std::vector<std::variant<ov::Tensor, size_t>> tokens = videochat_flash_utils::drop_image_placeholders(new_tokens);
    ov::Tensor inputs_embeds{ov::element::f32, {1, new_tokens.get_shape().at(1), m_vlm_config.hidden_size}};
    size_t offset = 0;
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    for (const std::variant<ov::Tensor, size_t>& chunk : tokens) {
        offset += std::visit(utils::overloaded{
            [&](const ov::Tensor& chunk) {
                const ov::Tensor& text_embeds = m_embedding->infer(req, chunk);
                size_t text_length = text_embeds.get_shape().at(1);
                std::copy_n(
                    text_embeds.data<float>(),
                    text_embeds.get_size(),
                    inputs_embeds.data<float>() + offset * m_vlm_config.hidden_size
                );
                return text_length;
            },
            [&](size_t image_id) {
                const ov::Tensor& image_embeds = images_features_proj.at(image_id - base_id);
                size_t im_length = image_embeds.get_shape().at(1);
                std::copy_n(
                    image_embeds.data<float>(),
                    image_embeds.get_size(),
                    inputs_embeds.data<float>() + offset * m_vlm_config.hidden_size
                );
                return im_length;
            }
        }, chunk);
    }

    if (!m_is_chat_conversation) {
        m_tokens_per_images.clear();
    }
    return inputs_embeds;
}

void InputsEmbedderVideoChat_Flash::update_chat_history(const std::string& decoded_results, const ov::genai::GenerationStatus generation_finish_status) {
    IInputsEmbedder::update_chat_history(decoded_results, generation_finish_status);
    if (generation_finish_status == ov::genai::GenerationStatus::CANCEL)
        m_tokens_per_images = m_prev_tokens_per_images;
    else
        m_prev_tokens_per_images = m_tokens_per_images;
}

void InputsEmbedderVideoChat_Flash::start_chat(const std::string& system_message) {
    IInputsEmbedder::start_chat(system_message);
    m_tokens_per_images.clear();
}

void InputsEmbedderVideoChat_Flash::finish_chat() {
    IInputsEmbedder::finish_chat();
    m_tokens_per_images.clear();
}

} // namespace ov::genai
