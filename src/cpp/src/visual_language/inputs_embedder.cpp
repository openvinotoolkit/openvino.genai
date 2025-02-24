// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "visual_language/inputs_embedder.hpp"

#include "visual_language/clip.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/embedding_model.hpp"
#include "openvino/opsets/opset13.hpp"

#include "utils.hpp"
#include <regex>

namespace ov::genai {

const ModelsMap::mapped_type& get_model_weights_pair(const ModelsMap& models_map, const std::string& key);

class InputsEmbedder::IInputsEmbedder {
protected:
    // VLM config
    VLMConfig m_vlm_config;
    // An encoder to infer embeddings of an image.
    VisionEncoder m_vision_encoder;
    // A model to compute token embeddings.
    // Input shape: [N, conversation length].
    // Output shape: [1, conversation length, hidden_size].
    EmbeddingsModel m_embedding;
    // A tokenizer encoding a prompt.
    Tokenizer m_tokenizer;
    // True if chat mode is activated to save conversation
    // history between generate() calls.
    bool m_is_chat_conversation = false;
    // Chat history
    ChatHistory m_history;
    // If sequence contains some symbols, which could be ambiguous encoded by tokenizer, we need to trim kv cache
    // If we use beam search sampling with chat mode we need to remove last answer of the model from kv cache and add best answer to history 
    // so, let's keep info about amount of tokens to trim from kv cache and amount of tokens to keep in history
    ov::genai::KVCacheTrimManager m_kv_history_trim_manager = {0, 2};
    // True if chat template should be applied for non-chat scenario
    bool m_apply_chat_template = true;
    // Finish reason of last generation for chat scenario
    ov::genai::GenerationStatus m_chat_generation_finish_status = ov::genai::GenerationStatus::RUNNING;
    // reflection of tokens contained in the kv cache
    KVCacheState m_kv_cache_state;

    std::set<int64_t> m_stop_token_ids;
public:
    virtual ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::Tensor>& images, ov::genai::VLMPerfMetrics& metrics) = 0;

    virtual std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size) {
        ov::Tensor position_ids = ov::Tensor{ov::element::i64, { 1, inputs_embeds_size }};
        std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), history_size);
        return {position_ids, std::nullopt};
    }

    EmbeddingsModel get_embedding_model() const {
        return m_embedding;
    }

    Tokenizer get_tokenizer() const {
        return m_tokenizer;
    }

    KVCacheState& get_kv_cache_state() {
        return m_kv_cache_state;
    }

    size_t get_num_tokens_to_remove_from_hist() const {
        return m_kv_history_trim_manager.num_tokens_to_trim;
    }

    void set_stop_token_ids(const std::set<int64_t>& stop_token_ids) {
        m_stop_token_ids = stop_token_ids;
    }

    void set_apply_chat_template_status(bool apply_chat_template) {
        m_apply_chat_template = apply_chat_template;
    }

    virtual void start_chat(const std::string& system_message) {
        m_is_chat_conversation = true;
        m_kv_history_trim_manager.reset();
        if (!m_kv_cache_state.get_state().empty()) {
            m_history.clear();
            m_kv_cache_state.reset_state();
        }
        if (system_message.empty()) {
            return;
        }
        m_history = {{{"role", "system"}, {"content", system_message}}};
    }

    void update_chat_history(const std::string& decoded_results) {
        // Tail of chat template is missing in KV cache.
        // Find the tail to concatenate it with the next input prompt.
        m_history.push_back({{"role", "assistant"}, {"content", decoded_results}});
        m_kv_history_trim_manager.reset();
    }

    virtual void finish_chat() {
        m_is_chat_conversation = false;
        m_kv_history_trim_manager.reset();

        m_history.clear();
        m_kv_cache_state.reset_state();
    }

protected:
    IInputsEmbedder(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config) :
        m_vlm_config{vlm_config},
        m_vision_encoder(model_dir, m_vlm_config.model_type, device, device_config),
        m_embedding(model_dir, m_vlm_config.scale_emb, device, device_config),
        m_tokenizer{model_dir, device_config} { }
    
    IInputsEmbedder(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config) :
        m_vlm_config{vlm_config},
        m_vision_encoder(
            get_model_weights_pair(models_map, "vision_embeddings").first,
            get_model_weights_pair(models_map, "vision_embeddings").second,
            config_dir_path,
            m_vlm_config.model_type,
            device,
            device_config
        ),
        m_embedding(
            get_model_weights_pair(models_map, "text_embeddings").first,
            get_model_weights_pair(models_map, "text_embeddings").second,
            m_vlm_config.scale_emb,
            device,
            device_config
        ),
        m_tokenizer(tokenizer) { }

    ov::Tensor apply_chat_template_tokenize(const std::string& prompt, ov::genai::VLMPerfMetrics& metrics) {
        if (m_is_chat_conversation) {
            m_history.push_back({{"role", "user"}, {"content", prompt}});
            constexpr bool add_generation_prompt = true;
            std::string new_templated_chat_history;
            new_templated_chat_history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
            auto start_tokenizer_time = std::chrono::steady_clock::now();
            ov::Tensor new_chat_tokens = m_tokenizer.encode(new_templated_chat_history, ov::genai::add_special_tokens(false)).input_ids;
            auto end_tokenizer_time = std::chrono::steady_clock::now();
            metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
            return new_chat_tokens;
        } else {
            ov::Tensor encoded_input_ids;
            auto start_tokenizer_time = std::chrono::steady_clock::now();
            if (m_apply_chat_template) {
                std::string templated_prompt;
                ChatHistory history({{{"role", "user"}, {"content", prompt}}});
                constexpr bool add_generation_prompt = true;

                templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
                encoded_input_ids = m_tokenizer.encode(templated_prompt, ov::genai::add_special_tokens(false)).input_ids;
            } else {
                encoded_input_ids = m_tokenizer.encode(prompt).input_ids;
            }
            auto end_tokenizer_time = std::chrono::steady_clock::now();
            metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
            return encoded_input_ids;
        }
    }

    ov::Tensor update_history(const ov::Tensor& new_chat_tokens) {
        ov::Tensor encoded_inputs;
        if (m_is_chat_conversation) {
            ov::genai::align_kv_cache_and_history(m_kv_history_trim_manager, new_chat_tokens, m_kv_cache_state);
            encoded_inputs = get_chat_encoded_input(new_chat_tokens, m_kv_cache_state).input_ids;
        } else {
            encoded_inputs = new_chat_tokens;
        }

        return encoded_inputs;
    }

    ov::Tensor get_encoded_input_ids(const std::string& prompt, ov::genai::VLMPerfMetrics& metrics) {
        const auto new_chat_tokens = apply_chat_template_tokenize(prompt, metrics);
        auto new_input_ids = update_history(new_chat_tokens);
        m_kv_cache_state.add_inputs(new_input_ids);

        return new_input_ids;
    }

    /**
    * @brief Unpads an image tensor of a padded and resized image.
    * Used for packing image features of llava_next models.
    *
    * @param tensor An image tensor with a shape (embed_dim, height, width)
    * @param original_size A size of original image
    * @return An unpadded image tensor with a shape (embed_dim, new_height, new_width)
    */

    /**
    * @brief Converts a vector of batched images ([NHWC]) into a vector of individual image tensors ([1HWC]).
    *
    * @param images A vector of tensors representing the images. Each tensor can have a shape of either [NHWC] or [HWC].
    * @return A vector of tensors where each tensor represents a single image with a shape of [1, H, W, C].
    */
    std::vector<ov::Tensor> to_single_image_tensors(const std::vector<ov::Tensor>& images) {
        std::vector<ov::Tensor> single_image_tensors;
        for (const auto& image : images) {
            ov::Tensor reshaped_image = image;
            ov::Shape image_shape = image.get_shape();
            switch (image_shape.size()) {
                case 3:
                    reshaped_image.set_shape({1, image_shape.at(0), image_shape.at(1), image_shape.at(2)});
                    break;
                case 4: break;
                default: OPENVINO_THROW("Input image must have [NHWC] or [HWC] layout");
            }
            ov::Shape reshaped_image_shape = reshaped_image.get_shape();
            for (size_t batch_idx = 0; batch_idx < reshaped_image_shape.at(0); ++batch_idx) {
                ov::Tensor single_image{
                    reshaped_image.get_element_type(),
                    {1, reshaped_image_shape.at(1), reshaped_image_shape.at(2), reshaped_image_shape.at(3)},
                    reshaped_image.data<uint8_t>() + batch_idx * reshaped_image_shape.at(1) * reshaped_image_shape.at(2) * reshaped_image_shape.at(3)
                };
                single_image_tensors.push_back(std::move(single_image));
            }
        }
        return single_image_tensors;
    }
};

class InputsEmbedderMiniCPM : public InputsEmbedder::IInputsEmbedder {
    // A resampler model to resample image embeddings.
    // [N, H*W, old_hidden_size] is the input shape.
    // [N, query_num, hidden_size] is the output shape.
    ov::InferRequest m_resampler;
    // Precomputed positional embeddings for the resampler.
    // [70, 70, hidden_size]. 70 is the initial guess of the image
    // height and width after dividing by patch_size.
    ov::Tensor m_pos_embed_cache;
    // Used to insert <image_id>i</image_id> per image (not a slice).
    size_t m_image_id = 0;

public:
    InputsEmbedderMiniCPM(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config) :
        IInputsEmbedder(vlm_config, model_dir, device, device_config) {
        auto compiled_model =
            utils::singleton_core().compile_model(model_dir / "openvino_resampler_model.xml", device, device_config);
        ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM resampler model");
        m_resampler = compiled_model.create_infer_request();

        m_pos_embed_cache = get_2d_sincos_pos_embed(m_vlm_config.hidden_size, {70, 70});
    }

    InputsEmbedderMiniCPM(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config) :
        IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
            m_resampler = utils::singleton_core().compile_model(
                get_model_weights_pair(models_map, "resampler").first,
                get_model_weights_pair(models_map, "resampler").second,
                device,
                device_config
            ).create_infer_request();

            m_pos_embed_cache = get_2d_sincos_pos_embed(m_vlm_config.hidden_size, {70, 70});
        }

    virtual ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::Tensor>& images, ov::genai::VLMPerfMetrics& metrics) override {
        std::string images_prompt;
        std::vector<EncodedImage> embeds;

        std::vector<ov::Tensor> single_images = to_single_image_tensors(images);

        for (const ov::Tensor& image : single_images) {
            EncodedImage encoded_image = m_vision_encoder.encode(image);
            if (m_vlm_config.use_image_id) {
                images_prompt += m_vlm_config.im_id_start + std::to_string(m_image_id) + m_vlm_config.im_id_end;
                ++m_image_id;
            }
            std::string unk64;
            for (size_t idx = 0; idx < m_vlm_config.query_num; ++idx) {
                unk64 += m_vlm_config.unk;
            }
            images_prompt += m_vlm_config.im_start + unk64 + m_vlm_config.im_end;
            if (encoded_image.slices) {
                ov::Shape slices_shape = encoded_image.slices.get_shape();
                for (size_t row_idx = 0; row_idx < slices_shape.at(0); ++row_idx) {
                    for (size_t col_idx = 0; col_idx < slices_shape.at(1); ++col_idx) {
                        images_prompt += m_vlm_config.slice_start + unk64 + m_vlm_config.slice_end;
                    }
                    images_prompt += '\n';
                }
            }
            if ('\n' != *(images_prompt.end() - 1)) {
                // Image wasn't sliced, add \n to the end of image anyway.
                // Strangely, \n isn't placed between </image><slice>.
                images_prompt += '\n';
            }
            embeds.push_back(std::move(encoded_image));
        }
        images_prompt += prompt;

        ov::Tensor encoded_input = get_encoded_input_ids(images_prompt, metrics);

        ov::Tensor inputs_embeds = m_embedding.infer(encoded_input);
        OPENVINO_ASSERT(
            m_vlm_config.hidden_size == inputs_embeds.get_shape().at(2),
            "Unexpected embedding size"
        );
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        ov::Tensor special_tokens = m_tokenizer.encode(
            m_vlm_config.im_start
            + m_vlm_config.im_end
            + m_vlm_config.slice_start
            + m_vlm_config.slice_end
        ).input_ids;
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
        metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        OPENVINO_ASSERT(
            4 == special_tokens.get_shape().at(1),
            "Every special token must be represented with a single int."
        );
        int64_t im_start_id = special_tokens.data<int64_t>()[0];
        int64_t im_end_id = special_tokens.data<int64_t>()[1];
        int64_t slice_start_id = special_tokens.data<int64_t>()[2];
        int64_t slice_end_id = special_tokens.data<int64_t>()[3];
        int64_t im_start_pos = 0, slice_start_pos = 0;
        int64_t* begin = encoded_input.data<int64_t>();
        int64_t* ids = begin;
        size_t encoded_input_size = encoded_input.get_size();
        int64_t* end = ids + encoded_input_size;
        float* inputs_embeds_data = inputs_embeds.data<float>();
        for (const EncodedImage& encoded_image : embeds) {
            const ov::Tensor& resampled_source = resample(encoded_image.resized_source, {encoded_image.resized_source_size});
            float* emb = resampled_source.data<float>();
            ids = std::find(ids, end, im_start_id);
            OPENVINO_ASSERT(end != ids);
            ++ids;
            std::copy_n(emb, resampled_source.get_size(), inputs_embeds_data + std::distance(begin, ids) * m_vlm_config.hidden_size);
            ids += m_vlm_config.query_num;
            if (encoded_image.slices) {
                size_t token_idx = 0;
                const ov::Shape& slices_shape = encoded_image.slices.get_shape();
                for (size_t i = 0; i < slices_shape.at(0); ++i) {
                    for (size_t ja = 0; ja < slices_shape.at(1); ++ja) {
                        size_t d2 = slices_shape.at(2);
                        size_t d3 = slices_shape.at(3);
                        ov::Tensor encoded_view{ov::element::f32, {1, d2, d3}, encoded_image.slices.data<float>() + (i * slices_shape.at(1) + ja) * d2 * d3};
                        const ov::Tensor& vision_embed_tensor_i_j = resample(encoded_view, {encoded_image.slices_size});
                        ids = std::find(ids, end, slice_start_id);
                        OPENVINO_ASSERT(end != ids);
                        ++ids;
                        std::copy_n(vision_embed_tensor_i_j.data<float>(), vision_embed_tensor_i_j.get_size(), inputs_embeds_data + std::distance(begin, ids) * m_vlm_config.hidden_size);
                        ids += m_vlm_config.query_num;
                    }
                }
            }
        }

        if (!m_is_chat_conversation) {
            m_image_id = 0;
        }
        return inputs_embeds;
    }

    virtual void start_chat(const std::string& system_message) override {
        IInputsEmbedder::start_chat(system_message);
        m_image_id = 0;
    }

    virtual void finish_chat() override {
        IInputsEmbedder::finish_chat();
        m_image_id = 0;
    }

private:
    ov::Tensor resample(const ov::Tensor& encoded_image, const std::vector<ImageSize>& target_sizes) {
        size_t bs = encoded_image.get_shape().at(0);
        std::vector<size_t> patch_len{target_sizes.size()};
        std::transform(target_sizes.begin(), target_sizes.end(), patch_len.begin(), [](const ImageSize& height_width) {
            return height_width.height * height_width.width;
        });
        adjust_pos_cache(
            target_sizes,
            m_vlm_config.hidden_size,
            m_pos_embed_cache
        );
        size_t max_patch_len = *std::max_element(patch_len.begin(), patch_len.end());
        ov::Tensor key_padding_mask(ov::element::f32, {bs, max_patch_len});
        float* mask_data = key_padding_mask.data<float>();
        size_t embed_len = m_pos_embed_cache.get_shape().at(2);
        ov::Tensor pos_embed(ov::element::f32, {max_patch_len, bs, embed_len});  // BLD => L * B * D
        float* pos_embed_data = pos_embed.data<float>();
        float* cache_data = m_pos_embed_cache.data<float>();
        size_t _d0 = m_pos_embed_cache.get_shape().at(0);
        size_t _d1 = m_pos_embed_cache.get_shape().at(1);
        for (size_t i = 0; i < bs; ++i) {
            size_t target_h = target_sizes.at(i).height;
            size_t target_w = target_sizes.at(i).width;
            for (size_t h_idx = 0; h_idx < target_h; ++h_idx) {
                for (size_t w_idx = 0; w_idx < target_w; ++w_idx) {
                    std::copy_n(
                        cache_data + (h_idx * _d1 + w_idx) * embed_len,
                        embed_len,
                        pos_embed_data + (h_idx * target_w + w_idx) * bs * embed_len + i * embed_len
                    );
                }
            }
            for (size_t flat = target_h * target_w; flat < max_patch_len; ++flat) {
                std::fill_n(pos_embed_data + flat * bs * embed_len + i * embed_len, embed_len, 0.0f);
            }
            std::fill_n(mask_data + i * max_patch_len, patch_len[i], 0.0f);
            std::fill_n(mask_data + i * max_patch_len + patch_len[i], max_patch_len - patch_len[i], 1.0f);
        }
        m_resampler.set_tensor("image_feature", encoded_image);  // [N, H*W, old_hidden_size]
        m_resampler.set_tensor("pos_embed", pos_embed);  // [H*W, N, new_hidden_size]
        m_resampler.set_tensor("key_padding_mask", key_padding_mask);  // [N, H*W]
        m_resampler.infer();
        return m_resampler.get_output_tensor();  // [N, query_num, new_hidden_size]
    }

    ov::Tensor concatenate_last_dim(const ov::Tensor& first, const ov::Tensor& second) {
        size_t res_d_0 = first.get_shape().at(0);
        size_t res_d_1 = first.get_shape().at(1);
        OPENVINO_ASSERT(second.get_shape().at(0) == res_d_0);
        OPENVINO_ASSERT(second.get_shape().at(1) == res_d_1);
        size_t res_d_2 = first.get_shape().at(2) + second.get_shape().at(2);
        ov::Tensor res{first.get_element_type(), {res_d_0, res_d_1, res_d_2}};
        float* first_data = first.data<float>();
        float* second_data = second.data<float>();
        float* res_data = res.data<float>();
        for (size_t i = 0; i < res_d_0; ++i) {
            for (size_t j = 0; j < res_d_1; ++j) {
                size_t k = 0;
                for (; k < first.get_shape().at(2); ++k) {
                    res_data[i * res_d_1 * res_d_2 + j * res_d_2 + k]
                        = first_data[i * res_d_1 * first.get_shape().at(2) + j * first.get_shape().at(2) + k];
                }
                for (size_t l = 0; l < second.get_shape().at(2); ++l, ++k) {
                    res_data[i * res_d_1 * res_d_2 + j * res_d_2 + k]
                        = second_data[i * res_d_1 * second.get_shape().at(2) + j * second.get_shape().at(2) + l];
                }
            }
        }
        return res;
    }

    /// embed_dim: output dimension for each position
    /// pos: a list of positions to be encoded: size (H, W)
    /// out: (H, W, D)
    ov::Tensor get_1d_sincos_pos_embed_from_grid_new(size_t embed_dim, const ov::Tensor& pos) {
        OPENVINO_ASSERT(embed_dim % 2 == 0);
        ov::Shape pos_shape = pos.get_shape();
        size_t H = pos_shape[0];
        size_t W = pos_shape[1];

        std::vector<float> omega(embed_dim / 2);
        for (size_t i = 0; i < omega.size(); ++i) {
            omega[i] = 1.0f / std::pow(10000.0f, float(i) / (embed_dim / 2));
        }

        std::vector<size_t> out_shape = {H, W, embed_dim};
        ov::Tensor emb(ov::element::f32, out_shape);

        float* pos_data = pos.data<float>();
        float* emb_data = emb.data<float>();

        size_t counter = 0;
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                for (size_t d = 0; d < embed_dim / 2; ++d) {
                    // Correctly access the 2D position grid
                    float value = omega[d] * pos_data[h * W + w];
                    emb_data[h * W * embed_dim + w * embed_dim + d] = std::sin(value);
                    emb_data[h * W * embed_dim + w * embed_dim + d + (embed_dim / 2)] = std::cos(value);
                }
            }
        }
        return emb;
    }

    ov::Tensor get_2d_sincos_pos_embed_from_grid(size_t embed_dim, const ov::Tensor& grid) {
        OPENVINO_ASSERT(embed_dim % 2 == 0);
        ov::Shape grid_shape = grid.get_shape();
        float* grid_data = grid.data<float>();
        ov::Shape plane_shape{grid_shape.at(1), grid_shape.at(2)};
        ov::Tensor emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, ov::Tensor{
            ov::element::f32,
            plane_shape,
            grid_data
        });  // (H, W, D/2)
        ov::Tensor emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, ov::Tensor{
            ov::element::f32,
            plane_shape,
            grid_data + plane_shape.at(0) * plane_shape.at(1)
        });  // (H, W, D/2)
        return concatenate_last_dim(emb_h, emb_w);
    }

    /// image_size: image_size or (image_height, image_width)
    /// return:
    /// pos_embed: [image_height, image_width, embed_dim]
    ov::Tensor get_2d_sincos_pos_embed(size_t embed_dim, const ImageSize& image_size) {
        size_t grid_h_size = image_size.height, grid_w_size = image_size.width;
        ov::Tensor grid(ov::element::f32, {2, grid_h_size, grid_w_size});
        float* data = grid.data<float>();
        for (size_t y = 0; y < grid_h_size; ++y) {
            std::iota(data, data + grid_w_size, 0.0f);
            data += grid_w_size;
        }
        for (float y = 0.0f; y < grid_h_size; ++y) {
            std::fill(data, data + grid_w_size, y);
            data += grid_w_size;
        }
        return get_2d_sincos_pos_embed_from_grid(embed_dim, grid);
    }

    void adjust_pos_cache(
        const std::vector<ImageSize>& target_sizes,
        size_t hidden_size,
        ov::Tensor& pos_embed_cache
    ) {
        size_t max_h = std::max_element(target_sizes.begin(), target_sizes.end(), [](const ImageSize& left, const ImageSize& right) {
            return left.height < right.height;
        })->height;
        size_t max_w = std::max_element(target_sizes.begin(), target_sizes.end(), [](const ImageSize& left, const ImageSize& right) {
            return left.width < right.width;
        })->width;
        size_t allocated_height, allocated_width;
        if (pos_embed_cache) {
            const ov::Shape& allocated_shape = pos_embed_cache.get_shape();
            allocated_height = allocated_shape.at(0);
            allocated_width = allocated_shape.at(1);
        } else {
            allocated_height = allocated_width = 70;
        }
        if (max_h > allocated_height || max_w > allocated_width) {
            allocated_height = std::max(max_h, allocated_height);
            allocated_width = std::max(max_w, allocated_width);
            pos_embed_cache = get_2d_sincos_pos_embed(
                hidden_size, {allocated_height, allocated_width}
            );
        }
    }
};

class InputsEmbedderLLaVA : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderLLaVA(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config) :
        IInputsEmbedder(vlm_config, model_dir, device, device_config) { }

    InputsEmbedderLLaVA(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config) :
        IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

    virtual ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::Tensor>& images, ov::genai::VLMPerfMetrics& metrics) override {
        std::string image_token = m_vlm_config.im_start;
        
        std::vector<ov::Tensor> single_images = to_single_image_tensors(images);

        std::string formatted_prompt;
        std::vector<ov::Tensor> image_embeds;
        image_embeds.reserve(single_images.size());

        for (const auto& image : single_images) {
            ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};
            EncodedImage encoded_image = m_vision_encoder.encode(image, vision_config);
            image_embeds.push_back(std::move(encoded_image.resized_source));
            formatted_prompt += image_token + "\n";
        }
        formatted_prompt += prompt;

        ov::Tensor input_ids = get_encoded_input_ids(formatted_prompt, metrics);
        ov::Tensor text_embeds = m_embedding.infer(input_ids);

        if (images.empty()) {
            return text_embeds;
        }
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        ov::Tensor encoded_image_token = m_tokenizer.encode(m_vlm_config.im_start, ov::genai::add_special_tokens(false)).input_ids;
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
        metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];
        return merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
    }

protected:
    ov::Tensor merge_text_and_image_embeddings_llava(
        const ov::Tensor& input_ids,
        const ov::Tensor& text_embeds,
        const std::vector<ov::Tensor>& image_embeds,
        int64_t image_token_id
    ) {
        auto text_embeds_shape = text_embeds.get_shape();
        size_t text_embeds_seq_length = text_embeds_shape[1];
        size_t hidden_size = text_embeds_shape[2];

        const int64_t* input_ids_data = input_ids.data<const int64_t>();
        const float* text_embeds_data = text_embeds.data<const float>();

        size_t num_image_tokens = 0;
        for (size_t s = 0; s < text_embeds_seq_length; ++s) {
            if (input_ids_data[s] == image_token_id) {
                num_image_tokens++;
            }
        }
        auto num_images = image_embeds.size();
        OPENVINO_ASSERT(
            num_image_tokens == num_images,
            "Number of image tokens in input_ids different from num_images."
        );

        size_t total_image_seq_length = 0;
        for (const auto& single_image_embeds : image_embeds) {
            OPENVINO_ASSERT(
                text_embeds_shape[2] == single_image_embeds.get_shape().at(2),
                "Incompatible shapes between text_embeds and image_embeds"
            );
            total_image_seq_length += single_image_embeds.get_shape().at(1);
        }
        size_t merged_seq_length = text_embeds_seq_length + total_image_seq_length - num_image_tokens;

        constexpr size_t BATCH_SIZE = 1;
        ov::Tensor merged_embeds(text_embeds.get_element_type(), {BATCH_SIZE, merged_seq_length, hidden_size});
        float* merged_data = merged_embeds.data<float>();

        size_t merged_idx = 0;
        size_t image_idx = 0;
        for (size_t s = 0; s < text_embeds_seq_length; ++s) {
            if (input_ids_data[s] == image_token_id) {
                const float* image_embeds_data = image_embeds[image_idx].data<const float>();
                size_t image_seq_length = image_embeds[image_idx].get_shape()[1];

                std::copy_n(image_embeds_data,
                            image_seq_length * hidden_size,
                            merged_data + merged_idx * hidden_size);
                merged_idx += image_seq_length;
                image_idx++;
            } else {
                std::copy_n(text_embeds_data + s * hidden_size,
                            hidden_size,
                            merged_data + merged_idx * hidden_size);
                merged_idx++;
            }
        }
        return merged_embeds;
    }
};

class InputsEmbedderLLaVANext : public InputsEmbedderLLaVA {
public:
    InputsEmbedderLLaVANext(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config) :
        InputsEmbedderLLaVA(vlm_config, model_dir, device, device_config) { }

    InputsEmbedderLLaVANext(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config) :
        InputsEmbedderLLaVA(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

    virtual ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::Tensor>& images, ov::genai::VLMPerfMetrics& metrics) override {
        std::string image_token = m_vlm_config.im_start;

        std::vector<ov::Tensor> single_images = to_single_image_tensors(images);

        std::string formatted_prompt;
        std::vector<ov::Tensor> image_embeds;
        image_embeds.reserve(single_images.size());
        
        ov::Tensor image_newline;

        for (const auto& image : single_images) {
            ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};
            EncodedImage encoded_image = m_vision_encoder.encode(image, vision_config);

            if (!image_newline) {
                size_t embed_dim = encoded_image.resized_source.get_shape().at(2);
                image_newline = ov::Tensor(encoded_image.resized_source.get_element_type(), {embed_dim});
                float* image_newline_data = image_newline.data<float>();
                std::copy(m_vlm_config.image_newline.begin(), m_vlm_config.image_newline.end(), image_newline_data);
            }

            ImageSize original_image_size{image.get_shape().at(1), image.get_shape().at(2)}; // [height, width]

            ov::Tensor packed_features = pack_image_features_llava_next(encoded_image, original_image_size, image_newline);

            image_embeds.push_back(std::move(packed_features));
            formatted_prompt += image_token + "\n";
        }
        formatted_prompt += prompt;

        ov::Tensor input_ids = get_encoded_input_ids(formatted_prompt, metrics);
        ov::Tensor text_embeds = m_embedding.infer(input_ids);

        if (images.empty()) {
            return text_embeds;
        }
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        ov::Tensor encoded_image_token = m_tokenizer.encode(m_vlm_config.im_start, ov::genai::add_special_tokens(false)).input_ids;
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
        metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];
        return merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
    }

private:
    /**
    * @brief Processes base and patches image features extracted from encoded image.
    * Used in getting inputs embeds for llava_next models.
    *
    * @param encoded_image An encoded image retrieved from vision encoder
    * @param original_image_size A size of the original image
    * @param image_newline An image newline tensor with a shape (embed_dim)
    * @return A tensor with a shape (1, new_seq_len, embed_dim)
    */
    ov::Tensor pack_image_features_llava_next(
        const EncodedImage& encoded_image,
        const ImageSize& original_image_size,
        const ov::Tensor& image_newline
    ) {
        auto image_feature = encoded_image.resized_source;
        auto image_feature_shape = image_feature.get_shape();
        size_t num_patches = image_feature_shape[0];
        size_t patch_seq_len = image_feature_shape[1];
        size_t embed_dim = image_feature_shape[2];

        const float* image_feature_data = image_feature.data<float>();
        const float* newline_data = image_newline.data<float>();

        if (num_patches > 1) {
            // Extract base image feature (first patch)
            ov::Tensor base_image_feature(image_feature.get_element_type(), {1, patch_seq_len, embed_dim});
            const float* src_data = image_feature.data<float>();
            float* dst_data = base_image_feature.data<float>();
            std::copy(src_data, src_data + patch_seq_len * embed_dim, dst_data);

            // Extract other grid patches
            ov::Tensor patches_image_feature(image_feature.get_element_type(), {num_patches - 1, patch_seq_len, embed_dim});
            dst_data = patches_image_feature.data<float>();
            std::copy(src_data + patch_seq_len * embed_dim,
                    src_data + num_patches * patch_seq_len * embed_dim,
                    dst_data);

            // Process grid patches image feature
            size_t height = encoded_image.resized_source_size.height;
            size_t width = encoded_image.resized_source_size.width;
            size_t num_patch_height = encoded_image.patches_grid.first;
            size_t num_patch_width = encoded_image.patches_grid.second;

            ov::Tensor reshaped_image_feature = reshape_and_rearrange_image_feature(patches_image_feature, num_patch_height, num_patch_width, height, width);

            ov::Tensor unpadded_image_feature = unpad_image(reshaped_image_feature, original_image_size);

            ov::Tensor image_feature_with_newline = add_image_newline(unpadded_image_feature, image_newline);

            ov::Tensor processed_image_feature = flatten_and_transpose(image_feature_with_newline);

            // Concatenate base image feature ([1, seq_len_1, emded_dim]) and patches image feature ([seq_len_2, embed_dim])
            auto base_shape = base_image_feature.get_shape();
            auto processed_shape = processed_image_feature.get_shape();

            const float* base_data = base_image_feature.data<float>();
            const float* processed_data = processed_image_feature.data<float>();

            ov::Tensor result(image_feature.get_element_type(), {1, base_shape[1] + processed_shape[0], embed_dim});
            // Copy base image feature data
            std::copy(base_data, base_data + base_shape[1] * embed_dim, result.data<float>());
            // Copy processed image feature data
            std::copy(processed_data,
                    processed_data + processed_shape[0] * embed_dim,
                    result.data<float>() + base_shape[1] * embed_dim);
            return result;
        } else {
            // If there is only one patch, return the original (base) image feature concatenated with image_newline
            ov::Tensor result(image_feature.get_element_type(), {1, patch_seq_len + 1, embed_dim});
            // Copy base image feature data
            std::copy(image_feature_data + embed_dim,
                    image_feature_data + patch_seq_len * embed_dim,
                    result.data<float>());
            // Append image_newline data
            std::copy(newline_data,
                    newline_data + embed_dim,
                    result.data<float>() + patch_seq_len * embed_dim);
            return result;
        }
    }

    /**
    * @brief Adds image newline tensor to patches image feature tensor.
    * Used for packing image features of llava_next models.
    *
    * @param image_feature A tensor with a shape (embed_dim, height, width)
    * @param image_newline A tensor with a shape (embed_dim)
    * @return A tensor with a shape (embed_dim, height, width + 1)
    */
    ov::Tensor add_image_newline(const ov::Tensor& image_feature, const ov::Tensor& image_newline) {
        auto shape = image_feature.get_shape();

        OPENVINO_ASSERT(shape.size() == 3, "Input image_feature must have 3 dimensions");

        size_t embed_dim = shape[0];
        size_t height = shape[1];
        size_t width = shape[2];

        OPENVINO_ASSERT(image_newline.get_shape()[0] == embed_dim, "image_newline dimension must match embed_dim of image_feature");

        const float* image_feature_data = image_feature.data<float>();
        const float* newline_data = image_newline.data<float>();

        ov::Tensor feature_with_newline{image_feature.get_element_type(), {embed_dim, height, width + 1}};
        float* feature_with_newline_data = feature_with_newline.data<float>();

        for (size_t e = 0; e < embed_dim; ++e) {
            for (size_t h = 0; h < height; ++h) {
                // Copy original image feature data
                std::copy(
                    image_feature_data + (e * height * width + h * width),
                    image_feature_data + (e * height * width + (h + 1) * width),
                    feature_with_newline_data + (e * height * (width + 1) + h * (width + 1))
                );
                // Add image newline
                feature_with_newline_data[e * height * (width + 1) + h * (width + 1) + width] = newline_data[e];
            }
        }

        return feature_with_newline;
    }

    /**
    * @brief Flattens and transposes tensor.
    * Used for packing image features of llava_next models.
    *
    * @param tensor A tensor with a shape (embed_dim, height, width)
    * @return A tensor with a shape (height * width, embed_dim)
    */
    ov::Tensor flatten_and_transpose(const ov::Tensor& tensor) {
        auto shape = tensor.get_shape();
        OPENVINO_ASSERT(shape.size() == 3, "Flattening tensor must have 3 dimensions");
        const float* data = tensor.data<float>();
        size_t embed_dim = shape[0];
        size_t height = shape[1];
        size_t width = shape[2];
        size_t flatten_dim = height * width;

        ov::Tensor flatten_feature(tensor.get_element_type(), {flatten_dim, embed_dim});
        float* flatten_feature_data = flatten_feature.data<float>();

        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                for (size_t e = 0; e < embed_dim; ++e) {
                    flatten_feature_data[(h * width + w) * embed_dim + e] = data[e * flatten_dim + h * width + w];
                }
            }
        }

        return flatten_feature;
    }


    ov::Tensor reshape_and_rearrange_image_feature(const ov::Tensor& image_feature,
                                                   int num_patch_height,
                                                   int num_patch_width,
                                                   int height,
                                                   int width) {
        auto shape = image_feature.get_shape();
        OPENVINO_ASSERT(shape.size() == 3, "image_feature tensor must have 3 dimensions");

        size_t num_patches = shape[0];
        size_t patch_seq_len = shape[1];
        size_t embed_dim = shape[2];

        OPENVINO_ASSERT(
            num_patches == num_patch_height * num_patch_width,
            "Number of patches does not match the specified grid size"
        );

        OPENVINO_ASSERT(
            patch_seq_len == height * width,
            "Patch sequence length does not match the specified height and width"
        );

        // Reshape tensor data and permute dimensions
        // [num_patches, patch_seq_len, embed_dim] -> [embed_dim, num_patch_height, height, num_patch_width, width]
        std::vector<float> reshaped_data(num_patches * patch_seq_len * embed_dim);
        const float* image_feature_data = image_feature.data<float>();

        for (int p = 0; p < num_patches; ++p) {
            for (int i = 0; i < patch_seq_len; ++i) {
                for (int e = 0; e < embed_dim; ++e) {
                    int h = i / width;
                    int w = i % width;
                    int ph = p / num_patch_width;
                    int pw = p % num_patch_width;
                    reshaped_data[((((e * num_patch_height + ph) * height + h) * num_patch_width + pw) * width + w)] =
                        image_feature_data[(p * patch_seq_len + i) * embed_dim + e];
                }
            }
        }

        ov::Tensor result(image_feature.get_element_type(),
                          {static_cast<size_t>(embed_dim),
                           static_cast<size_t>(num_patch_height * height),
                           static_cast<size_t>(num_patch_width * width)}
        );
        std::copy(reshaped_data.begin(), reshaped_data.end(), result.data<float>());
        return result;
    }

    /**
    * @brief Unpads an image tensor of a padded and resized image.
    * Used for packing image features of llava_next models.
    *
    * @param tensor An image tensor with a shape (embed_dim, height, width)
    * @param original_size A size of original image
    * @return An unpadded image tensor with a shape (embed_dim, new_height, new_width)
    */
    ov::Tensor unpad_image(const ov::Tensor& tensor, const ImageSize& original_size) {
        size_t original_height = original_size.height;
        size_t original_width = original_size.width;
        auto shape = tensor.get_shape();
        size_t embed_dim = shape[0];
        size_t current_height = shape[1];
        size_t current_width = shape[2];

        float original_aspect_ratio = static_cast<float>(original_width) / original_height;
        float current_aspect_ratio = static_cast<float>(current_width) / current_height;

        ov::Tensor unpadded_tensor;

        if (original_aspect_ratio > current_aspect_ratio) {
            float scale_factor = static_cast<float>(current_width) / original_width;
            size_t new_height = static_cast<size_t>(original_height * scale_factor);
            size_t padding = (current_height - new_height) / 2;
            size_t unpadded_height_dim = new_height + 1;
            unpadded_tensor = ov::Tensor(tensor.get_element_type(), {embed_dim, unpadded_height_dim, current_width});

            for (size_t e = 0; e < embed_dim; ++e) {
                for (int h = 0; h < unpadded_height_dim; ++h) {
                    std::copy(
                        tensor.data<float>() + (e * current_height * current_width + (padding + h) * current_width),
                        tensor.data<float>() + (e * current_height * current_width + (padding + h) * current_width + current_width),
                        unpadded_tensor.data<float>() + (e * unpadded_height_dim * current_width + h * current_width)
                    );
                }
            }
        } else {
            float scale_factor = static_cast<float>(current_height) / original_height;
            size_t new_width = static_cast<size_t>(original_width * scale_factor);
            size_t padding = (current_width - new_width) / 2;
            size_t unpadded_width_dim = new_width + 1;
            unpadded_tensor = ov::Tensor(tensor.get_element_type(), {embed_dim, current_height, unpadded_width_dim});

            for (size_t e = 0; e < embed_dim; ++e) {
                for (int h = 0; h < current_height; ++h) {
                    std::copy(
                        tensor.data<float>() + (e * current_height * current_width + h * current_width + padding),
                        tensor.data<float>() + (e * current_height * current_width + h * current_width + padding + unpadded_width_dim),
                        unpadded_tensor.data<float>() + (e * current_height * unpadded_width_dim + h * unpadded_width_dim)
                    );
                }
            }
        }

        return unpadded_tensor;
    }
};

class InputsEmbedderInternVLChat : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderInternVLChat(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config) :
        IInputsEmbedder(vlm_config, model_dir, device, device_config) { }

    InputsEmbedderInternVLChat(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config) :
        IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

    virtual ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::Tensor>& images, ov::genai::VLMPerfMetrics& metrics) override {
        std::string image_start_token = m_vlm_config.image_start_token;
        std::string image_context_token = m_vlm_config.image_context_token;
        std::string image_end_token = m_vlm_config.image_end_token;
        
        std::vector<ov::Tensor> single_images = to_single_image_tensors(images);

        std::string formatted_prompt;
        std::vector<ov::Tensor> image_embeds;
        image_embeds.reserve(single_images.size());
        
        for (const auto& image : single_images) {
            EncodedImage encoded_image = m_vision_encoder.encode(image);
            ov::Tensor single_image_embeds = encoded_image.resized_source;

            const size_t num_patches = single_image_embeds.get_shape().at(0);
            const size_t num_image_tokens = single_image_embeds.get_shape().at(1);
            
            formatted_prompt += image_start_token;
            for (int i = 0; i < num_patches * num_image_tokens; ++i) {
                formatted_prompt += image_context_token;
            }
            formatted_prompt += image_end_token + "\n";

            image_embeds.push_back(std::move(single_image_embeds));
        }
        formatted_prompt += prompt;

        ov::Tensor input_ids = get_encoded_input_ids(formatted_prompt, metrics);
        ov::Tensor text_embeds = m_embedding.infer(input_ids);

        if (images.empty()) {
            return text_embeds;
        }
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        ov::Tensor encoded_image_context_token = m_tokenizer.encode(image_context_token, ov::genai::add_special_tokens(false)).input_ids;
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
        metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        int64_t image_context_token_id = encoded_image_context_token.data<int64_t>()[encoded_image_context_token.get_size() - 1];
        return merge_text_and_image_embeddings_internvl(input_ids, text_embeds, image_embeds, image_context_token_id);
    }

protected:
    ov::Tensor merge_text_and_image_embeddings_internvl(
        const ov::Tensor& input_ids,
        const ov::Tensor& text_embeds,
        const std::vector<ov::Tensor>& image_embeds,
        int64_t image_context_token_id
    ) {
        auto text_embeds_shape = text_embeds.get_shape();
        size_t batch_size = text_embeds_shape.at(0);
        size_t seq_len = text_embeds_shape.at(1);
        size_t embed_dim = text_embeds_shape.at(2);

        ov::Tensor merged_embeds(text_embeds.get_element_type(), text_embeds_shape);

        const float* text_embeds_data = text_embeds.data<float>();
        const int64_t* input_ids_data = input_ids.data<int64_t>();
        float* merged_embeds_data = merged_embeds.data<float>();

        size_t flattened_size = batch_size * seq_len;
        std::vector<bool> image_context_tokens_mask(flattened_size, false);
        size_t image_context_tokens_count = 0;

        for (size_t i = 0; i < flattened_size; ++i) {
            if (input_ids_data[i] == image_context_token_id) {
                image_context_tokens_mask[i] = true;
                ++image_context_tokens_count;
            }
        }

        OPENVINO_ASSERT(image_context_tokens_count > 0, "input_ids does not contain image context token ids");

        size_t image_idx = 0;
        size_t image_context_token_idx = 0;
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                size_t flat_idx = i * seq_len + j;
                size_t offset = flat_idx * embed_dim;

                if (image_context_tokens_mask[flat_idx]) {
                    const ov::Tensor& single_image_embeds = image_embeds[image_idx];
                    const size_t num_all_image_tokens = single_image_embeds.get_shape().at(0) * single_image_embeds.get_shape().at(1); // num_patches * num_image_tokens
                    const float* image_embeds_data = single_image_embeds.data<float>();
                    std::copy_n(image_embeds_data + image_context_token_idx * embed_dim,
                                embed_dim,
                                merged_embeds_data + offset);
                    
                    ++image_context_token_idx;

                    if (image_context_token_idx == num_all_image_tokens) {
                        ++image_idx;
                        image_context_token_idx = 0;
                    }
                } else {
                    std::copy_n(text_embeds_data + offset, embed_dim, merged_embeds_data + offset);
                }
            }
        }

        return merged_embeds;
    }
};

namespace {
namespace phi3_v {
// Reimplementation of python
// N, L, C = image_features.shape
// assert L == 24 * 24 and C == 1024 and N % (h_crop * w_crop) == 0
// num_images = N // (h_crop * w_crop)
// H = int(L**0.5)
// print(L, H)
// image_features_hd = (
//     image_features.reshape(N, H, H, C)  # N, 24, 24, 1024
//     .reshape(N, H // 2, 2, H // 2, 2, C)  # N, 12, 2, 12, 2, 1024
//     .permute(0, 1, 3, 2, 4, 5)  # N, 12, 12, 2, 2, 1024
//     .reshape(N, -1, 4 * C)  # N, 144, 4096
//     .reshape(num_images, h_crop, w_crop, H // 2, H // 2, -1)  # n_img, h_crop, w_crop, 12, 12, 4096
//     .permute(0, 1, 3, 2, 4, 5)  # n_img, h_crop, 12, w_crop, 12, 4096
//     .reshape(num_images, h_crop * H // 2, w_crop * H // 2, 4 * C)  # n_img, h_crop*12, w_crop*12, 4096
// )
// Obtained in the following way
// import torch
// import openvino as ov
// import numpy as np
// class Model(torch.nn.Module):
//     def forward(self, image_features, h_crop, w_crop):
//         """
//         image_features: (num_images*num_crops, 24*24, 1024)
//         output: (num_images, h_crop*12, w_crop*12, 4096), h_crop*w_crop == num_crops
//         """
//         N, L, C = image_features.shape
//         num_images = N // (h_crop * w_crop)
//         H = (torch.tensor(L, dtype=torch.float32)**0.5).int()
//         image_features_hd = (
//             image_features.reshape(N, H, H, C)  # N, 24, 24, 1024
//             .reshape(N, H // 2, 2, H // 2, 2, C)  # N, 12, 2, 12, 2, 1024
//             .permute(0, 1, 3, 2, 4, 5)  # N, 12, 12, 2, 2, 1024
//             .reshape(N, -1, 4 * C)  # N, 144, 4096
//             .reshape(num_images, h_crop, w_crop, H // 2, H // 2, -1)  # n_img, h_crop, w_crop, 12, 12, 4096
//             .permute(0, 1, 3, 2, 4, 5)  # n_img, h_crop, 12, w_crop, 12, 4096
//             .reshape(num_images, h_crop * H // 2, w_crop * H // 2, 4 * C)  # n_img, h_crop*12, w_crop*12, 4096
//         return {"o": image_features_hd}
// model = Model()
// example_input = {"image_features": torch.rand((4, 576, 1024), dtype=torch.float32), "h_crop": torch.tensor(2, dtype=torch.int32), "w_crop": torch.tensor(2, dtype=torch.int32)}
// ov_model = ov.convert_model(model, example_input=example_input, input=ov.PartialShape([-1, 576, 1024]))
// # ov_model.outputs[0].get_tensor().set_names({"out"})
// ov.save_model(ov_model, "reshape_hd_patches_2x2merge.xml")
// inp = np.arange(4 * 576 * 1024).reshape([4, 576, 1024])
// test = ov.Core().compile_model(ov_model, "CPU")
// print(ov_model)
// print(test([inp, 2, 2])["o"].flatten())
// 2. Run https://github.com/slyalin/openvino_devtools/blob/bcd4a51b1354b24b2316ac3e1c77b2f87ae7a497/openvino_devtools/ov2py.py with the IR.
// 3. Translate the printed Python implementation to C++.
ov::InferRequest create_hd_feature_transformer() {
    using namespace ov;
    using namespace element;
    using namespace opset13;
    using namespace std;
    auto t0 = make_shared<Parameter>(f32, PartialShape{-1, 576, 1024});
    auto t1 = make_shared<Parameter>(i32, PartialShape{});
    auto t2 = make_shared<Parameter>(i32, PartialShape{});
    auto t3 = make_shared<ShapeOf>(t0);
    auto t4 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{0});
    auto t5 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{0});
    auto t6 = make_shared<Gather>(t3, t4, t5);
    auto t7 = make_shared<Constant>(i64, Shape{1}, vector<int64_t>{1});
    auto t8 = make_shared<Reshape>(t6, t7, false);
    auto t9 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{1});
    auto t10 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{0});
    auto t11 = make_shared<Gather>(t3, t9, t10);
    auto t12 = make_shared<Convert>(t11, element::f32);
    auto t13 = make_shared<Constant>(f32, Shape{}, vector<float>{0.5});
    auto t14 = make_shared<Power>(t12, t13, "numpy");
    auto t15 = make_shared<Convert>(t14, element::i32);
    auto t16 = make_shared<Convert>(t15, element::i64);
    auto t17 = make_shared<Constant>(i32, Shape{}, vector<int32_t>{0});
    auto t18 = make_shared<Unsqueeze>(t16, t17);
    auto t19 = make_shared<Constant>(i64, Shape{1}, vector<int64_t>{2});
    auto t20 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{0});
    auto t21 = make_shared<Gather>(t3, t19, t20);
    auto t22 = make_shared<Concat>(NodeVector{t8, t18, t18, t21}, 0);
    auto t23 = make_shared<Reshape>(t0, t22, false);
    auto t24 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{2});
    auto t25 = make_shared<Divide>(t16, t24, "numpy");
    auto t26 = make_shared<Floor>(t25);
    auto t27 = make_shared<Constant>(i32, Shape{}, vector<int32_t>{0});
    auto t28 = make_shared<Unsqueeze>(t26, t27);
    auto t29 = make_shared<Constant>(i64, Shape{1}, vector<int64_t>{2});
    auto t30 = make_shared<Constant>(i64, Shape{1}, vector<int64_t>{2});
    auto t31 = make_shared<Concat>(NodeVector{t8, t28, t29, t28, t30, t21}, 0);
    auto t32 = make_shared<Reshape>(t23, t31, false);
    auto t33 = make_shared<Constant>(i64, Shape{6}, vector<int64_t>{0, 1, 3, 2, 4, 5});
    auto t34 = make_shared<Transpose>(t32, t33);
    auto t35 = make_shared<Constant>(i64, Shape{1}, vector<int64_t>{-1});
    auto t36 = make_shared<Constant>(i64, Shape{1}, vector<int64_t>{4});
    auto t37 = make_shared<Multiply>(t21, t36, "numpy");
    auto t38 = make_shared<Concat>(NodeVector{t8, t35, t37}, 0);
    auto t39 = make_shared<Reshape>(t34, t38, false);
    auto t40 = make_shared<Multiply>(t1, t2, "numpy");
    auto t41 = make_shared<Convert>(t40, element::i64);
    auto t42 = make_shared<Divide>(t6, t41, "numpy");
    auto t43 = make_shared<Floor>(t42);
    auto t44 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{0});
    auto t45 = make_shared<Unsqueeze>(t43, t44);
    auto t46 = make_shared<Convert>(t1, element::i64);
    auto t47 = make_shared<Unsqueeze>(t46, t44);
    auto t48 = make_shared<Convert>(t2, element::i64);
    auto t49 = make_shared<Unsqueeze>(t48, t44);
    auto t50 = make_shared<Constant>(i64, Shape{1}, vector<int64_t>{-1});
    auto t51 = make_shared<Concat>(NodeVector{t45, t47, t49, t28, t28, t50}, 0);
    auto t52 = make_shared<Reshape>(t39, t51, false);
    auto t53 = make_shared<Constant>(i64, Shape{6}, vector<int64_t>{0, 1, 3, 2, 4, 5});
    auto t54 = make_shared<Transpose>(t52, t53);
    auto t55 = make_shared<Multiply>(t1, t15, "numpy");
    auto t56 = make_shared<Convert>(t55, element::i64);
    auto t57 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{2});
    auto t58 = make_shared<Divide>(t56, t57, "numpy");
    auto t59 = make_shared<Floor>(t58);
    auto t60 = make_shared<Constant>(i32, Shape{}, vector<int32_t>{0});
    auto t61 = make_shared<Unsqueeze>(t59, t60);
    auto t62 = make_shared<Multiply>(t2, t15, "numpy");
    auto t63 = make_shared<Convert>(t62, element::i64);
    auto t64 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{2});
    auto t65 = make_shared<Divide>(t63, t64, "numpy");
    auto t66 = make_shared<Floor>(t65);
    auto t67 = make_shared<Unsqueeze>(t66, t60);
    auto t68 = make_shared<Concat>(NodeVector{t45, t61, t67, t37}, 0);
    auto t69 = make_shared<Reshape>(t54, t68, false);
    shared_ptr<Model> model = make_shared<Model>(make_shared<Result>(t69), ParameterVector{t0, t1, t2});
    return utils::singleton_core().compile_model(
        model, "CPU"
    ).create_infer_request();
}

ov::Tensor reshape_hd_patches_2x2merge(const ov::Tensor& image_features, size_t h_crop, size_t w_crop, InferRequest& hd_feature_transformer) {
    ov::Shape shape = image_features.get_shape();
    OPENVINO_ASSERT(3 == shape.size());
    OPENVINO_ASSERT(24 * 24 == shape.at(1));
    OPENVINO_ASSERT(1024 == shape.at(2));
    hd_feature_transformer.set_input_tensor(0, image_features);
    ov::Tensor height{ov::element::i32, {}, &h_crop};
    hd_feature_transformer.set_input_tensor(1, height);
    ov::Tensor width{ov::element::i32, {}, &w_crop};
    hd_feature_transformer.set_input_tensor(2, width);
    hd_feature_transformer.infer();
    return hd_feature_transformer.get_output_tensor();
}

// image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
// output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
ov::Tensor add_image_newline(const ov::Tensor& image_features_hd, const std::vector<float>& sub_GN) {
    const ov::Shape& nhwc = image_features_hd.get_shape();  // [N, 12*h_crop, 12*w_crop, 4096]
    const float* in = image_features_hd.data<float>();
    ov::Tensor image_features_hd_new_line{ov::element::f32, {nhwc.at(0), nhwc.at(1) * (nhwc.at(2) + 1), nhwc.at(3)}};
    float* out = image_features_hd_new_line.data<float>();
    for (size_t batch_id = 0; batch_id < nhwc.at(0); ++batch_id) {
        for (size_t row_id = 0; row_id < nhwc.at(1); ++row_id) {
            for (size_t col_id = 0; col_id < nhwc.at(2); ++col_id) {
                std::copy_n(
                    in + batch_id * nhwc.at(1) * nhwc.at(2) * nhwc.at(3) + row_id * nhwc.at(2) * nhwc.at(3) + col_id * nhwc.at(3),
                    nhwc.at(3),
                    out + batch_id * nhwc.at(1) * (nhwc.at(2) + 1) * nhwc.at(3) + row_id * (nhwc.at(2) + 1) * nhwc.at(3) + col_id * nhwc.at(3)
                );
            }
            std::copy(
                sub_GN.begin(),
                sub_GN.end(),
                out + batch_id * nhwc.at(1) * (nhwc.at(2) + 1) * nhwc.at(3) + row_id * (nhwc.at(2) + 1) * nhwc.at(3) + nhwc.at(2) * nhwc.at(3)
            );
        }
    }
    return image_features_hd_new_line;
}

ov::Tensor concatenate_2d(const ov::Tensor& first_1lf, const std::vector<float>& second_f, const ov::Tensor& third_1lf) {
    size_t first_l = first_1lf.get_shape().at(1);
    constexpr size_t second_l = 1;
    size_t third_l = third_1lf.get_shape().at(1);
    size_t features = first_1lf.get_shape().at(2);
    OPENVINO_ASSERT(second_f.size() == features);
    ov::Tensor out_1lf{ov::element::f32, {1, first_l + second_l + third_l, features}};
    float* out = out_1lf.data<float>();
    std::copy_n(first_1lf.data<float>(), first_l * features, out);
    std::copy(second_f.begin(), second_f.end(), out + first_l * features);
    std::copy_n(third_1lf.data<float>(), third_l * features, out + (first_l + second_l) * features);
    return out_1lf;
}

// image_features.resized_source: (num_crops+1, 24*24, 1024)
ov::Tensor hd_feature_transform(const EncodedImage& image_features, InferRequest& hd_feature_transformer, const std::vector<float>& sub_GN, const std::vector<float>& glb_GN, ov::InferRequest& vision_projection) {
    const ov::Shape& image_features_shape = image_features.resized_source.get_shape();
    ov::Tensor global_image_features{ov::element::f32, {1, image_features_shape.at(1), image_features_shape.at(2)}, image_features.resized_source.data<float>()};
    // global feature can be viewed as a special HD case with num_crops 1x1
    ov::Tensor global_image_features_hd = reshape_hd_patches_2x2merge(global_image_features, 1, 1, hd_feature_transformer);
    ov::Tensor global_image_features_hd_newline = add_image_newline(global_image_features_hd, sub_GN);  // [1,12*(12+1),4096]
    constexpr size_t INPUT_IMAGE_SIZE = 336;
    size_t h_crop = image_features.resized_source_size.height / INPUT_IMAGE_SIZE;
    size_t w_crop = image_features.resized_source_size.width / INPUT_IMAGE_SIZE;
    size_t num_crops = h_crop * w_crop;

    // NOTE: real num_crops is padded
    // (num_crops, 24*24, 1024)
    ov::Tensor sub_image_features{ov::element::f32, {
        num_crops,
        image_features_shape.at(1),
        image_features_shape.at(2)
    }, image_features.resized_source.data<float>() + image_features_shape.at(1) * image_features_shape.at(2)};
    ov::Tensor sub_image_features_hd = reshape_hd_patches_2x2merge(sub_image_features, h_crop, w_crop, hd_feature_transformer);  // [1, 24, 24, 4096]
    ov::Tensor sub_image_features_hd_newline = add_image_newline(sub_image_features_hd, sub_GN);  // [1,h_crop*12*(w_crop*12+1), 4096]
    ov::Tensor image_embeddings = concatenate_2d(sub_image_features_hd_newline, glb_GN, global_image_features_hd_newline);  // [1,l,4096]
    vision_projection.set_input_tensor(image_embeddings);
    vision_projection.infer();
    ov::Tensor out = vision_projection.get_output_tensor();
    ov::Tensor res{out.get_element_type(), out.get_shape()};
    out.copy_to(res);
    return res;
}

std::vector<ov::Tensor> split_tokenize(const std::string& text, ov::genai::Tokenizer& tokenizer) {
    constexpr int make_suffix_iterator = -1;
    std::regex rgx{R"(<\|image_\d+\|>)"};
    std::sregex_token_iterator iter{
        text.begin(),
        text.end(),
        rgx,
        make_suffix_iterator
    };
    std::vector<ov::Tensor> tokenized;
    for ( ; iter != std::sregex_token_iterator{}; ++iter) {
        if (iter->str().empty()) {
            continue;
        }
        std::string substr = *iter;
        tokenized.push_back(tokenizer.encode(substr, ov::genai::add_special_tokens(true)).input_ids);
    }
    return tokenized;
}

ov::Tensor insert_image_placeholders(const std::vector<ov::Tensor>& chunks, const std::vector<size_t>& tokens_per_images) {
    size_t merged_length = 0;
    for (const ov::Tensor& chunk : chunks) {
        merged_length += chunk.get_shape().at(1);
    }
    merged_length += std::accumulate(tokens_per_images.begin(), tokens_per_images.end(), 0);
    ov::Tensor merged{ov::element::i64, {1, merged_length}};
    size_t offset = 0;
    int64_t image_id = 0;
    for (const ov::Tensor& chunk : chunks) {
        size_t length = chunk.get_shape().at(1);
        std::copy_n(
            chunk.data<int64_t>(),
            length,
            merged.data<int64_t>() + offset
        );
        offset += length;
        if (offset < merged_length) {
            std::fill_n(
                merged.data<int64_t>() + offset,
                tokens_per_images.at(image_id),
                -image_id - 1  // It could be just -image_id. -1 is for consistency with the original implementation.
            );
            offset += tokens_per_images.at(image_id);
            ++image_id;
        }
    }
    return merged;
}

std::vector<ov::Tensor> drop_image_placeholders(const ov::Tensor& tokens) {
    std::vector<ov::Tensor> chunks;
    size_t offset = 0;
    while (offset < tokens.get_shape().at(1)) {
        size_t length = 0;
        while (offset + length < tokens.get_shape().at(1) && tokens.data<int64_t>()[offset + length] >= 0) {
            ++length;
        }
        chunks.emplace_back(ov::element::i64, ov::Shape{1, length}, tokens.data<int64_t>() + offset);
        offset += length;
        while (offset < tokens.get_shape().at(1) && tokens.data<int64_t>()[offset] < 0) {
            ++offset;
        }
    }
    return chunks;
}
}  // namespace phi3_v
}  // anonymous namespace

class InputsEmbedderPhi3V : public InputsEmbedder::IInputsEmbedder {
public:
    ov::InferRequest m_hd_feature_transformer;
    ov::InferRequest m_vision_projection;
    std::vector<size_t> m_tokens_per_images;

    InputsEmbedderPhi3V(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config
    ):
        IInputsEmbedder(vlm_config, model_dir, device, device_config),
        m_hd_feature_transformer{phi3_v::create_hd_feature_transformer()},
        m_vision_projection{utils::singleton_core().compile_model(model_dir / "openvino_vision_projection_model.xml", device, {}).create_infer_request()} {}

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::Tensor>& images, ov::genai::VLMPerfMetrics& metrics) override {
        std::vector<ov::Tensor> images_features_proj;
        std::stringstream images_prompt;
        for (const ov::Tensor& image : to_single_image_tensors(images)) {
            EncodedImage encoded_image = m_vision_encoder.encode(image);
            images_features_proj.push_back(phi3_v::hd_feature_transform(encoded_image, m_hd_feature_transformer, m_vlm_config.sub_GN, m_vlm_config.glb_GN, m_vision_projection));
            m_tokens_per_images.push_back(images_features_proj.back().get_shape().at(1));
            images_prompt << "<|image_" << m_tokens_per_images.size() << "|>\n";
        }
        images_prompt << prompt;
        std::vector<ov::Tensor> new_chat_tokens;
        if (m_is_chat_conversation) {
            m_history.push_back({{"role", "user"}, {"content", images_prompt.str()}});
            constexpr bool add_generation_prompt = true;
            std::string new_templated_chat_history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
            auto start_tokenizer_time = std::chrono::steady_clock::now();
            new_chat_tokens = phi3_v::split_tokenize(new_templated_chat_history, m_tokenizer);
            auto end_tokenizer_time = std::chrono::steady_clock::now();
            metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        } else {
            std::string templated_prompt;
            if (m_apply_chat_template) {
                ChatHistory history({{{"role", "user"}, {"content", images_prompt.str()}}});
                constexpr bool add_generation_prompt = true;
                templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
            } else {
                templated_prompt = images_prompt.str();
            }
            auto start_tokenizer_time = std::chrono::steady_clock::now();
            new_chat_tokens = phi3_v::split_tokenize(templated_prompt, m_tokenizer);
            auto end_tokenizer_time = std::chrono::steady_clock::now();
            metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        }
        ov::Tensor new_merged_tokens = phi3_v::insert_image_placeholders(new_chat_tokens, m_tokens_per_images);
        ov::Tensor new_tokens = update_history(new_merged_tokens);
        m_kv_cache_state.add_inputs(new_tokens);

        std::vector<ov::Tensor> tokens = phi3_v::drop_image_placeholders(new_tokens);
        // if <|image_i|> tag is in the begining, it doesn't split tokes into separate sequences and tokens.size() == images_features_proj.size().
        OPENVINO_ASSERT(tokens.size() == images_features_proj.size() + 1 || tokens.size() == images_features_proj.size());
        size_t features_length = 0;
        for (size_t im_id = 0; im_id < images_features_proj.size(); ++im_id) {
            size_t text_length = tokens.at(im_id).get_shape().at(1);
            size_t im_length = images_features_proj.at(im_id).get_shape().at(1);
            features_length += text_length + im_length;
        }
        if (tokens.size() > images_features_proj.size()) {
            features_length += tokens.back().get_shape().at(1);
        }
        ov::Tensor inputs_embeds{ov::element::f32, {1, features_length, m_vlm_config.hidden_size}};
        size_t offset = 0;
        if (tokens.size() > images_features_proj.size()) {
            const ov::Tensor& text_embeds = m_embedding.infer(tokens.at(0));
            size_t text_length = text_embeds.get_shape().at(1);
            std::copy_n(
                text_embeds.data<float>(),
                text_embeds.get_size(),
                inputs_embeds.data<float>()
            );
            offset = text_length;
            tokens.erase(tokens.begin());
        }
        for (size_t im_id = 0; im_id < images_features_proj.size(); ++im_id) {
            const ov::Tensor& image_embeds = images_features_proj.at(im_id);
            size_t im_length = image_embeds.get_shape().at(1);
            std::copy_n(
                image_embeds.data<float>(),
                image_embeds.get_size(),
                inputs_embeds.data<float>() + offset * m_vlm_config.hidden_size
            );
            offset += im_length;
            const ov::Tensor& text_embeds = m_embedding.infer(tokens.at(im_id));
            size_t text_length = text_embeds.get_shape().at(1);
            std::copy_n(
                text_embeds.data<float>(),
                text_embeds.get_size(),
                inputs_embeds.data<float>() + offset * m_vlm_config.hidden_size
            );
            offset += text_length;
        }

        if (!m_is_chat_conversation) {
            m_tokens_per_images.clear();
        }

        return inputs_embeds;
    }

    virtual void start_chat(const std::string& system_message) override {
        IInputsEmbedder::start_chat(system_message);
        m_tokens_per_images.clear();
    }

    virtual void finish_chat() override {
        IInputsEmbedder::finish_chat();
        m_tokens_per_images.clear();
    }
};

class InputsEmbedderQwen2VL : public InputsEmbedder::IInputsEmbedder {
    // A model for merging image embeddings (hidden states), rotary_pos_emb and attension_mask.
    // Inputs:
    //  - hidden_states: [N, embed_dim]
    //  - rotary_pos_emb: [?, 40]
    //  - attention_mask: [1, ?, ?]
    // Output: [N, hidden_size]
    ov::InferRequest m_vision_embeddings_merger;

    ov::Tensor m_position_ids;
    int64_t m_rope_delta = 0;

public:
    InputsEmbedderQwen2VL(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config) :
        IInputsEmbedder(vlm_config, model_dir, device, device_config) {
            auto compiled_model = utils::singleton_core().compile_model(model_dir / "openvino_vision_embeddings_merger_model.xml", device, device_config);
            ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings merger model");
            m_vision_embeddings_merger = compiled_model.create_infer_request();
        }

    InputsEmbedderQwen2VL(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer, 
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config) :
        IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
            m_vision_embeddings_merger = utils::singleton_core().compile_model(
                get_model_weights_pair(models_map, "vision_embeddings_merger").first,
                get_model_weights_pair(models_map, "vision_embeddings_merger").second,
                device,
                device_config
            ).create_infer_request();
        }
    
    virtual ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::Tensor>& images, ov::genai::VLMPerfMetrics& metrics) override {
        std::string formatted_prompt;

        std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
        std::vector<ov::Tensor> image_embeds;
        std::vector<std::array<size_t, 3>> images_grid_thw;
        image_embeds.reserve(single_images.size());
        images_grid_thw.reserve(single_images.size());
        
        for (const auto& image : single_images) {
            EncodedImage encoded_image = m_vision_encoder.encode(image);
            ov::Tensor single_image_embeds = encoded_image.resized_source;
            image_embeds.push_back(std::move(single_image_embeds));

            size_t grid_t = 1;
            size_t grid_h = encoded_image.resized_source_size.height;
            size_t grid_w = encoded_image.resized_source_size.width;
            images_grid_thw.push_back({grid_t, grid_h, grid_w});

            size_t merge_length = std::pow(m_vision_encoder.m_processor_config.merge_size, 2);
            size_t num_image_pad_tokens = grid_t * grid_h * grid_w / merge_length;

            formatted_prompt += m_vlm_config.vision_start_token;
            for (int i = 0; i < num_image_pad_tokens; i++) {
                formatted_prompt += m_vlm_config.image_pad_token;
            }
            formatted_prompt += m_vlm_config.vision_end_token;
        }
        formatted_prompt += prompt;

        ov::Tensor input_ids = get_encoded_input_ids(formatted_prompt, metrics);
        ov::Tensor text_embeds = m_embedding.infer(input_ids);

        auto start_tokenizer_time = std::chrono::steady_clock::now();
        ov::Tensor encoded_vision_start_token = m_tokenizer.encode(m_vlm_config.vision_start_token, ov::genai::add_special_tokens(false)).input_ids;
        ov::Tensor encoded_image_pad_token = m_tokenizer.encode(m_vlm_config.image_pad_token, ov::genai::add_special_tokens(false)).input_ids;
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
        metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        int64_t vision_start_token_id = encoded_vision_start_token.data<int64_t>()[encoded_vision_start_token.get_size() - 1];
        int64_t image_pad_token_id = encoded_image_pad_token.data<int64_t>()[encoded_image_pad_token.get_size() - 1];

        m_position_ids = create_position_ids(input_ids, images_grid_thw, vision_start_token_id);

        int64_t position_ids_max_element = *std::max_element(m_position_ids.data<int64_t>(), m_position_ids.data<int64_t>() + m_position_ids.get_size());
        m_rope_delta = position_ids_max_element + 1 - static_cast<int64_t>(input_ids.get_shape().at(1));

        if (images.empty()) {
            return text_embeds;
        }

        return merge_text_and_image_embeddings_qwen2vl(input_ids, text_embeds, image_embeds, images_grid_thw, image_pad_token_id);
    }

    virtual std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size) override {
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

    virtual void start_chat(const std::string& system_message) override {
        IInputsEmbedder::start_chat(system_message);
        m_position_ids = ov::Tensor();
        m_rope_delta = 0;
    }

    virtual void finish_chat() override {
        IInputsEmbedder::finish_chat();
        m_position_ids = ov::Tensor();
        m_rope_delta = 0;
    }
protected:
    ov::Tensor merge_text_and_image_embeddings_qwen2vl(
        const ov::Tensor& input_ids,
        const ov::Tensor& text_embeds,
        const std::vector<ov::Tensor>& image_embeds,
        const std::vector<std::array<size_t, 3>> images_grid_thw,
        const int64_t image_pad_token_id
    ) {
        // Calculate cumulative sequence lengths for attention mask
        std::vector<int32_t> cu_seqlens;
        cu_seqlens.push_back(0);
        int32_t cumsum = 0;
        for (const auto& grid_thw : images_grid_thw) {
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

        // Concatenate image embeddings 
        ov::Tensor concatenated_images;
        if (image_embeds.size() == 1) {
            concatenated_images = image_embeds.at(0);
        } else {
            size_t total_length = 0;
            for (const auto& embed : image_embeds) {
                total_length += embed.get_shape().at(0);
            }
            size_t hidden_dim = image_embeds.at(0).get_shape().at(1);
            
            concatenated_images = ov::Tensor(image_embeds.at(0).get_element_type(), {total_length, hidden_dim});
            float* concat_data = concatenated_images.data<float>();
            
            size_t offset = 0;
            for (const auto& embed : image_embeds) {
                size_t embed_size = embed.get_shape().at(0) * embed.get_shape().at(1);
                std::memcpy(concat_data + offset, embed.data(), embed.get_byte_size());
                offset += embed_size;
            }
        }

        ov::Tensor rotary_pos_emb = get_rotary_pos_emb(images_grid_thw);

        m_vision_embeddings_merger.set_tensor("hidden_states", concatenated_images);
        m_vision_embeddings_merger.set_tensor("attention_mask", attention_mask);
        m_vision_embeddings_merger.set_tensor("rotary_pos_emb", rotary_pos_emb);
        m_vision_embeddings_merger.infer();
        ov::Tensor processed_vision_embeds = m_vision_embeddings_merger.get_output_tensor();

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

    ov::Tensor get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw) {  
        const size_t spatial_merge_size = m_vision_encoder.m_processor_config.merge_size;

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
        const size_t dim = m_vision_embeddings_merger.get_tensor("rotary_pos_emb").get_shape().at(1);
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

    ov::Tensor create_position_ids(
        const ov::Tensor& input_ids_tensor,
        const std::vector<std::array<size_t, 3>>& images_grid_thw,
        const int64_t vision_start_token_id
    ) {
        const size_t spatial_merge_size = m_vision_encoder.m_processor_config.merge_size;
        
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
            if (grid_idx < images_grid_thw.size()) {
                const auto& grid = images_grid_thw.at(grid_idx);
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
};

InputsEmbedder::InputsEmbedder(const VLMConfig& vlm_config,
                               const std::filesystem::path& model_dir,
                               const std::string& device,
                               const ov::AnyMap device_config) {
    if (vlm_config.model_type == VLMModelType::MINICPM) {
        m_impl = std::make_shared<InputsEmbedderMiniCPM>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA) {
        m_impl = std::make_shared<InputsEmbedderLLaVA>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA_NEXT) {
        m_impl = std::make_shared<InputsEmbedderLLaVANext>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::INTERNVL_CHAT) {
        m_impl = std::make_shared<InputsEmbedderInternVLChat>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::PHI3_V) {
        m_impl = std::make_shared<InputsEmbedderPhi3V>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::QWEN2_VL) {
        m_impl = std::make_shared<InputsEmbedderQwen2VL>(vlm_config, model_dir, device, device_config);
    } else {
        OPENVINO_THROW("Unsupported model type in VLM InputsEmbedder class. Please, create feature request on new model support");
    }
}

InputsEmbedder::InputsEmbedder(const VLMConfig& vlm_config,
                               const ModelsMap& models_map,
                               const Tokenizer& tokenizer,
                               const std::filesystem::path& config_dir_path,
                               const std::string& device,
                               const ov::AnyMap device_config) {
    if (vlm_config.model_type == VLMModelType::MINICPM) {
        m_impl = std::make_shared<InputsEmbedderMiniCPM>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA) {
        m_impl = std::make_shared<InputsEmbedderLLaVA>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA_NEXT) {
        m_impl = std::make_shared<InputsEmbedderLLaVANext>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::INTERNVL_CHAT) {
        m_impl = std::make_shared<InputsEmbedderInternVLChat>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::QWEN2_VL) {
        m_impl = std::make_shared<InputsEmbedderQwen2VL>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else {
        OPENVINO_THROW("Unsupported model type in VLM InputsEmbedder class. Please, create feature request on new model support");
    }
}

ov::Tensor InputsEmbedder::get_inputs_embeds(const std::string& prompt, const std::vector<ov::Tensor>& images, ov::genai::VLMPerfMetrics& metrics) {
    return m_impl->get_inputs_embeds(prompt, images, metrics);
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedder::get_position_ids(const size_t inputs_embeds_size, const size_t history_size) {
    return m_impl->get_position_ids(inputs_embeds_size, history_size);
}

EmbeddingsModel InputsEmbedder::get_embedding_model() const {
    return m_impl->get_embedding_model();
}

void InputsEmbedder::set_stop_token_ids(const std::set<int64_t>& stop_token_ids) {
    return m_impl->set_stop_token_ids(stop_token_ids);
}

KVCacheState& InputsEmbedder::get_kv_cache_state() {
    return  m_impl->get_kv_cache_state();
}

size_t InputsEmbedder::get_num_tokens_to_remove_from_hist() const {
    return m_impl->get_num_tokens_to_remove_from_hist();
}

Tokenizer InputsEmbedder::get_tokenizer() const {
    return m_impl->get_tokenizer();
}

void InputsEmbedder::start_chat(const std::string& system_message) {
    return m_impl->start_chat(system_message);
}

void InputsEmbedder::update_chat_history(const std::string& decoded_results) {
    return m_impl->update_chat_history(decoded_results);
}

void InputsEmbedder::set_apply_chat_template_status(bool apply_chat_template) {
    return m_impl->set_apply_chat_template_status(apply_chat_template);
}

void InputsEmbedder::finish_chat() {
    return m_impl->finish_chat();
}

} // namespace ov::genai
