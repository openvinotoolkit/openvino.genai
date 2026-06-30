// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/embedding_pipeline.hpp"

#include <algorithm>
#include <future>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <tuple>
#include <unordered_set>

#include "openvino/core/except.hpp"
#include "openvino/genai/chat_history.hpp"
#include "openvino/runtime/core.hpp"
#include "text_embedding_utils.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"

namespace {

ov::genai::TextEmbeddingPipeline::Config get_multimodal_config(const ov::AnyMap& properties) {
    ov::genai::TextEmbeddingPipeline::Config config(properties);
    if (!properties.count(ov::genai::pooling_type.name())) {
        config.pooling_type = ov::genai::TextEmbeddingPipeline::PoolingType::LAST_TOKEN;
    }
    return config;
}

ov::Tensor make_text_position_ids(size_t sequence_length) {
    ov::Tensor position_ids(ov::element::i64, {3, 1, sequence_length});
    for (size_t dim_idx = 0; dim_idx < 3; ++dim_idx) {
        int64_t* data = position_ids.data<int64_t>() + dim_idx * sequence_length;
        std::iota(data, data + sequence_length, 0);
    }
    return position_ids;
}

ov::Tensor stack_tensors(const std::vector<ov::Tensor>& tensors) {
    if (tensors.empty()) {
        return ov::Tensor(ov::element::f32, {0, 0});
    }

    const ov::Shape first_shape = tensors.front().get_shape();
    OPENVINO_ASSERT(first_shape.size() == 2 && first_shape[0] == 1,
                    "Expected rank-2 single embedding tensor");
    const size_t embedding_size = first_shape[1];
    ov::Tensor result(ov::element::f32, {tensors.size(), embedding_size});
    float* result_data = result.data<float>();

    for (size_t row_idx = 0; row_idx < tensors.size(); ++row_idx) {
        const ov::Tensor& tensor = tensors[row_idx];
        const ov::Shape shape = tensor.get_shape();
        OPENVINO_ASSERT(tensor.get_element_type() == ov::element::f32,
                        "Expected f32 embedding tensor, got ",
                        tensor.get_element_type());
        OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1 && shape[1] == embedding_size,
                        "Expected all embeddings to have shape [1, ",
                        embedding_size,
                        "]");
        std::copy_n(tensor.data<const float>(), embedding_size, result_data + row_idx * embedding_size);
    }

    return result;
}

ov::Tensor embedding_results_to_tensor(const ov::genai::EmbeddingResults& embedding_results) {
    return std::visit([](const auto& values) {
        if (values.empty()) {
            return ov::Tensor(ov::element::f32, {0, 0});
        }

        const size_t embedding_size = values.front().size();
        ov::Tensor result(ov::element::f32, {values.size(), embedding_size});
        float* result_data = result.data<float>();

        for (size_t row_idx = 0; row_idx < values.size(); ++row_idx) {
            OPENVINO_ASSERT(values[row_idx].size() == embedding_size,
                            "All embedding vectors must have the same size");
            for (size_t column_idx = 0; column_idx < embedding_size; ++column_idx) {
                result_data[row_idx * embedding_size + column_idx] = static_cast<float>(values[row_idx][column_idx]);
            }
        }

        return result;
    }, embedding_results);
}

bool has_visual_tag(const std::string& text) {
    return text.find("<ov_genai_image_") != std::string::npos ||
           text.find("<ov_genai_video_") != std::string::npos ||
           text.find("<|vision_start|><|image_pad|><|vision_end|>") != std::string::npos ||
           text.find("<|vision_start|><|video_pad|><|vision_end|>") != std::string::npos;
}

std::string append_visual_tags(const std::string& text, size_t image_count, size_t video_count) {
    if ((image_count == 0 && video_count == 0) || has_visual_tag(text)) {
        return text;
    }

    std::string result = text;
    for (size_t image_idx = 0; image_idx < image_count; ++image_idx) {
        result += "<ov_genai_image_" + std::to_string(image_idx) + ">";
    }
    for (size_t video_idx = 0; video_idx < video_count; ++video_idx) {
        result += "<ov_genai_video_" + std::to_string(video_idx) + ">";
    }
    return result;
}

}  // namespace

namespace ov {
namespace genai {

class EmbeddingPipeline::EmbeddingPipelineImpl {
public:
    EmbeddingPipelineImpl(const std::filesystem::path& models_path,
                          const std::string& device,
                          const TextEmbeddingPipeline::Config& config,
                          const ov::AnyMap& properties)
        : m_config{config} {
        m_config.validate();
        const ov::AnyMap plugin_properties = utils::remove_config_properties(properties);
        try {
            init_multimodal(models_path, device, plugin_properties);
            m_mode = Mode::MULTIMODAL;
        } catch (const std::exception& multimodal_error) {
            try {
                m_text_embedding_pipeline = std::make_unique<TextEmbeddingPipeline>(models_path,
                                                                                   device,
                                                                                   m_config,
                                                                                   plugin_properties);
                m_mode = Mode::TEXT_ONLY;
            } catch (const std::exception& text_error) {
                OPENVINO_THROW("EmbeddingPipeline initialization failed. "
                               "Multimodal initialization error: ",
                               multimodal_error.what(),
                               ". TextEmbeddingPipeline fallback error: ",
                               text_error.what());
            }
        }
    }

    EmbeddingPipelineImpl(const std::filesystem::path& models_path,
                          const std::string& device,
                          const ov::AnyMap& properties)
        : m_config{get_multimodal_config(properties)} {
        m_config.validate();
        const ov::AnyMap plugin_properties = utils::remove_config_properties(properties);
        try {
            init_multimodal(models_path, device, plugin_properties);
            m_mode = Mode::MULTIMODAL;
        } catch (const std::exception& multimodal_error) {
            try {
                m_text_embedding_pipeline = std::make_unique<TextEmbeddingPipeline>(models_path,
                                                                                   device,
                                                                                   TextEmbeddingPipeline::Config(properties),
                                                                                   plugin_properties);
                m_mode = Mode::TEXT_ONLY;
            } catch (const std::exception& text_error) {
                OPENVINO_THROW("EmbeddingPipeline initialization failed. "
                               "Multimodal initialization error: ",
                               multimodal_error.what(),
                               ". TextEmbeddingPipeline fallback error: ",
                               text_error.what());
            }
        }
    }

    ov::Tensor embed(const EmbeddingPipeline::TextInput& text, const std::optional<std::string>& prompt) {
        if (m_mode == Mode::TEXT_ONLY) {
            std::lock_guard<std::mutex> async_lock(m_async_mutex);
            OPENVINO_ASSERT(!m_text_async_pending, "Previous asynchronous embed request is still pending");
            std::lock_guard<std::mutex> request_lock(m_request_mutex);
            if (std::holds_alternative<std::string>(text)) {
                const std::vector<std::string> texts{std::get<std::string>(text)};
                if (prompt.has_value()) {
                    return embedding_results_to_tensor(m_text_embedding_pipeline->embed(texts, *prompt));
                }
                return embedding_results_to_tensor(m_text_embedding_pipeline->embed_documents(texts));
            } else {
                const std::vector<std::string>& texts = std::get<std::vector<std::string>>(text);
                if (prompt.has_value()) {
                    return embedding_results_to_tensor(m_text_embedding_pipeline->embed(texts, *prompt));
                }
                return embedding_results_to_tensor(m_text_embedding_pipeline->embed_documents(texts));
            }
        }
        std::lock_guard<std::mutex> async_lock(m_async_mutex);
        OPENVINO_ASSERT(!m_embed_future.valid(), "Previous asynchronous embed request is still pending");
        std::lock_guard<std::mutex> request_lock(m_request_mutex);

        std::vector<std::string> texts = std::holds_alternative<std::string>(text)
            ? std::vector<std::string>{std::get<std::string>(text)}
            : std::get<std::vector<std::string>>(text);

        std::vector<EncodedImage> encoded_images;
        std::vector<EncodedVideo> encoded_videos;
        return extract_multimodal_batch(texts, encoded_images, encoded_videos, prompt);
    }

    ov::Tensor wait() {
        if (m_mode == Mode::TEXT_ONLY) {
            std::lock_guard<std::mutex> async_lock(m_async_mutex);
            OPENVINO_ASSERT(m_text_async_pending, "Asynchronous embed request was not started");
            ov::Tensor result = embedding_results_to_tensor(m_text_embedding_pipeline->wait_embed_documents());
            m_text_async_pending = false;
            return result;
        }
        std::lock_guard<std::mutex> async_lock(m_async_mutex);
        OPENVINO_ASSERT(m_embed_future.valid(), "Asynchronous embed request was not started");
        return m_embed_future.get();
    }

    ov::Tensor embed(const EmbeddingPipeline::TextInput& text,
                     const std::vector<ov::Tensor>& images,
                     const std::vector<ov::Tensor>& videos,
                     const std::vector<VideoMetadata>& videos_metadata,
                     const std::optional<std::string>& prompt) {
        if (m_mode == Mode::TEXT_ONLY) {
            OPENVINO_ASSERT(images.empty() && videos.empty() && videos_metadata.empty(),
                            "TextEmbeddingPipeline fallback is active and does not support image/video input");
            return embed(text, prompt);
        }
        std::lock_guard<std::mutex> async_lock(m_async_mutex);
        OPENVINO_ASSERT(!m_embed_future.valid(), "Previous asynchronous embed request is still pending");
        std::lock_guard<std::mutex> request_lock(m_request_mutex);

        OPENVINO_ASSERT(videos_metadata.empty() || videos_metadata.size() == videos.size(),
                        "videos_metadata size (",
                        videos_metadata.size(),
                        ") must be equal to videos size (",
                        videos.size(),
                        ") or empty");

        std::vector<EncodedImage> encoded_images = m_inputs_embedder->encode_images(images);
        std::vector<EncodedVideo> encoded_videos = m_inputs_embedder->encode_videos(videos, videos_metadata);

        std::vector<std::string> texts = std::holds_alternative<std::string>(text)
            ? std::vector<std::string>{std::get<std::string>(text)}
            : std::get<std::vector<std::string>>(text);

        return extract_multimodal_batch(texts, encoded_images, encoded_videos, prompt);
    }

    void start_embed_async(const EmbeddingPipeline::TextInput& text,
                           const std::vector<ov::Tensor>& images,
                           const std::vector<ov::Tensor>& videos,
                           const std::vector<VideoMetadata>& videos_metadata,
                           const std::optional<std::string>& prompt) {
        const bool is_single_text = std::holds_alternative<std::string>(text);
        if (m_mode == Mode::TEXT_ONLY) {
            OPENVINO_ASSERT(images.empty() && videos.empty() && videos_metadata.empty(),
                            "TextEmbeddingPipeline fallback is active and does not support image/video input");
            std::lock_guard<std::mutex> async_lock(m_async_mutex);
            OPENVINO_ASSERT(!m_text_async_pending, "Previous asynchronous embed request is still pending");
            std::lock_guard<std::mutex> request_lock(m_request_mutex);
            if (prompt.has_value()) {
                const std::vector<std::string> texts = is_single_text ? std::vector<std::string>{std::get<std::string>(text)}
                                                                      : std::get<std::vector<std::string>>(text);
                m_text_embedding_pipeline->start_embed_async(texts, *prompt);
            } else if (is_single_text) {
                m_text_embedding_pipeline->start_embed_documents_async(std::vector<std::string>{std::get<std::string>(text)});
            } else {
                m_text_embedding_pipeline->start_embed_documents_async(std::get<std::vector<std::string>>(text));
            }
            m_text_async_pending = true;
            return;
        }
        std::lock_guard<std::mutex> async_lock(m_async_mutex);
        OPENVINO_ASSERT(!m_embed_future.valid(), "Previous asynchronous embed request is still pending");
        std::lock_guard<std::mutex> request_lock(m_request_mutex);
        m_embed_future = std::async(std::launch::async, [this, text, images, videos, videos_metadata, prompt]() {
            std::lock_guard<std::mutex> lock(m_request_mutex);

            OPENVINO_ASSERT(videos_metadata.empty() || videos_metadata.size() == videos.size(),
                            "videos_metadata size (",
                            videos_metadata.size(),
                            ") must be equal to videos size (",
                            videos.size(),
                            ") or empty");

            std::vector<EncodedImage> encoded_images = m_inputs_embedder->encode_images(images);
            std::vector<EncodedVideo> encoded_videos = m_inputs_embedder->encode_videos(videos, videos_metadata);

            std::vector<std::string> texts = std::holds_alternative<std::string>(text)
                ? std::vector<std::string>{std::get<std::string>(text)}
                : std::get<std::vector<std::string>>(text);

            return extract_multimodal_batch(texts, encoded_images, encoded_videos, prompt);
        });
    }

private:
    enum class Mode {
        MULTIMODAL,
        TEXT_ONLY,
    };

    void init_multimodal(const std::filesystem::path& models_path,
                         const std::string& device,
                         const ov::AnyMap& properties) {
        m_inputs_embedder = std::make_shared<InputsEmbedder>(models_path, device, properties);
        m_inputs_embedder->set_apply_chat_template_status(false);

        ov::AnyMap properties_copy = properties;
        utils::extract_extensions_to_core(properties_copy);
        std::shared_ptr<ov::Model> language_model =
            utils::singleton_core().read_model(models_path / "openvino_language_model.xml");
        language_model = utils::apply_postprocessing(language_model, m_config);
        m_compiled_language_model = utils::singleton_core().compile_model(language_model, device, properties_copy);
        m_language_model_request = m_compiled_language_model.create_infer_request();

        for (const auto& input : m_compiled_language_model.inputs()) {
            const auto& input_names = input.get_names();
            m_language_model_input_names.insert(input_names.begin(), input_names.end());
        }

        for (const auto& output : m_compiled_language_model.outputs()) {
            const auto& output_names = output.get_names();
            m_language_model_output_names.insert(output_names.begin(), output_names.end());
        }

        if (has_lm_output("last_hidden_state")) {
            m_embedding_output_name = "last_hidden_state";
        } else if (has_lm_output("logits")) {
            // Some embedding-style exports expose the sequence embedding under 'logits'.
            m_embedding_output_name = "logits";
        } else {
            OPENVINO_THROW("Language model must expose 'last_hidden_state' or 'logits' output for EmbeddingPipeline");
        }

        OPENVINO_ASSERT(has_lm_input("inputs_embeds"),
                        "Language model must expose 'inputs_embeds' input for EmbeddingPipeline");
    }

    ov::Tensor extract_multimodal_batch(const std::vector<std::string>& texts,
                                        const std::vector<EncodedImage>& encoded_images,
                                        const std::vector<EncodedVideo>& encoded_videos,
                                        const std::optional<std::string>& prompt) {
        // Prepare all normalized prompts and get inputs_embeds for each
        struct BatchItem {
            ov::Tensor inputs_embeds;
            std::optional<ov::Tensor> token_type_ids;
            std::optional<ov::Tensor> position_ids;
        };

        std::vector<BatchItem> batch_items;
        batch_items.reserve(texts.size());
        size_t max_seq_length = 0;

        VLMPerfMetrics metrics;
        for (const std::string& batch_text : texts) {
            const std::string formatted_text = prompt.has_value() ?
                append_visual_tags(batch_text, encoded_images.size(), encoded_videos.size()) :
                batch_text;
            const NormalizedPrompt normalized_prompt =
                m_inputs_embedder->normalize_prompt(format_prompt(formatted_text, prompt), 0, 0, encoded_images, encoded_videos);

            BatchItem item;
            if (m_inputs_embedder->has_token_type_ids()) {
                std::tie(item.inputs_embeds, item.token_type_ids) =
                    m_inputs_embedder->get_inputs_embeds_with_token_type_ids(normalized_prompt.unified_prompt,
                                                                             encoded_images,
                                                                             encoded_videos,
                                                                             metrics,
                                                                             true,
                                                                             normalized_prompt.images_sequence,
                                                                             normalized_prompt.videos_sequence);
            } else {
                item.inputs_embeds = m_inputs_embedder->get_inputs_embeds(normalized_prompt.unified_prompt,
                                                                          encoded_images,
                                                                          encoded_videos,
                                                                          metrics,
                                                                          true,
                                                                          normalized_prompt.images_sequence,
                                                                          normalized_prompt.videos_sequence);
            }

            const size_t seq_length = item.inputs_embeds.get_shape().at(1);
            max_seq_length = std::max(max_seq_length, seq_length);

            if (encoded_images.empty() && encoded_videos.empty() && has_lm_input("visual_pos_masks")) {
                item.position_ids = make_text_position_ids(seq_length);
            } else {
                std::tie(item.position_ids, std::ignore) = m_inputs_embedder->get_position_ids(seq_length, 0);
            }

            batch_items.push_back(std::move(item));
        }

        // Stack and pad all tensors to max_seq_length
        const size_t batch_size = texts.size();
        const size_t embed_dim = batch_items[0].inputs_embeds.get_shape().at(2);

        ov::Tensor batched_inputs_embeds(ov::element::f32, {batch_size, max_seq_length, embed_dim});
        ov::Tensor batched_attention_mask(ov::element::i64, {batch_size, max_seq_length});
        std::fill_n(batched_attention_mask.data<int64_t>(), batched_attention_mask.get_size(), 0);

        std::optional<ov::Tensor> batched_token_type_ids;
        if (batch_items[0].token_type_ids.has_value()) {
            batched_token_type_ids = ov::Tensor(ov::element::i64, {batch_size, max_seq_length});
            std::fill_n(batched_token_type_ids->data<int64_t>(), batched_token_type_ids->get_size(), 0);
        }

        std::optional<ov::Tensor> batched_position_ids;
        if (batch_items[0].position_ids.has_value()) {
            const auto& first_pos_shape = batch_items[0].position_ids->get_shape();
            ov::Shape batched_pos_shape = first_pos_shape;
            // For position_ids with shape [3, batch, seq] or [batch, seq], update the batch dimension
            if (batched_pos_shape.size() == 3) {
                batched_pos_shape[1] = batch_size;  // [3, 1, seq] -> [3, batch_size, seq]
            } else if (batched_pos_shape.size() == 2) {
                batched_pos_shape[0] = batch_size;  // [1, seq] -> [batch_size, seq]
            }
            batched_pos_shape.back() = max_seq_length;
            batched_position_ids = ov::Tensor(ov::element::i64, batched_pos_shape);
            std::fill_n(batched_position_ids->data<int64_t>(), batched_position_ids->get_size(), 0);
        }

        for (size_t i = 0; i < batch_size; ++i) {
            const auto& item = batch_items[i];
            const size_t seq_length = item.inputs_embeds.get_shape().at(1);

            // Copy inputs_embeds
            float* dst_embeds = batched_inputs_embeds.data<float>() + i * max_seq_length * embed_dim;
            const float* src_embeds = item.inputs_embeds.data<const float>();
            std::copy_n(src_embeds, seq_length * embed_dim, dst_embeds);

            // Set attention mask (1 for valid tokens, 0 for padding)
            int64_t* dst_mask = batched_attention_mask.data<int64_t>() + i * max_seq_length;
            std::fill_n(dst_mask, seq_length, 1);

            // Copy token_type_ids if present
            if (item.token_type_ids.has_value()) {
                int64_t* dst_tti = batched_token_type_ids->data<int64_t>() + i * max_seq_length;
                const int64_t* src_tti = item.token_type_ids->data<const int64_t>();
                std::copy_n(src_tti, seq_length, dst_tti);
            }

            // Copy position_ids if present
            if (item.position_ids.has_value()) {
                const auto& pos_shape = item.position_ids->get_shape();
                const auto& batched_pos_shape = batched_position_ids->get_shape();
                const int64_t* src_pos = item.position_ids->data<const int64_t>();

                if (batched_pos_shape.size() == 3) {
                    // Shape: [3, batch_size, max_seq_length]
                    // For each of the 3 dimensions, copy the position values for this batch item
                    const size_t num_dims = batched_pos_shape[0];
                    for (size_t dim_idx = 0; dim_idx < num_dims; ++dim_idx) {
                        int64_t* dst_pos = batched_position_ids->data<int64_t>() +
                                          dim_idx * batch_size * max_seq_length +  // offset to dimension
                                          i * max_seq_length;                        // offset to batch item
                        const int64_t* src_dim = src_pos + dim_idx * seq_length;
                        std::copy_n(src_dim, seq_length, dst_pos);
                    }
                } else {
                    // Shape: [batch_size, max_seq_length]
                    int64_t* dst_pos = batched_position_ids->data<int64_t>() + i * max_seq_length;
                    std::copy_n(src_pos, seq_length, dst_pos);
                }
            }
        }

        m_language_model_request.reset_state();
        m_language_model_request.set_tensor("inputs_embeds", batched_inputs_embeds);

        if (has_lm_input("attention_mask")) {
            m_language_model_request.set_tensor("attention_mask", batched_attention_mask);
        }

        if (batched_token_type_ids.has_value() && has_lm_input("token_type_ids")) {
            m_language_model_request.set_tensor("token_type_ids", *batched_token_type_ids);
        }

        if (batched_position_ids.has_value() && has_lm_input("position_ids")) {
            m_language_model_request.set_tensor("position_ids", *batched_position_ids);
        }

        if (has_lm_input("beam_idx")) {
            ov::Tensor beam_idx(ov::element::i32, {batch_size});
            std::fill_n(beam_idx.data<int32_t>(), batch_size, 0);
            m_language_model_request.set_tensor("beam_idx", beam_idx);
        }

        const auto& lm_extra_inputs = m_inputs_embedder->get_lm_extra_inputs();
        for (const auto& [name, tensor] : lm_extra_inputs) {
            if (has_lm_input(name) && tensor.get_size() > 0) {
                m_language_model_request.set_tensor(name, tensor);
            }
        }

        m_language_model_request.infer();

        ov::Tensor batched_output = m_language_model_request.get_tensor(m_embedding_output_name);

        // Extract individual embeddings from batched output
        const ov::Shape output_shape = batched_output.get_shape();
        const size_t output_embed_dim = output_shape.back();
        std::vector<ov::Tensor> outputs;
        outputs.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            ov::Tensor single_output(ov::element::f32, {1, output_embed_dim});
            const float* src = batched_output.data<const float>() + i * output_embed_dim;
            float* dst = single_output.data<float>();
            std::copy_n(src, output_embed_dim, dst);
            outputs.push_back(single_output);
        }

        return stack_tensors(outputs);
    }

private:
    bool has_lm_input(const std::string& input_name) const {
        return m_language_model_input_names.count(input_name) > 0;
    }

    bool has_lm_output(const std::string& output_name) const {
        return m_language_model_output_names.count(output_name) > 0;
    }

    std::string format_prompt(const std::string& text, const std::optional<std::string>& prompt) const {
        Tokenizer tokenizer = m_inputs_embedder->get_tokenizer();
        if (!prompt.has_value()) {
            return append_added_special_tokens(tokenizer, text);
        }
        ChatHistory history({{{"role", "system"}, {"content", *prompt}}, {{"role", "user"}, {"content", text}}});
        constexpr bool add_generation_prompt = false;
        return append_added_special_tokens(tokenizer, tokenizer.apply_chat_template(history, add_generation_prompt));;
    }

    std::string append_added_special_tokens(Tokenizer& tokenizer, const std::string& text) const {
        const ov::Tensor with_special_tokens = tokenizer.encode(text, ov::genai::add_special_tokens(true)).input_ids;
        const ov::Tensor without_special_tokens = tokenizer.encode(text, ov::genai::add_special_tokens(false)).input_ids;
        const ov::Shape with_shape = with_special_tokens.get_shape();
        const ov::Shape without_shape = without_special_tokens.get_shape();
        OPENVINO_ASSERT(with_shape.size() == 2 && without_shape.size() == 2,
                        "Expected rank-2 tokenized prompt tensors");
        OPENVINO_ASSERT(with_shape[0] == 1 && without_shape[0] == 1,
                        "Expected single tokenized prompt");
        OPENVINO_ASSERT(with_shape[1] >= without_shape[1],
                        "Tokenizer with special tokens returned fewer tokens than tokenizer without special tokens");

        const size_t added_tokens_num = with_shape[1] - without_shape[1];
        if (added_tokens_num == 0) {
            return text;
        }

        const int64_t* with_data = with_special_tokens.data<const int64_t>();
        const int64_t* without_data = without_special_tokens.data<const int64_t>();
        for (size_t idx = 0; idx < without_shape[1]; ++idx) {
            OPENVINO_ASSERT(with_data[idx] == without_data[idx],
                            "Tokenizer special tokens are expected only at the end of the prompt");
        }

        std::vector<int64_t> added_tokens(with_data + without_shape[1], with_data + with_shape[1]);
        return text + tokenizer.decode(added_tokens, ov::genai::skip_special_tokens(false));
    }

    Mode m_mode = Mode::MULTIMODAL;
    std::shared_ptr<InputsEmbedder> m_inputs_embedder;
    std::unique_ptr<TextEmbeddingPipeline> m_text_embedding_pipeline;
    TextEmbeddingPipeline::Config m_config;
    ov::CompiledModel m_compiled_language_model;
    ov::InferRequest m_language_model_request;
    std::mutex m_request_mutex;
    std::mutex m_async_mutex;
    std::unordered_set<std::string> m_language_model_input_names;
    std::unordered_set<std::string> m_language_model_output_names;
    std::string m_embedding_output_name;
    std::future<ov::Tensor> m_embed_future;
    bool m_text_async_pending = false;
};

EmbeddingPipeline::EmbeddingPipeline(const std::filesystem::path& models_path,
                                     const std::string& device,
                                     const TextEmbeddingPipeline::Config& config,
                                     const ov::AnyMap& properties)
    : m_impl(std::make_unique<EmbeddingPipelineImpl>(models_path, device, config, properties)) {}

EmbeddingPipeline::EmbeddingPipeline(const std::filesystem::path& models_path,
                                     const std::string& device,
                                     const ov::AnyMap& properties)
    : m_impl(std::make_unique<EmbeddingPipelineImpl>(models_path, device, properties)) {}

// void EmbeddingPipeline::start_embed_async(const EmbeddingPipeline::TextInput& text,
//                                           const std::optional<std::string>& prompt) {
//     return m_impl->start_embed_async(text, prompt);
// }

ov::Tensor EmbeddingPipeline::wait() {
    return m_impl->wait();
}

ov::Tensor EmbeddingPipeline::embed(const EmbeddingPipeline::TextInput& text,
                                    const std::vector<ov::Tensor>& images,
                                    const std::vector<ov::Tensor>& videos,
                                    const std::vector<VideoMetadata>& videos_metadata,
                                    const std::optional<std::string>& prompt) {
    return m_impl->embed(text, images, videos, videos_metadata, prompt);
}

void EmbeddingPipeline::start_embed_async(const EmbeddingPipeline::TextInput& text,
                                          const std::vector<ov::Tensor>& images,
                                          const std::vector<ov::Tensor>& videos,
                                          const std::vector<VideoMetadata>& videos_metadata,
                                          const std::optional<std::string>& prompt) {
    return m_impl->start_embed_async(text, images, videos, videos_metadata, prompt);
}

ov::Tensor EmbeddingPipeline::embed(const ov::AnyMap& properties) {
    std::vector<ov::Tensor> images_vec;
    std::vector<ov::Tensor> videos_vec;
    std::vector<VideoMetadata> videos_metadata_vec;
    std::optional<std::string> prompt_val;

    utils::read_anymap_param(properties, ov::genai::images.name(), images_vec);
    utils::read_anymap_param(properties, ov::genai::videos.name(), videos_vec);
    utils::read_anymap_param(properties, ov::genai::videos_metadata.name(), videos_metadata_vec);
    utils::read_anymap_param(properties, ov::genai::prompt.name(), prompt_val);

    return m_impl->embed(std::string{}, images_vec, videos_vec, videos_metadata_vec, prompt_val);
}

void EmbeddingPipeline::start_embed_async(const ov::AnyMap& properties) {
    std::vector<ov::Tensor> images_vec;
    std::vector<ov::Tensor> videos_vec;
    std::vector<VideoMetadata> videos_metadata_vec;
    std::optional<std::string> prompt_val;

    utils::read_anymap_param(properties, ov::genai::images.name(), images_vec);
    utils::read_anymap_param(properties, ov::genai::videos.name(), videos_vec);
    utils::read_anymap_param(properties, ov::genai::videos_metadata.name(), videos_metadata_vec);
    utils::read_anymap_param(properties, ov::genai::prompt.name(), prompt_val);

    return m_impl->start_embed_async(std::string{}, images_vec, videos_vec, videos_metadata_vec, prompt_val);
}

EmbeddingPipeline::~EmbeddingPipeline() = default;

}  // namespace genai
}  // namespace ov
