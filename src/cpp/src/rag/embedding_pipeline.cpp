// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/embedding_pipeline.hpp"

#include <cmath>
#include <future>
#include <mutex>
#include <numeric>
#include <optional>
#include <unordered_set>

#include "openvino/core/except.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "openvino/runtime/core.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"

namespace {

template <typename ElementType>
ov::Tensor last_token_pool_embeddings_impl(const ov::Tensor& tensor) {
    const ov::Shape shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "Expected embedding tensor rank 3, got rank ", shape.size());
    OPENVINO_ASSERT(shape[0] == 1, "Expected batch size 1, got ", shape[0]);
    OPENVINO_ASSERT(shape[1] > 0, "Sequence length must be greater than 0");

    const size_t sequence_length = shape[1];
    const size_t hidden_size = shape[2];
    const ElementType* embeddings = tensor.data<const ElementType>();

    ov::Tensor pooled_tensor(ov::element::f32, {1, hidden_size});
    float* pooled = pooled_tensor.data<float>();
    const size_t token_offset = (sequence_length - 1) * hidden_size;
    for (size_t hidden_idx = 0; hidden_idx < hidden_size; ++hidden_idx) {
        pooled[hidden_idx] = static_cast<float>(embeddings[token_offset + hidden_idx]);
    }

    return pooled_tensor;
}

template <typename ElementType>
ov::Tensor get_tensor_as_f32(const ov::Tensor& tensor) {
    const ov::Shape shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 2, "Expected embedding tensor rank 2, got rank ", shape.size());
    OPENVINO_ASSERT(shape[0] == 1, "Expected batch size 1, got ", shape[0]);
    OPENVINO_ASSERT(shape[1] > 0, "Embedding size must be greater than 0");

    const size_t embedding_size = shape[1];
    const ElementType* embeddings = tensor.data<const ElementType>();

    ov::Tensor result(ov::element::f32, {1, embedding_size});
    float* result_data = result.data<float>();
    for (size_t i = 0; i < embedding_size; ++i) {
        result_data[i] = static_cast<float>(embeddings[i]);
    }
    return result;
}

ov::Tensor pool_embeddings(const ov::Tensor& tensor) {
    const ov::Shape shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 || shape.size() == 3,
                    "Expected embedding tensor rank 2 or 3, got rank ",
                    shape.size());

    const auto element_type = tensor.get_element_type();

    // Rank-2 tensor is already pooled: [1, hidden_size].
    if (shape.size() == 2) {
        if (element_type == ov::element::f32) {
            return tensor;
        }
        if (element_type == ov::element::f16) {
            return get_tensor_as_f32<ov::float16>(tensor);
        }
        if (element_type == ov::element::bf16) {
            return get_tensor_as_f32<ov::bfloat16>(tensor);
        }
    }

    // Rank-3 tensor needs last-token pooling over sequence: [1, seq_len, hidden_size].
    if (element_type == ov::element::f32) {
        return last_token_pool_embeddings_impl<float>(tensor);
    }
    if (element_type == ov::element::f16) {
        return last_token_pool_embeddings_impl<ov::float16>(tensor);
    }
    if (element_type == ov::element::bf16) {
        return last_token_pool_embeddings_impl<ov::bfloat16>(tensor);
    }

    OPENVINO_THROW("Unsupported embedding element type: ", element_type);
}

ov::Tensor normalize_embeddings(const ov::Tensor& tensor) {
    const ov::Shape shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 2, "Expected embedding tensor rank 2, got rank ", shape.size());
    OPENVINO_ASSERT(tensor.get_element_type() == ov::element::f32,
                    "Expected f32 embedding tensor, got ",
                    tensor.get_element_type());

    const float* embeddings = tensor.data<const float>();
    float squared_norm = 0.0f;
    for (size_t idx = 0; idx < tensor.get_size(); ++idx) {
        squared_norm += embeddings[idx] * embeddings[idx];
    }

    ov::Tensor normalized(ov::element::f32, shape);
    float* normalized_data = normalized.data<float>();
    const float norm = std::sqrt(squared_norm);
    if (norm == 0.0f) {
        std::copy_n(embeddings, tensor.get_size(), normalized_data);
        return normalized;
    }

    for (size_t idx = 0; idx < tensor.get_size(); ++idx) {
        normalized_data[idx] = embeddings[idx] / norm;
    }
    return normalized;
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

ov::Tensor embedding_result_to_tensor(const ov::genai::EmbeddingResult& embedding_result) {
    return std::visit([](const auto& values) {
        ov::Tensor result(ov::element::f32, {1, values.size()});
        float* result_data = result.data<float>();
        for (size_t idx = 0; idx < values.size(); ++idx) {
            result_data[idx] = static_cast<float>(values[idx]);
        }
        return result;
    }, embedding_result);
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
                          const ov::AnyMap& properties) {
        try {
            init_multimodal(models_path, device, properties);
            m_mode = Mode::MULTIMODAL;
        } catch (const std::exception& multimodal_error) {
            try {
                m_text_embedding_pipeline = std::make_unique<TextEmbeddingPipeline>(models_path, device, properties);
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
            OPENVINO_ASSERT(m_async_request_type == AsyncRequestType::NONE, "Previous asynchronous embed request is still pending");
            OPENVINO_ASSERT(!prompt.has_value(), "Prompt is supported only by multimodal EmbeddingPipeline");
            std::lock_guard<std::mutex> request_lock(m_request_mutex);
            const std::vector<std::string> texts = std::holds_alternative<std::string>(text) ? std::vector<std::string>{std::get<std::string>(text)}
                                                                                          : std::get<std::vector<std::string>>(text);
            return embedding_results_to_tensor(m_text_embedding_pipeline->embed_documents(texts));
        }
        std::lock_guard<std::mutex> async_lock(m_async_mutex);
        OPENVINO_ASSERT(!m_embed_future.valid(), "Previous asynchronous embed request is still pending");
        std::lock_guard<std::mutex> request_lock(m_request_mutex);
        return extract_multimodal(text, {}, {}, {}, prompt);
    }

    void start_embed_async(const EmbeddingPipeline::TextInput& text, const std::optional<std::string>& prompt) {
        const bool is_single_text = std::holds_alternative<std::string>(text);
        if (m_mode == Mode::TEXT_ONLY) {
            std::lock_guard<std::mutex> async_lock(m_async_mutex);
            OPENVINO_ASSERT(m_async_request_type == AsyncRequestType::NONE, "Previous asynchronous embed request is still pending");
            OPENVINO_ASSERT(!prompt.has_value(), "Prompt is supported only by multimodal EmbeddingPipeline");
            std::lock_guard<std::mutex> request_lock(m_request_mutex);
            if (is_single_text) {
                const std::vector<std::string> texts{std::get<std::string>(text)};
                m_text_embedding_pipeline->start_embed_documents_async(texts);
            } else {
                m_text_embedding_pipeline->start_embed_documents_async(std::get<std::vector<std::string>>(text));
            }
            m_async_request_type = AsyncRequestType::BATCH;
            return;
        }
        std::lock_guard<std::mutex> async_lock(m_async_mutex);
        OPENVINO_ASSERT(!m_embed_future.valid(), "Previous asynchronous embed request is still pending");
        std::lock_guard<std::mutex> request_lock(m_request_mutex);
        m_embed_future = std::async(std::launch::async, [this, text, prompt]() {
            std::lock_guard<std::mutex> lock(m_request_mutex);
            return extract_multimodal(text, {}, {}, {}, prompt);
        });
        m_async_request_type = is_single_text ? AsyncRequestType::SINGLE : AsyncRequestType::BATCH;
    }

    ov::Tensor wait() {
        if (m_mode == Mode::TEXT_ONLY) {
            std::lock_guard<std::mutex> async_lock(m_async_mutex);
            OPENVINO_ASSERT(m_async_request_type != AsyncRequestType::NONE,
                            "Asynchronous embed request was not started");
            ov::Tensor result = embedding_results_to_tensor(m_text_embedding_pipeline->wait_embed_documents());
            m_async_request_type = AsyncRequestType::NONE;
            return result;
        }
        std::lock_guard<std::mutex> async_lock(m_async_mutex);
        OPENVINO_ASSERT(m_embed_future.valid(), "Asynchronous embed request was not started");
        ov::Tensor result = m_embed_future.get();
        m_async_request_type = AsyncRequestType::NONE;
        return result;
    }

    ov::Tensor embed(const EmbeddingPipeline::TextInput& text,
                     const std::vector<ov::Tensor>& images,
                     const std::vector<ov::Tensor>& videos,
                     const std::vector<VideoMetadata>& videos_metadata,
                     const std::optional<std::string>& prompt) {
        if (m_mode == Mode::TEXT_ONLY) {
            OPENVINO_ASSERT(images.empty() && videos.empty(),
                            "TextEmbeddingPipeline fallback is active and does not support image/video input");
            OPENVINO_ASSERT(videos_metadata.empty(),
                            "TextEmbeddingPipeline fallback is active and does not support video metadata input");
            OPENVINO_ASSERT(!prompt.has_value(), "Prompt is supported only by multimodal EmbeddingPipeline");
            return embed(text, prompt);
        }
        std::lock_guard<std::mutex> async_lock(m_async_mutex);
        OPENVINO_ASSERT(!m_embed_future.valid(), "Previous asynchronous embed request is still pending");
        std::lock_guard<std::mutex> request_lock(m_request_mutex);
        return extract_multimodal(text, images, videos, videos_metadata, prompt);
    }

    void start_embed_async(const EmbeddingPipeline::TextInput& text,
                           const std::vector<ov::Tensor>& images,
                           const std::vector<ov::Tensor>& videos,
                           const std::vector<VideoMetadata>& videos_metadata,
                           const std::optional<std::string>& prompt) {
        const bool is_single_text = std::holds_alternative<std::string>(text);
        if (m_mode == Mode::TEXT_ONLY) {
            OPENVINO_ASSERT(images.empty() && videos.empty(),
                            "TextEmbeddingPipeline fallback is active and does not support image/video input");
            OPENVINO_ASSERT(videos_metadata.empty(),
                            "TextEmbeddingPipeline fallback is active and does not support video metadata input");
            start_embed_async(text, prompt);
            return;
        }
        std::lock_guard<std::mutex> async_lock(m_async_mutex);
        OPENVINO_ASSERT(!m_embed_future.valid(), "Previous asynchronous embed request is still pending");
        std::lock_guard<std::mutex> request_lock(m_request_mutex);
        m_embed_future = std::async(std::launch::async, [this, text, images, videos, videos_metadata, prompt]() {
            std::lock_guard<std::mutex> lock(m_request_mutex);
            return extract_multimodal(text, images, videos, videos_metadata, prompt);
        });
        m_async_request_type = is_single_text ? AsyncRequestType::SINGLE : AsyncRequestType::BATCH;
    }

private:
    enum class Mode {
        MULTIMODAL,
        TEXT_ONLY,
    };

    enum class AsyncRequestType {
        NONE,
        SINGLE,
        BATCH,
    };

    void init_multimodal(const std::filesystem::path& models_path,
                         const std::string& device,
                         const ov::AnyMap& properties) {
        m_inputs_embedder = std::make_shared<InputsEmbedder>(models_path, device, properties);
        m_inputs_embedder->set_apply_chat_template_status(false);
        m_inputs_embedder->set_add_special_tokens(false);

        ov::Core core;
        std::shared_ptr<ov::Model> language_model =
            core.read_model(models_path / "openvino_language_model.xml");
        m_compiled_language_model = core.compile_model(language_model, device, properties);
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

    ov::Tensor extract_multimodal(const EmbeddingPipeline::TextInput& text,
                                  const std::vector<ov::Tensor>& images,
                                  const std::vector<ov::Tensor>& videos,
                                  const std::vector<VideoMetadata>& videos_metadata,
                                  const std::optional<std::string>& prompt) {
        if (const auto* single_text = std::get_if<std::string>(&text)) {
            return extract_multimodal(*single_text, images, videos, videos_metadata, prompt);
        }

        const std::vector<std::string>& texts = std::get<std::vector<std::string>>(text);
        std::vector<ov::Tensor> out;
        out.reserve(texts.size());
        if (texts.empty()) {
            return stack_tensors(out);
        }

        OPENVINO_ASSERT(videos_metadata.empty() || videos_metadata.size() == videos.size(),
                        "videos_metadata size (",
                        videos_metadata.size(),
                        ") must be equal to videos size (",
                        videos.size(),
                        ") or empty");

        std::vector<EncodedImage> encoded_images = m_inputs_embedder->encode_images(images);
        std::vector<EncodedVideo> encoded_videos = m_inputs_embedder->encode_videos(videos, videos_metadata);
        for (const std::string& batch_text : texts) {
            out.push_back(extract_multimodal(batch_text, encoded_images, encoded_videos, prompt));
        }
        return stack_tensors(out);
    }

    ov::Tensor extract_multimodal(const std::string& text,
                                  const std::vector<ov::Tensor>& images,
                                  const std::vector<ov::Tensor>& videos,
                                  const std::vector<VideoMetadata>& videos_metadata,
                                  const std::optional<std::string>& prompt) {
        OPENVINO_ASSERT(videos_metadata.empty() || videos_metadata.size() == videos.size(),
                        "videos_metadata size (",
                        videos_metadata.size(),
                        ") must be equal to videos size (",
                        videos.size(),
                        ") or empty");

        std::vector<EncodedImage> encoded_images = m_inputs_embedder->encode_images(images);
        std::vector<EncodedVideo> encoded_videos = m_inputs_embedder->encode_videos(videos, videos_metadata);

        return extract_multimodal(text, encoded_images, encoded_videos, prompt);
    }

    ov::Tensor extract_multimodal(const std::string& text,
                                  const std::vector<EncodedImage>& encoded_images,
                                  const std::vector<EncodedVideo>& encoded_videos,
                                  const std::optional<std::string>& prompt) {
        const std::string formatted_text = prompt.has_value() ?
            append_visual_tags(text, encoded_images.size(), encoded_videos.size()) :
            text;
        const NormalizedPrompt normalized_prompt =
            m_inputs_embedder->normalize_prompt(format_prompt(formatted_text, prompt), 0, 0, encoded_images, encoded_videos);
        m_inputs_embedder->set_add_special_tokens(!prompt.has_value());

        Tokenizer tokenizer = m_inputs_embedder->get_tokenizer();
        const ov::Tensor input_ids = tokenizer.encode(normalized_prompt.unified_prompt,
                                                      ov::genai::add_special_tokens(false)).input_ids;

        VLMPerfMetrics metrics;
        ov::Tensor inputs_embeds;
        std::optional<ov::Tensor> token_type_ids;

        if (m_inputs_embedder->has_token_type_ids()) {
            std::tie(inputs_embeds, token_type_ids) =
                m_inputs_embedder->get_inputs_embeds_with_token_type_ids(normalized_prompt.unified_prompt,
                                                                         encoded_images,
                                                                         encoded_videos,
                                                                         metrics,
                                                                         true,
                                                                         normalized_prompt.images_sequence,
                                                                         normalized_prompt.videos_sequence);
        } else {
            inputs_embeds = m_inputs_embedder->get_inputs_embeds(normalized_prompt.unified_prompt,
                                                                 encoded_images,
                                                                 encoded_videos,
                                                                 metrics,
                                                                 true,
                                                                 normalized_prompt.images_sequence,
                                                                 normalized_prompt.videos_sequence);
        }

        m_language_model_request.reset_state();

        m_language_model_request.set_tensor("inputs_embeds", inputs_embeds);

        const size_t input_sequence_length = inputs_embeds.get_shape().at(1);
        if (has_lm_input("attention_mask")) {
            ov::Tensor attention_mask(ov::element::i64, {1, input_sequence_length});
            std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);
            m_language_model_request.set_tensor("attention_mask", attention_mask);
        }

        if (token_type_ids.has_value() && has_lm_input("token_type_ids")) {
            m_language_model_request.set_tensor("token_type_ids", *token_type_ids);
        }

        std::optional<ov::Tensor> position_ids;
        std::optional<int64_t> rope_delta;
        if (encoded_images.empty() && encoded_videos.empty() && has_lm_input("visual_pos_masks")) {
            position_ids = make_text_position_ids(input_sequence_length);
        } else {
            std::tie(position_ids, rope_delta) = m_inputs_embedder->get_position_ids(input_sequence_length, 0);
        }
        if (position_ids.has_value() && has_lm_input("position_ids")) {
            m_language_model_request.set_tensor("position_ids", *position_ids);
        }

        if (has_lm_input("beam_idx")) {
            ov::Tensor beam_idx(ov::element::i32, {1});
            beam_idx.data<int32_t>()[0] = 0;
            m_language_model_request.set_tensor("beam_idx", beam_idx);
        }

        const auto& lm_extra_inputs = m_inputs_embedder->get_lm_extra_inputs();
        for (const auto& [name, tensor] : lm_extra_inputs) {
            if (has_lm_input(name) && tensor.get_size() > 0) {
                m_language_model_request.set_tensor(name, tensor);
            }
        }

        m_language_model_request.infer();

        const ov::Tensor hidden_or_logits = m_language_model_request.get_tensor(m_embedding_output_name);
        if (encoded_images.empty() && encoded_videos.empty()) {
            return pool_embeddings(hidden_or_logits);
        }
        return normalize_embeddings(pool_embeddings(hidden_or_logits));
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
    ov::CompiledModel m_compiled_language_model;
    ov::InferRequest m_language_model_request;
    std::mutex m_request_mutex;
    std::mutex m_async_mutex;
    std::unordered_set<std::string> m_language_model_input_names;
    std::unordered_set<std::string> m_language_model_output_names;
    std::string m_embedding_output_name;
    std::future<ov::Tensor> m_embed_future;
    AsyncRequestType m_async_request_type = AsyncRequestType::NONE;
};

EmbeddingPipeline::EmbeddingPipeline(const std::filesystem::path& models_path,
                                     const std::string& device,
                                     const ov::AnyMap& properties)
    : m_impl(std::make_unique<EmbeddingPipelineImpl>(models_path, device, properties)) {}

ov::Tensor EmbeddingPipeline::embed(const EmbeddingPipeline::TextInput& text, const std::optional<std::string>& prompt) {
    return m_impl->embed(text, prompt);
}

void EmbeddingPipeline::start_embed_async(const EmbeddingPipeline::TextInput& text,
                                          const std::optional<std::string>& prompt) {
    return m_impl->start_embed_async(text, prompt);
}

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

EmbeddingPipeline::~EmbeddingPipeline() = default;

}  // namespace genai
}  // namespace ov
