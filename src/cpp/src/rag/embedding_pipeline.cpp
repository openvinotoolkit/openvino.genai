// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/embedding_pipeline.hpp"

#include <future>
#include <optional>
#include <unordered_set>

#include "openvino/core/except.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "openvino/runtime/core.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"

namespace {

template <typename ElementType>
std::vector<float> mean_pool_embeddings_impl(const ov::Tensor& tensor) {
    const ov::Shape shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "Expected embedding tensor rank 3, got rank ", shape.size());
    OPENVINO_ASSERT(shape[0] == 1, "Expected batch size 1, got ", shape[0]);
    OPENVINO_ASSERT(shape[1] > 0, "Sequence length must be greater than 0");

    const size_t sequence_length = shape[1];
    const size_t hidden_size = shape[2];
    const ElementType* embeddings = tensor.data<const ElementType>();

    std::vector<float> pooled(hidden_size, 0.0f);
    for (size_t token_idx = 0; token_idx < sequence_length; ++token_idx) {
        const size_t token_offset = token_idx * hidden_size;
        for (size_t hidden_idx = 0; hidden_idx < hidden_size; ++hidden_idx) {
            pooled[hidden_idx] += static_cast<float>(embeddings[token_offset + hidden_idx]);
        }
    }

    const float inv_sequence = 1.0f / static_cast<float>(sequence_length);
    for (float& value : pooled) {
        value *= inv_sequence;
    }

    return pooled;
}

template <typename ElementType>
std::vector<float> get_vec_from_tensor(const ov::Tensor& tensor) {
    const ov::Shape shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 2, "Expected embedding tensor rank 2, got rank ", shape.size());
    OPENVINO_ASSERT(shape[0] == 1, "Expected batch size 1, got ", shape[0]);
    OPENVINO_ASSERT(shape[1] > 0, "Embedding size must be greater than 0");

    const size_t embedding_size = shape[1];
    const ElementType* embeddings = tensor.data<const ElementType>();

    std::vector<float> vec(embedding_size);
    for (size_t i = 0; i < embedding_size; ++i) {
        vec[i] = static_cast<float>(embeddings[i]);
    }
    return vec;
}

std::vector<float> mean_pool_embeddings(const ov::Tensor& tensor) {
    const ov::Shape shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 || shape.size() == 3,
                    "Expected embedding tensor rank 2 or 3, got rank ",
                    shape.size());

    const auto element_type = tensor.get_element_type();

    // Rank-2 tensor is already pooled: [1, hidden_size].
    if (shape.size() == 2) {
        if (element_type == ov::element::f32) {
            return get_vec_from_tensor<float>(tensor);
        }
        if (element_type == ov::element::f16) {
            return get_vec_from_tensor<ov::float16>(tensor);
        }
        if (element_type == ov::element::bf16) {
            return get_vec_from_tensor<ov::bfloat16>(tensor);
        }
    }

    // Rank-3 tensor needs mean pooling over sequence: [1, seq_len, hidden_size].
    if (element_type == ov::element::f32) {
        return mean_pool_embeddings_impl<float>(tensor);
    }
    if (element_type == ov::element::f16) {
        return mean_pool_embeddings_impl<ov::float16>(tensor);
    }
    if (element_type == ov::element::bf16) {
        return mean_pool_embeddings_impl<ov::bfloat16>(tensor);
    }

    OPENVINO_THROW("Unsupported embedding element type: ", element_type);
}

std::vector<float> embedding_result_to_float(const ov::genai::EmbeddingResult& embedding_result) {
    if (const auto* floats = std::get_if<std::vector<float>>(&embedding_result)) {
        return *floats;
    }
    if (const auto* int8_values = std::get_if<std::vector<int8_t>>(&embedding_result)) {
        std::vector<float> result;
        result.reserve(int8_values->size());
        for (const int8_t value : *int8_values) {
            result.push_back(static_cast<float>(value));
        }
        return result;
    }
    const auto* uint8_values = std::get_if<std::vector<uint8_t>>(&embedding_result);
    OPENVINO_ASSERT(uint8_values != nullptr, "Unexpected embedding result type");

    std::vector<float> result;
    result.reserve(uint8_values->size());
    for (const uint8_t value : *uint8_values) {
        result.push_back(static_cast<float>(value));
    }
    return result;
}

std::vector<float> embedding_results_to_float_single(const ov::genai::EmbeddingResults& embedding_results) {
    if (const auto* float_vectors = std::get_if<std::vector<std::vector<float>>>(&embedding_results)) {
        OPENVINO_ASSERT(float_vectors->size() == 1,
                        "Expected single embedding result, got ",
                        float_vectors->size());
        return float_vectors->at(0);
    }
    if (const auto* int8_vectors = std::get_if<std::vector<std::vector<int8_t>>>(&embedding_results)) {
        OPENVINO_ASSERT(int8_vectors->size() == 1,
                        "Expected single embedding result, got ",
                        int8_vectors->size());
        std::vector<float> result;
        result.reserve(int8_vectors->at(0).size());
        for (const int8_t value : int8_vectors->at(0)) {
            result.push_back(static_cast<float>(value));
        }
        return result;
    }

    const auto* uint8_vectors = std::get_if<std::vector<std::vector<uint8_t>>>(&embedding_results);
    OPENVINO_ASSERT(uint8_vectors != nullptr, "Unexpected embedding results type");
    OPENVINO_ASSERT(uint8_vectors->size() == 1,
                    "Expected single embedding result, got ",
                    uint8_vectors->size());
    std::vector<float> result;
    result.reserve(uint8_vectors->at(0).size());
    for (const uint8_t value : uint8_vectors->at(0)) {
        result.push_back(static_cast<float>(value));
    }
    return result;
}

std::vector<std::vector<float>> embedding_results_to_float_vectors(const ov::genai::EmbeddingResults& embedding_results) {
    if (const auto* float_vectors = std::get_if<std::vector<std::vector<float>>>(&embedding_results)) {
        return *float_vectors;
    }
    if (const auto* int8_vectors = std::get_if<std::vector<std::vector<int8_t>>>(&embedding_results)) {
        std::vector<std::vector<float>> out;
        out.reserve(int8_vectors->size());
        for (const auto& row : *int8_vectors) {
            std::vector<float> converted;
            converted.reserve(row.size());
            for (const int8_t value : row) {
                converted.push_back(static_cast<float>(value));
            }
            out.push_back(std::move(converted));
        }
        return out;
    }

    const auto* uint8_vectors = std::get_if<std::vector<std::vector<uint8_t>>>(&embedding_results);
    OPENVINO_ASSERT(uint8_vectors != nullptr, "Unexpected embedding results type");
    std::vector<std::vector<float>> out;
    out.reserve(uint8_vectors->size());
    for (const auto& row : *uint8_vectors) {
        std::vector<float> converted;
        converted.reserve(row.size());
        for (const uint8_t value : row) {
            converted.push_back(static_cast<float>(value));
        }
        out.push_back(std::move(converted));
    }
    return out;
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

    std::vector<float> embed(const std::string& text) {
        if (m_mode == Mode::TEXT_ONLY) {
            return embedding_result_to_float(m_text_embedding_pipeline->embed_query(text));
        }
        return extract_multimodal(text, {}, {}, {});
    }

    std::vector<float> embed_document(const std::string& text) {
        if (m_mode == Mode::TEXT_ONLY) {
            return embedding_results_to_float_single(m_text_embedding_pipeline->embed_documents({text}));
        }
        return extract_multimodal(text, {}, {}, {});
    }

    std::vector<std::vector<float>> embed_documents(const std::vector<std::string>& texts) {
        start_embed_documents_async(texts);
        return wait_embed_documents();
    }

    void start_embed_documents_async(const std::vector<std::string>& texts) {
        if (m_mode == Mode::TEXT_ONLY) {
            m_text_embedding_pipeline->start_embed_documents_async(texts);
            return;
        }
        OPENVINO_ASSERT(!m_embed_documents_future.valid(), "Previous asynchronous embed_documents request is still pending");
        m_embed_documents_future = std::async(std::launch::async, [this, texts]() {
            std::vector<std::vector<float>> out;
            out.reserve(texts.size());
            for (const auto& text : texts) {
                out.push_back(extract_multimodal(text, {}, {}, {}));
            }
            return out;
        });
    }

    std::vector<std::vector<float>> wait_embed_documents() {
        if (m_mode == Mode::TEXT_ONLY) {
            return embedding_results_to_float_vectors(m_text_embedding_pipeline->wait_embed_documents());
        }
        OPENVINO_ASSERT(m_embed_documents_future.valid(), "Asynchronous embed_documents request was not started");
        return m_embed_documents_future.get();
    }

    void start_embed_async(const std::string& text) {
        if (m_mode == Mode::TEXT_ONLY) {
            m_text_embedding_pipeline->start_embed_query_async(text);
            return;
        }
        OPENVINO_ASSERT(!m_embed_future.valid(), "Previous asynchronous embed request is still pending");
        m_embed_future = std::async(std::launch::async, [this, text]() {
            return extract_multimodal(text, {}, {}, {});
        });
    }

    std::vector<float> wait_embed() {
        if (m_mode == Mode::TEXT_ONLY) {
            return embedding_result_to_float(m_text_embedding_pipeline->wait_embed_query());
        }
        OPENVINO_ASSERT(m_embed_future.valid(), "Asynchronous embed request was not started");
        return m_embed_future.get();
    }

    std::vector<float> embed(const std::string& text, const std::vector<ov::Tensor>& images) {
        if (m_mode == Mode::TEXT_ONLY) {
            OPENVINO_ASSERT(images.empty(),
                            "TextEmbeddingPipeline fallback is active and does not support image input");
            return embedding_result_to_float(m_text_embedding_pipeline->embed_query(text));
        }
        return extract_multimodal(text, images, {}, {});
    }

    std::vector<float> embed(const std::string& text,
                             const std::vector<ov::Tensor>& images,
                             const std::vector<ov::Tensor>& videos,
                             const std::vector<VideoMetadata>& videos_metadata) {
        if (m_mode == Mode::TEXT_ONLY) {
            OPENVINO_ASSERT(images.empty() && videos.empty(),
                            "TextEmbeddingPipeline fallback is active and does not support image/video input");
            OPENVINO_ASSERT(videos_metadata.empty(),
                            "TextEmbeddingPipeline fallback is active and does not support video metadata input");
            return embedding_result_to_float(m_text_embedding_pipeline->embed_query(text));
        }
        return extract_multimodal(text, images, videos, videos_metadata);
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

    std::vector<float> extract_multimodal(const std::string& text,
                                          const std::vector<ov::Tensor>& images,
                                          const std::vector<ov::Tensor>& videos,
                                          const std::vector<VideoMetadata>& videos_metadata) {
        OPENVINO_ASSERT(videos_metadata.empty() || videos_metadata.size() == videos.size(),
                        "videos_metadata size (",
                        videos_metadata.size(),
                        ") must be equal to videos size (",
                        videos.size(),
                        ") or empty");

        std::vector<EncodedImage> encoded_images = m_inputs_embedder->encode_images(images);
        std::vector<EncodedVideo> encoded_videos = m_inputs_embedder->encode_videos(videos, videos_metadata);

        const NormalizedPrompt normalized_prompt =
            m_inputs_embedder->normalize_prompt(text, 0, 0, encoded_images, encoded_videos);

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
        std::tie(position_ids, rope_delta) = m_inputs_embedder->get_position_ids(input_sequence_length, 0);
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
        return mean_pool_embeddings(hidden_or_logits);
    }

private:
    bool has_lm_input(const std::string& input_name) const {
        return m_language_model_input_names.count(input_name) > 0;
    }

    bool has_lm_output(const std::string& output_name) const {
        return m_language_model_output_names.count(output_name) > 0;
    }

    Mode m_mode = Mode::MULTIMODAL;
    std::shared_ptr<InputsEmbedder> m_inputs_embedder;
    std::unique_ptr<TextEmbeddingPipeline> m_text_embedding_pipeline;
    ov::CompiledModel m_compiled_language_model;
    ov::InferRequest m_language_model_request;
    std::unordered_set<std::string> m_language_model_input_names;
    std::unordered_set<std::string> m_language_model_output_names;
    std::string m_embedding_output_name;
    std::future<std::vector<float>> m_embed_future;
    std::future<std::vector<std::vector<float>>> m_embed_documents_future;
};

EmbeddingPipeline::EmbeddingPipeline(const std::filesystem::path& models_path,
                                     const std::string& device,
                                     const ov::AnyMap& properties)
    : m_impl(std::make_unique<EmbeddingPipelineImpl>(models_path, device, properties)) {}

std::vector<float> EmbeddingPipeline::embed(const std::string& text) {
    return m_impl->embed(text);
}

std::vector<float> EmbeddingPipeline::embed_document(const std::string& text) {
    return m_impl->embed_document(text);
}

std::vector<std::vector<float>> EmbeddingPipeline::embed_documents(const std::vector<std::string>& texts) {
    return m_impl->embed_documents(texts);
}

void EmbeddingPipeline::start_embed_documents_async(const std::vector<std::string>& texts) {
    return m_impl->start_embed_documents_async(texts);
}

std::vector<std::vector<float>> EmbeddingPipeline::wait_embed_documents() {
    return m_impl->wait_embed_documents();
}

void EmbeddingPipeline::start_embed_async(const std::string& text) {
    return m_impl->start_embed_async(text);
}

std::vector<float> EmbeddingPipeline::wait_embed() {
    return m_impl->wait_embed();
}

std::vector<float> EmbeddingPipeline::embed(const std::string& text, const std::vector<ov::Tensor>& images) {
    return m_impl->embed(text, images);
}

std::vector<float> EmbeddingPipeline::embed(const std::string& text,
                                            const std::vector<ov::Tensor>& images,
                                            const std::vector<ov::Tensor>& videos,
                                            const std::vector<VideoMetadata>& videos_metadata) {
    return m_impl->embed(text, images, videos, videos_metadata);
}

EmbeddingPipeline::~EmbeddingPipeline() = default;

}  // namespace genai
}  // namespace ov
