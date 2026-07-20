// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/embedding_pipeline.hpp"

#include <algorithm>
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
    if (properties.count(ov::genai::text_embedding_config.name())) {
        return properties.at(ov::genai::text_embedding_config.name())
            .as<ov::genai::TextEmbeddingPipeline::Config>();
    }

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

    EmbedResult embed(const StringInputs& text, const ov::AnyMap& properties) {
        std::optional<std::string> prompt;
        utils::read_anymap_param(properties, ov::genai::embedding_prompt.name(), prompt);
        if (m_mode == Mode::TEXT_ONLY) {
            if (std::holds_alternative<std::string>(text)) {
                const std::vector<std::string> texts{std::get<std::string>(text)};
                if (prompt.has_value()) {
                    return EmbedResult{embedding_results_to_tensor(m_text_embedding_pipeline->embed(texts, *prompt))};
                }
                return EmbedResult{embedding_results_to_tensor(m_text_embedding_pipeline->embed_documents(texts))};
            } else {
                const std::vector<std::string>& texts = std::get<std::vector<std::string>>(text);
                if (prompt.has_value()) {
                    return EmbedResult{embedding_results_to_tensor(m_text_embedding_pipeline->embed(texts, *prompt))};
                }
                return EmbedResult{embedding_results_to_tensor(m_text_embedding_pipeline->embed_documents(texts))};
            }
        }
        std::vector<std::string> texts = std::holds_alternative<std::string>(text)
            ? std::vector<std::string>{std::get<std::string>(text)}
            : std::get<std::vector<std::string>>(text);

        std::vector<EncodedImage> encoded_images;
        std::vector<EncodedVideo> encoded_videos;
        return multimodal_embed(texts, encoded_images, encoded_videos, prompt);
    }

    EmbedResult embed(const StringInputs& text,
                     const std::vector<ov::Tensor>& images,
                     const std::vector<ov::Tensor>& videos,
                     const std::vector<VideoMetadata>& videos_metadata,
                     const ov::AnyMap& properties) {
        if (m_mode == Mode::TEXT_ONLY) {
            OPENVINO_ASSERT(images.empty() && videos.empty() && videos_metadata.empty(),
                            "TextEmbeddingPipeline fallback is active and does not support image/video input");
            return embed(text, properties);
        }
        std::optional<std::string> prompt;
        utils::read_anymap_param(properties, ov::genai::embedding_prompt.name(), prompt);

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

        return multimodal_embed(texts, encoded_images, encoded_videos, prompt);
    }

private:
    enum class Mode {
        MULTIMODAL,
        TEXT_ONLY,
    };

    void init_multimodal(const std::filesystem::path& models_path,
                         const std::string& device,
                         const ov::AnyMap& properties) {
        ov::AnyMap properties_copy = properties;
        utils::extract_extensions_to_core(properties_copy);

        m_inputs_embedder = std::make_shared<InputsEmbedder>(models_path, device, properties_copy);
        m_inputs_embedder->set_apply_chat_template_status(false);
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

        OPENVINO_ASSERT(has_lm_output("logits"),
                        "Language model must expose 'logits' output for EmbeddingPipeline");
        m_embedding_output_name = "logits";

        OPENVINO_ASSERT(has_lm_input("inputs_embeds"),
                        "Language model must expose 'inputs_embeds' input for EmbeddingPipeline");
    }

    EmbedResult multimodal_embed(const std::vector<std::string>& texts,
                                        const std::vector<EncodedImage>& encoded_images,
                                        const std::vector<EncodedVideo>& encoded_videos,
                                        const std::optional<std::string>& prompt) {
        if (texts.empty()) {
            return EmbedResult{ov::Tensor(ov::element::f32, {0, 0})};
        }
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
            const std::string formatted_text =
                append_visual_tags(batch_text, encoded_images.size(), encoded_videos.size());
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
        OPENVINO_ASSERT(batch_size > 0, "embed() called with an empty batch");

        const size_t embed_dim = batch_items[0].inputs_embeds.get_shape().at(2);

        ov::Tensor batched_inputs_embeds(ov::element::f32, {batch_size, max_seq_length, embed_dim});
        std::fill_n(batched_inputs_embeds.data<float>(), batched_inputs_embeds.get_size(), 0.0f);
        ov::Tensor batched_attention_mask(ov::element::i64, {batch_size, max_seq_length});
        std::fill_n(batched_attention_mask.data<int64_t>(), batched_attention_mask.get_size(), 0);

        std::optional<ov::Tensor> batched_token_type_ids;
        if (batch_items[0].token_type_ids.has_value()) {
            batched_token_type_ids = ov::Tensor(ov::element::i64, {batch_size, max_seq_length});
            std::fill_n(batched_token_type_ids->data<int64_t>(), batched_token_type_ids->get_size(), 0);
        }

        std::optional<ov::Tensor> batched_position_ids;
        if (batch_items[0].position_ids.has_value() && has_lm_input("position_ids")) {
            const auto& first_pos_shape = batch_items[0].position_ids->get_shape();
            OPENVINO_ASSERT(first_pos_shape.size() == 3,
                            "Expected position_ids to have rank 3, got rank ", first_pos_shape.size());
            ov::Shape batched_pos_shape = first_pos_shape;
            batched_pos_shape[1] = batch_size;  // [3, 1, seq] -> [3, batch_size, seq]
            batched_pos_shape.back() = max_seq_length;
            batched_position_ids = ov::Tensor(ov::element::i64, batched_pos_shape);
            std::fill_n(batched_position_ids->data<int64_t>(), batched_position_ids->get_size(), 0);
        }

        for (size_t i = 0; i < batch_size; ++i) {
            const auto& item = batch_items[i];
            const size_t seq_length = item.inputs_embeds.get_shape().at(1);

            OPENVINO_ASSERT(item.inputs_embeds.get_element_type() == ov::element::f32,
                             "inputs_embeds must be float32 for EmbeddingPipeline batching, got ",
                             item.inputs_embeds.get_element_type());
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

            if (batched_position_ids.has_value() && item.position_ids.has_value()) {
                const size_t num_dims = batched_position_ids->get_shape()[0];
                const int64_t* src_pos = item.position_ids->data<const int64_t>();
                for (size_t dim_idx = 0; dim_idx < num_dims; ++dim_idx) {
                    int64_t* dst_pos = batched_position_ids->data<int64_t>() +
                                      dim_idx * batch_size * max_seq_length +
                                      i * max_seq_length;
                    std::copy_n(src_pos + dim_idx * seq_length, seq_length, dst_pos);
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
        OPENVINO_ASSERT(output_shape.size() == 2,
                        "Expected rank-2 language model output after pooling (shape [batch, embed_dim]), got rank ",
                        output_shape.size(),
                        ". Verify that apply_postprocessing reduces the sequence dimension.");
        OPENVINO_ASSERT(output_shape[0] == batch_size,
                        "Language model output batch dimension mismatch: expected ",
                        batch_size,
                        ", got ",
                        output_shape[0]);
        OPENVINO_ASSERT(batched_output.get_element_type() == ov::element::f32,
                        "Language model output must be float32, got ",
                        batched_output.get_element_type(),
                        ". Export the model with float32 output precision.");
        const size_t output_embed_dim = output_shape[1];
        ov::Tensor output_copy(ov::element::f32, output_shape);
        std::copy_n(batched_output.data<const float>(), batch_size * output_embed_dim, output_copy.data<float>());
        return EmbedResult{std::move(output_copy)};
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
        
        if (tokenizer.get_chat_template().empty()) {
             return append_added_special_tokens(tokenizer, *prompt + text);
        }

        ChatHistory history({{{"role", "system"}, {"content", *prompt}}, {{"role", "user"}, {"content", text}}});
        constexpr bool add_generation_prompt = false;
        return append_added_special_tokens(tokenizer, tokenizer.apply_chat_template(history, add_generation_prompt));
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
    std::unordered_set<std::string> m_language_model_input_names;
    std::unordered_set<std::string> m_language_model_output_names;
    std::string m_embedding_output_name;
};

EmbeddingPipeline::EmbeddingPipeline(const std::filesystem::path& models_path,
                                     const std::string& device,
                                     const ov::AnyMap& properties)
    : m_impl(std::make_unique<EmbeddingPipelineImpl>(models_path, device, properties)) {}

EmbedResult EmbeddingPipeline::embed(const StringInputs& text,
                                    const std::vector<ov::Tensor>& images,
                                    const std::vector<ov::Tensor>& videos,
                                    const std::vector<VideoMetadata>& videos_metadata,
                                    const ov::AnyMap& properties) {
    return m_impl->embed(text, images, videos, videos_metadata, properties);
}

EmbedResult EmbeddingPipeline::embed(const ov::AnyMap& properties) {
    std::vector<ov::Tensor> images_vec;
    std::vector<ov::Tensor> videos_vec;
    std::vector<VideoMetadata> videos_metadata_vec;
    std::variant<std::string, std::vector<std::string>> text_variant{std::string{}};

    utils::read_anymap_param(properties, ov::genai::images.name(), images_vec);
    utils::read_anymap_param(properties, ov::genai::videos.name(), videos_vec);
    utils::read_anymap_param(properties, ov::genai::videos_metadata.name(), videos_metadata_vec);
    
    const auto text_it = properties.find(ov::genai::text.name());
    if (text_it != properties.end() && !text_it->second.empty()) {
        if (text_it->second.is<std::string>()) {
            text_variant = text_it->second.as<std::string>();
        } else if (text_it->second.is<std::vector<std::string>>()) {
            text_variant = text_it->second.as<std::vector<std::string>>();
        } else if (text_it->second.is<std::variant<std::string, std::vector<std::string>>>()) {
            text_variant = text_it->second.as<std::variant<std::string, std::vector<std::string>>>();
        } else {
            OPENVINO_THROW("Unsupported type for 'text' property. Expected std::string or std::vector<std::string>.");
        }
    }

    StringInputs text_input = text_variant;

    return m_impl->embed(text_input, images_vec, videos_vec, videos_metadata_vec, properties);
}

EmbeddingPipeline::~EmbeddingPipeline() = default;

}  // namespace genai
}  // namespace ov
