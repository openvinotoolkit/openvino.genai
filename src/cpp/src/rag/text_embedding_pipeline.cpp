// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/text_embedding_pipeline.hpp"

#include <fstream>
#include <utility>

#include <nlohmann/json.hpp>

#include "json_utils.hpp"
#include "logger.hpp"
#include "npu/text_embedding_pipeline.hpp"
#include "openvino/core/except.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "text_embedding_utils.hpp"
#include "utils.hpp"

namespace {
using namespace ov::genai;
using namespace ov;

std::optional<size_t> read_max_position_embeddings(const std::filesystem::path& models_path) {
    // config.json not found. Skip parameters initialization from file, use defaults.
    const std::filesystem::path& json_path = models_path / "config.json";
    if (!std::filesystem::exists(json_path)) {
        return std::nullopt;
    }

    using ov::genai::utils::read_json_param;

    std::ifstream f(json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '", json_path);

    nlohmann::json data = nlohmann::json::parse(f);

    std::optional<size_t> max_position_embeddings;
    read_json_param(data, "max_position_embeddings", max_position_embeddings);
    return max_position_embeddings;
}

template <typename Ports>
bool has_port(const Ports& ports, const std::string& name) {
    return std::any_of(ports.begin(), ports.end(), [&](const auto& port) {
        return port.get_names().count(name) > 0;
    });
}

ov::Tensor ones_mask(size_t batch, size_t seq_len) {
    ov::Tensor mask(ov::element::i64, Shape{batch, seq_len});
    std::fill_n(mask.data<int64_t>(), batch * seq_len, static_cast<int64_t>(1));
    return mask;
}

ov::Tensor position_ids_from_mask(const ov::Tensor& attention_mask, size_t chunk_len) {
    const ov::Shape& mask_shape = attention_mask.get_shape();
    const size_t batch = mask_shape[0];
    const size_t seq_len = mask_shape[1];
    OPENVINO_ASSERT(chunk_len <= seq_len, "attention_mask must cover the tokens passed to the model");
    const int64_t* mask = attention_mask.data<int64_t>();

    ov::Tensor position_ids(ov::element::i64, Shape{batch, chunk_len});
    int64_t* data = position_ids.data<int64_t>();
    const size_t cached = seq_len - chunk_len;

    for (size_t b = 0; b < batch; b++) {
        int64_t position = 0;
        for (size_t i = 0; i < seq_len; i++) {
            if (i >= cached) {
                data[b * chunk_len + i - cached] = position;
            }
            position += mask[b * seq_len + i];
        }
    }
    return position_ids;
}

}  // namespace

namespace ov {
namespace genai {
using utils::read_anymap_param;

TextEmbeddingPipeline::Config::Config(const ov::AnyMap& properties) {
    read_anymap_param(properties, ov::genai::max_length.name(), max_length);
    read_anymap_param(properties, ov::genai::pad_to_max_length.name(), pad_to_max_length);
    read_anymap_param(properties, ov::genai::batch_size.name(), batch_size);
    read_anymap_param(properties, ov::genai::pooling_type.name(), pooling_type);
    read_anymap_param(properties, ov::genai::normalize.name(), normalize);
    read_anymap_param(properties, ov::genai::embed_instruction.name(), embed_instruction);
    read_anymap_param(properties, ov::genai::query_instruction.name(), query_instruction);
    read_anymap_param(properties, ov::genai::padding_side.name(), padding_side);
};

void TextEmbeddingPipeline::Config::validate() const {
    if (max_length.has_value()) {
        OPENVINO_ASSERT(max_length.value() > 0, "max_length should be greater than 0");
    }

    if (batch_size.has_value()) {
        OPENVINO_ASSERT(batch_size.value() > 0, "batch_size should be greater than 0");
    }
}

class TextEmbeddingPipeline::TextEmbeddingPipelineImpl {
public:
    TextEmbeddingPipelineImpl(const std::filesystem::path& models_path,
                              const std::string& device,
                              const Config& config,
                              const ov::AnyMap& properties = {})
        : m_config{config},
          m_tokenizer{models_path},
          m_max_position_embeddings{read_max_position_embeddings(models_path)} {
        m_config.validate();

        ov::Core core = utils::singleton_core();

        auto model = core.read_model(models_path / "openvino_model.xml", {}, std::as_const(properties));

        // Models replacing the embedding by classification heads (e.g. Qwen3Guard) have no
        // `last_hidden_state`, so there is nothing to pool and every output is returned as is.
        m_scores_tokens = !has_port(model->outputs(), "last_hidden_state");
        m_has_position_ids = has_port(model->inputs(), "position_ids");
        m_stateful = has_port(model->inputs(), "beam_idx");
        for (const auto& output : model->outputs()) {
            m_output_names.push_back(output.get_any_name());
        }

        bool is_seq_len_fixed = true;
        m_tokenization_params.insert({truncation.name(), true});

        if (m_config.max_length) {
            m_tokenization_params.insert({max_length.name(), *m_config.max_length});
        } else {
            is_seq_len_fixed = false;
        }

        if (m_config.pad_to_max_length) {
            m_tokenization_params.insert({pad_to_max_length.name(), *m_config.pad_to_max_length});
            is_seq_len_fixed &= m_config.pad_to_max_length.value();
        } else {
            is_seq_len_fixed = false;
        }

        if (m_config.padding_side) {
            m_tokenization_params.insert({padding_side.name(), *m_config.padding_side});
        } else if (m_scores_tokens) {
            // these models are causal, so padding has to go after the scored token, never before it
            m_tokenization_params.insert({padding_side.name(), "right"});
        }

        if (device == "NPU" && !m_scores_tokens) {
            m_request = create_text_embedding_npu_request(model,
                                                          m_config,
                                                          properties,
                                                          m_max_position_embeddings,
                                                          is_seq_len_fixed);
            m_post_request = create_text_embedding_npu_post_request(model, m_config);
        } else {
            if (m_config.batch_size.has_value() || m_config.max_length.has_value()) {
                OPENVINO_ASSERT(!(m_stateful && m_config.pad_to_max_length.value_or(false)),
                                "pad_to_max_length fixes the sequence length, which a model carrying its own "
                                "KV cache cannot use. Export the model with `--task feature-extraction`.");
                utils::reshape_model(model, m_config, m_max_position_embeddings);
            }
            if (!m_scores_tokens) {
                model = utils::apply_postprocessing(model, m_config);
            }
            auto compiled_model = core.compile_model(model, device, properties);
            utils::print_compiled_model_properties(compiled_model, "text embedding model");
            m_request = compiled_model.create_infer_request();
        }
    };

    EmbeddingResults embed_documents(const std::vector<std::string>& texts) {
        start_embed_documents_async(texts);
        return wait_embed_documents();
    };

    EmbeddingResults embed(const std::vector<std::string>& texts, const std::string& prompt) {
        auto formatted_texts = format_texts(texts, prompt);
        start_embed_async(formatted_texts);
        return wait_embed_documents();
    };

    void start_embed_documents_async(const std::vector<std::string>& texts) {
        auto formatted_texts = format_texts(texts);
        start_embed_async(formatted_texts);
    };

    EmbeddingResults wait_embed_documents() {
        return wait_embed();
    };

    EmbeddingResult embed_query(const std::string& text) {
        start_embed_query_async(text);
        return wait_embed_query();
    };

    void start_embed_query_async(const std::string& text) {
        std::vector<std::string> formatted_query{format_query(text)};
        start_embed_async(formatted_query);
    };

    EmbeddingResult wait_embed_query() {
        const EmbeddingResults results = wait_embed();
        if (auto floats = std::get_if<std::vector<std::vector<float>>>(&results)) {
            return (*floats)[0];
        } else if (auto int8s = std::get_if<std::vector<std::vector<int8_t>>>(&results)) {
            return (*int8s)[0];
        } else if (auto uint8s = std::get_if<std::vector<std::vector<uint8_t>>>(&results)) {
            return (*uint8s)[0];
        }
        OPENVINO_THROW("Embedding result type is not supported");
    };

    bool scores_tokens() const {
        return m_scores_tokens;
    };

    bool is_stateful() const {
        return m_stateful;
    };

    std::map<std::string, ov::Tensor> score(const std::vector<std::string>& texts) {
        OPENVINO_ASSERT(!texts.empty(), "No texts to score");
        const TokenizedInputs encoded = m_tokenizer.encode(texts, m_tokenization_params);
        return score(encoded.input_ids, encoded.attention_mask);
    };

    std::map<std::string, ov::Tensor> score(const ov::Tensor& input_ids, const ov::Tensor& attention_mask) {
        const ov::Shape shape = validate_scoring_input(input_ids);
        const ov::Tensor mask = attention_mask ? attention_mask : ones_mask(shape[0], shape[1]);
        OPENVINO_ASSERT(mask.get_shape() == shape, "attention_mask must have the same shape as input_ids");

        // a fresh score of a whole prompt must not see the sequence cached before it
        reset_state();
        set_score_inputs(input_ids, mask);
        return run_scoring();
    };

    std::map<std::string, ov::Tensor> score_next(const ov::Tensor& input_ids) {
        OPENVINO_ASSERT(m_stateful,
                        "score_next() needs the KV cache to live in the model. Export it with "
                        "`--task feature-extraction-with-past`, or call score() instead.");
        const ov::Shape shape = validate_scoring_input(input_ids);
        OPENVINO_ASSERT(shape[0] == 1, "score_next() continues a single sequence, expected [1, chunk_len]");

        // the model only sees the new tokens, but attends over the whole sequence cached so far
        set_score_inputs(input_ids, ones_mask(1, m_past_length + shape[1]));
        m_past_length += shape[1];
        return run_scoring();
    };

    void reset_state() {
        if (m_stateful) {
            m_request.reset_state();
        }
        m_past_length = 0;
    };

private:
    Tokenizer m_tokenizer;
    InferRequest m_request;
    InferRequest m_post_request;
    Config m_config;
    AnyMap m_tokenization_params;
    std::optional<size_t> m_max_position_embeddings;
    ov::Tensor m_attention_mask;

    bool m_scores_tokens = false;
    bool m_has_position_ids = false;
    bool m_stateful = false;
    size_t m_past_length = 0;
    std::vector<std::string> m_output_names;

    ov::Shape validate_scoring_input(const ov::Tensor& input_ids) const {
        OPENVINO_ASSERT(m_scores_tokens,
                        "This model pools its output into a single embedding, use the embed* family instead. "
                        "Per token scores are only produced by models replacing the embedding with "
                        "classification heads, e.g. Qwen3Guard.");
        const ov::Shape shape = input_ids.get_shape();
        OPENVINO_ASSERT(shape.size() == 2, "input_ids must be shaped [batch, seq_len], got ", shape);
        OPENVINO_ASSERT(input_ids.get_element_type() == ov::element::i64,
                        "input_ids must be i64, got ",
                        input_ids.get_element_type());
        return shape;
    }

    void set_score_inputs(const ov::Tensor& input_ids, const ov::Tensor& attention_mask) {
        m_request.set_tensor("input_ids", input_ids);
        m_request.set_tensor("attention_mask", attention_mask);

        if (m_has_position_ids) {
            m_request.set_tensor("position_ids", position_ids_from_mask(attention_mask, input_ids.get_shape()[1]));
        }

        if (m_stateful) {
            // scoring never branches, so every sequence keeps its own cache slot
            ov::Tensor beam_idx(ov::element::i32, Shape{input_ids.get_shape()[0]});
            std::fill_n(beam_idx.data<int32_t>(), beam_idx.get_size(), 0);
            m_request.set_tensor("beam_idx", beam_idx);
        }
    }

    std::map<std::string, ov::Tensor> run_scoring() {
        m_request.infer();

        std::map<std::string, ov::Tensor> scores;
        for (const std::string& name : m_output_names) {
            const ov::Tensor scored = m_request.get_tensor(name);
            // a copy, so the result stays valid once the next call overwrites the request's tensors
            ov::Tensor copy(scored.get_element_type(), scored.get_shape());
            scored.copy_to(copy);
            scores.emplace(name, copy);
        }
        return scores;
    }

    ov::Tensor post_model_infer(const ov::Tensor& input) {
        if (!m_post_request) {
            return input;
        }

        const auto input_shape = input.get_shape();
        const size_t sequence_length = input_shape[1];
        const size_t original_mask_size = m_attention_mask.get_size();
        OPENVINO_ASSERT(sequence_length >= original_mask_size,
                        "Attention mask size mismatch: expected at least ",
                        original_mask_size,
                        " elements, but got ",
                        sequence_length);

        // Create attention mask tensor matching the embedding output shape
        ov::Tensor attention_mask_tensor{ov::element::i64, {1, sequence_length}};

        // Copy original attention mask
        std::copy_n(m_attention_mask.data<int64_t>(), original_mask_size, attention_mask_tensor.data<int64_t>());

        // When prefill-chunk is enabled, the input sequence length is aligned to the chunk size.
        // For example, if the input sequence length is 3800 and the chunk size is 1024, the input
        // sequence length will be reset to 4096. In this case, the attention_mask_tensor size is 4096,
        // which is greater than the original m_attention_mask size of 3800. We need to zero-fill
        // the remaining elements in the attention_mask_tensor to ensure correct masking behavior.
        if (sequence_length > original_mask_size) {
            std::fill_n(attention_mask_tensor.data<int64_t>() + original_mask_size,
                        sequence_length - original_mask_size,
                        0);
        }

        // Run post-processing inference
        m_post_request.set_tensor("attention_mask", attention_mask_tensor);
        m_post_request.set_tensor("embedding_hidden_state", input);
        m_post_request.infer();

        return m_post_request.get_tensor("last_hidden_state");
    }

    void start_embed_async(std::vector<std::string>& texts) {
        OPENVINO_ASSERT(!m_scores_tokens,
                        "This model scores every token through several outputs instead of pooling them into "
                        "a single embedding, use score() instead of the embed* family.");

        if (m_config.batch_size.has_value()) {
            // if batch_size is set, model shape is fixed
            // provide user friendly error message if number of texts is not equal to batch_size
            OPENVINO_ASSERT(texts.size() == *m_config.batch_size,
                            "Number of texts passed to pipeline should be equal to batch_size(",
                            *m_config.batch_size,
                            ")");
        }

        const auto encoded = m_tokenizer.encode(texts, m_tokenization_params);
        m_request.set_tensor("input_ids", encoded.input_ids);
        m_request.set_tensor("attention_mask", encoded.attention_mask);

        m_attention_mask = encoded.attention_mask;

        // fill token_type_ids
        // todo: pass token_type_ids from tokenizer
        if (utils::has_token_type_ids_input(m_request.get_compiled_model().inputs())) {
            ov::Tensor token_type_ids{ov::element::i64, encoded.input_ids.get_shape()};
            std::fill_n(token_type_ids.data<int64_t>(), encoded.input_ids.get_size(), 0);
            m_request.set_tensor("token_type_ids", token_type_ids);
        }

        m_request.start_async();
    };

    EmbeddingResults wait_embed() {
        m_request.wait();

        // [batch_size, hidden_size]
        const auto last_hidden_state = m_request.get_tensor("last_hidden_state");
        return to_embedding_result(post_model_infer(last_hidden_state));
    };

    std::vector<std::string> format_texts(const std::vector<std::string>& texts) {
        if (!m_config.embed_instruction) {
            return texts;
        }

        std::vector<std::string> formatted;
        formatted.reserve(texts.size());

        for (auto& text : texts) {
            formatted.emplace_back(*m_config.embed_instruction + text);
        }
        return formatted;
    }

    std::vector<std::string> format_texts(const std::vector<std::string>& texts, const std::string& prompt) {
        std::vector<std::string> formatted;
        formatted.reserve(texts.size());

        for (const std::string& text : texts) {
            formatted.emplace_back(prompt + text);
        }
        return formatted;
    }

    std::string format_query(const std::string& text) {
        if (!m_config.query_instruction) {
            return text;
        }

        return *m_config.query_instruction + text;
    }

    EmbeddingResults to_embedding_result(const Tensor& last_hidden_state) {
        const float* last_hidden_state_data = last_hidden_state.data<float>();

        std::vector<std::vector<float>> result;
        const auto shape = last_hidden_state.get_shape();

        const size_t batch_size = shape[0];
        const size_t hidden_size = shape[1];

        for (size_t batch = 0; batch < batch_size; batch++) {
            const auto batch_offset = batch * hidden_size;
            const float* batch_data = last_hidden_state_data + batch_offset;
            const std::vector<float> batch_result(batch_data, batch_data + hidden_size);
            result.push_back(batch_result);
        }

        return result;
    }
};

TextEmbeddingPipeline::TextEmbeddingPipeline(const std::filesystem::path& models_path,
                                             const std::string& device,
                                             const Config& config,
                                             const ov::AnyMap& properties) {
    m_impl = std::make_unique<TextEmbeddingPipelineImpl>(models_path, device, config, properties);
};

TextEmbeddingPipeline::TextEmbeddingPipeline(const std::filesystem::path& models_path,
                                             const std::string& device,
                                             const ov::AnyMap& properties) {
    const auto plugin_properties = utils::remove_config_properties(properties);

    m_impl = std::make_unique<TextEmbeddingPipelineImpl>(models_path, device, Config(properties), plugin_properties);
};

EmbeddingResults TextEmbeddingPipeline::embed_documents(const std::vector<std::string>& texts) {
    return m_impl->embed_documents(texts);
}

EmbeddingResults TextEmbeddingPipeline::embed(const std::vector<std::string>& texts, const std::string& prompt) {
    return m_impl->embed(texts, prompt);
}

void TextEmbeddingPipeline::start_embed_documents_async(const std::vector<std::string>& texts) {
    return m_impl->start_embed_documents_async(texts);
}

EmbeddingResults TextEmbeddingPipeline::wait_embed_documents() {
    return m_impl->wait_embed_documents();
}

EmbeddingResult TextEmbeddingPipeline::embed_query(const std::string& text) {
    return m_impl->embed_query(text);
}

void TextEmbeddingPipeline::start_embed_query_async(const std::string& text) {
    return m_impl->start_embed_query_async(text);
}

EmbeddingResult TextEmbeddingPipeline::wait_embed_query() {
    return m_impl->wait_embed_query();
}

bool TextEmbeddingPipeline::scores_tokens() const {
    return m_impl->scores_tokens();
}

bool TextEmbeddingPipeline::is_stateful() const {
    return m_impl->is_stateful();
}

std::map<std::string, ov::Tensor> TextEmbeddingPipeline::score(const std::vector<std::string>& texts) {
    return m_impl->score(texts);
}

std::map<std::string, ov::Tensor> TextEmbeddingPipeline::score(const ov::Tensor& input_ids,
                                                               const ov::Tensor& attention_mask) {
    return m_impl->score(input_ids, attention_mask);
}

std::map<std::string, ov::Tensor> TextEmbeddingPipeline::score_next(const ov::Tensor& input_ids) {
    return m_impl->score_next(input_ids);
}

void TextEmbeddingPipeline::reset_state() {
    m_impl->reset_state();
}

TextEmbeddingPipeline::~TextEmbeddingPipeline() = default;

}  // namespace genai
}  // namespace ov
