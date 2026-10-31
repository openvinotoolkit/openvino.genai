// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/text_rerank_pipeline.hpp"

#include <algorithm>
#include <fstream>
#include <string_view>
#include <vector>

#include "debug_utils.hpp"
#include "json_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace {
using namespace ov::genai;
using namespace ov;

ov::AnyMap remove_config_properties(const ov::AnyMap& properties) {
    auto properties_copy = properties;

    properties_copy.erase(top_n.name());
    properties_copy.erase(max_length.name());
    properties_copy.erase(pad_to_max_length.name());
    properties_copy.erase(truncation.name());
    properties_copy.erase(padding_side.name());
    properties_copy.erase(instruction.name());

    return properties_copy;
}

std::optional<std::string> read_model_type(const std::filesystem::path& models_path) {
    // config.json not found. Skip parameters initialization from file, use defaults.
    const std::filesystem::path& json_path = models_path / "config.json";
    if (!std::filesystem::exists(json_path)) {
        return std::nullopt;
    }

    using ov::genai::utils::read_json_param;

    std::ifstream f(json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '", json_path);

    nlohmann::json data = nlohmann::json::parse(f);

    std::optional<std::string> model_type;
    read_json_param(data, "model_type", model_type);
    return model_type;
}

bool has_input(const std::shared_ptr<Model>& model, const std::string& name) {
    for (const auto& input : model->inputs()) {
        if (input.get_any_name() == name) {
            return true;
        }
    }
    return false;
}

// Qwen3 rerankers are instruction-following causal LMs. apply_qwen3_postprocessing() below reads the score from
// the logits of the last token, so the suffix must end exactly where the next token is the "yes"/"no" verdict.
// Mirrors https://huggingface.co/Qwen/Qwen3-Reranker-0.6B#transformers-usage
constexpr std::string_view QWEN3_RERANK_PREFIX =
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
    "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
constexpr std::string_view QWEN3_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
constexpr std::string_view QWEN3_RERANK_INSTRUCTION =
    "Given a web search query, retrieve relevant passages that answer the query";

std::string format_qwen3_query(const std::string& query, std::string_view instruction) {
    std::string formatted{QWEN3_RERANK_PREFIX};
    formatted.append("<Instruct>: ").append(instruction).append("\n<Query>: ").append(query).append("\n");
    return formatted;
}

std::string format_qwen3_document(const std::string& document) {
    // The suffix is added by encode_qwen3(): in the prompt string when nothing is truncated, or at the
    // token level after truncation. It is intentionally omitted from the content string here.
    return "<Document>: " + document;
}

std::vector<int64_t> tokenize_qwen3_suffix(Tokenizer& tokenizer) {
    const TokenizedInputs encoded =
        tokenizer.encode(std::string{QWEN3_RERANK_SUFFIX}, ov::genai::add_special_tokens(false));
    const int64_t* data = encoded.input_ids.data<int64_t>();
    return {data, data + encoded.input_ids.get_size()};
}

std::shared_ptr<Model> apply_postprocessing(std::shared_ptr<Model> model) {
    PartialShape output_shape = model->get_output_partial_shape(0);

    ov::preprocess::PrePostProcessor processor(model);

    processor.output().postprocess().custom(
        [&output_shape](const ov::Output<ov::Node>& node) -> std::shared_ptr<ov::Node> {
            if (output_shape[1] == 1) {
                return std::make_shared<op::v0::Sigmoid>(node);
            }

            // apply softmax to the axis = 1
            const auto softmax = std::make_shared<op::v8::Softmax>(node, 1);

            // take first class score only
            auto start = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
            auto stop = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
            auto step = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
            auto axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});

            auto slice = std::make_shared<op::v8::Slice>(softmax, start, stop, step, axis);
            return slice;
        });

    return processor.build();
}

struct Qwen3PostprocessingParams {
    int64_t token_true_id;
    int64_t token_false_id;
};

std::shared_ptr<Model> apply_qwen3_postprocessing(std::shared_ptr<Model> model,
                                                  const Qwen3PostprocessingParams& params) {
    PartialShape output_shape = model->get_output_partial_shape(0);

    ov::preprocess::PrePostProcessor processor(model);

    processor.output().postprocess().custom([&output_shape,
                                             &params](const ov::Output<ov::Node>& node) -> std::shared_ptr<ov::Node> {
        // to support models with embedded postprocessing like tomaarsen/Qwen3-Reranker-0.6B-seq-cls
        if (output_shape[1] == 1) {
            return std::make_shared<op::v0::Sigmoid>(node);
        }

        auto start = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
        auto stop = std::make_shared<op::v0::Constant>(ov::element::i64,
                                                       ov::Shape{1},
                                                       std::vector<int64_t>{std::numeric_limits<int64_t>::max()});
        auto step = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        auto axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});

        auto last_token_slice = std::make_shared<op::v8::Slice>(node, start, stop, step, axis);

        auto squeeze = std::make_shared<op::v0::Squeeze>(last_token_slice, axis);

        auto indices =
            std::make_shared<op::v0::Constant>(ov::element::i64,
                                               ov::Shape{2},
                                               std::vector<int64_t>{params.token_false_id, params.token_true_id});
        auto gather_axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        auto gather = std::make_shared<op::v8::Gather>(squeeze, indices, gather_axis);

        auto softmax = std::make_shared<op::v8::Softmax>(gather, 1);

        auto last_token_slice_2 = std::make_shared<op::v8::Slice>(softmax, start, stop, step, axis);

        auto squeeze_2 = std::make_shared<op::v0::Squeeze>(last_token_slice_2, axis);
        return squeeze_2;
    });

    return processor.build();
}

}  // namespace

namespace ov {
namespace genai {
using utils::read_anymap_param;

TextRerankPipeline::Config::Config(const ov::AnyMap& properties) {
    read_anymap_param(properties, ov::genai::top_n.name(), top_n);
    read_anymap_param(properties, ov::genai::max_length.name(), max_length);
    read_anymap_param(properties, ov::genai::padding_side.name(), padding_side);
    read_anymap_param(properties, ov::genai::pad_to_max_length.name(), pad_to_max_length);
    read_anymap_param(properties, ov::genai::instruction.name(), instruction);
};

class TextRerankPipeline::TextRerankPipelineImpl {
public:
    TextRerankPipelineImpl(const std::filesystem::path& models_path,
                           const std::string& device,
                           const Config& config,
                           const ov::AnyMap& properties = {})
        : m_config{config} {
        const auto model_type = read_model_type(models_path);
        const bool is_qwen3 = model_type.has_value() && model_type.value() == "qwen3";
        m_is_qwen3 = is_qwen3;

        m_tokenization_params.insert({truncation.name(), true});

        if (m_config.max_length) {
            m_tokenization_params.insert({max_length.name(), *m_config.max_length});
        }

        if (m_config.pad_to_max_length) {
            m_tokenization_params.insert({pad_to_max_length.name(), *m_config.pad_to_max_length});
        }

        if (m_config.padding_side) {
            m_tokenization_params.insert({padding_side.name(), *m_config.padding_side});
        }

        // qwen3 tokenizer doesn't support add_second_input(true)
        m_tokenizer = Tokenizer(models_path, ov::genai::add_second_input(!is_qwen3));

        ov::Core core = utils::singleton_core();

        auto model = core.read_model(models_path / "openvino_model.xml", {}, properties);

        m_has_position_ids = has_input(model, "position_ids");
        m_has_beam_idx = has_input(model, "beam_idx");

        if (is_qwen3) {
            m_qwen3_suffix_ids = tokenize_qwen3_suffix(m_tokenizer);
            const auto vocab = m_tokenizer.get_vocab();
            const auto token_true_id = vocab.at("yes");
            const auto token_false_id = vocab.at("no");
            model = apply_qwen3_postprocessing(model, {token_true_id, token_false_id});
        } else {
            model = apply_postprocessing(model);
        }

        // A stateful decoder reranker (a Qwen3-Reranker-style causal LM, recognized by its
        // beam_idx input) runs on NPU through the NPUW LLM pipeline, scoring a [N, L] batch
        // row by row. Encoder cross-encoders keep the plain compile path.
        ov::CompiledModel compiled_model;
        if (device == "NPU" && m_has_beam_idx) {
            compiled_model =
                utils::compile_decoder_for_npu_text_rerank(model, properties, utils::get_kv_axes_pos(model), m_config)
                    .first;
        } else {
            compiled_model = core.compile_model(model, device, properties);
        }

        utils::print_compiled_model_properties(compiled_model, "text rerank model");
        m_request = compiled_model.create_infer_request();
    };

    std::vector<std::pair<size_t, float>> rerank(const std::string& query, const std::vector<std::string>& texts) {
        start_rerank_async(query, texts);
        return wait_rerank();
    }

    void start_rerank_async(const std::string& query, const std::vector<std::string>& texts) {
        const TokenizedInputs& encoded = tokenize(query, texts);

        m_request.set_tensor("input_ids", encoded.input_ids);
        m_request.set_tensor("attention_mask", encoded.attention_mask);

        if (encoded.token_type_ids.has_value()) {
            m_request.set_tensor("token_type_ids", *encoded.token_type_ids);
        }

        if (m_has_position_ids) {
            ov::Tensor position_ids(encoded.input_ids.get_element_type(), encoded.input_ids.get_shape());
            utils::initialize_position_ids(position_ids, encoded.attention_mask, 0);
            m_request.set_tensor("position_ids", position_ids);
        }

        if (m_has_beam_idx) {
            const size_t batch_size = encoded.input_ids.get_shape()[0];
            ov::Tensor beam_idx = ov::Tensor(ov::element::i32, {batch_size});
            std::fill_n(beam_idx.data<int32_t>(), batch_size, 0);
            m_request.set_tensor("beam_idx", beam_idx);
        }

        m_request.start_async();
    }

    std::vector<std::pair<size_t, float>> wait_rerank() {
        m_request.wait();

        // postprocessing applied to output, it's the scores tensor
        auto scores_tensor = m_request.get_tensor("logits");
        auto scores_tensor_shape = scores_tensor.get_shape();
        const size_t batch_size = scores_tensor_shape[0];

        auto scores_data = scores_tensor.data<float>();

        std::vector<std::pair<size_t, float>> results;
        results.reserve(batch_size);

        for (size_t batch = 0; batch < batch_size; batch++) {
            results.emplace_back(batch, scores_data[batch]);
        }

        const size_t top_n = m_config.top_n;

        // partial sort to get top_n results
        std::partial_sort(results.begin(),
                          results.begin() + std::min(top_n, results.size()),
                          results.end(),
                          [](const auto& a, const auto& b) {
                              return a.second > b.second;
                          });

        if (top_n < results.size()) {
            results.resize(top_n);
        }

        if (m_has_beam_idx) {
            m_request.reset_state();
        }

        return results;
    }

private:
    Tokenizer m_tokenizer;
    InferRequest m_request;
    Config m_config;
    AnyMap m_tokenization_params;
    bool m_has_position_ids = false;
    bool m_has_beam_idx = false;
    bool m_is_qwen3 = false;
    std::vector<int64_t> m_qwen3_suffix_ids;

    TokenizedInputs tokenize(const std::string& query, const std::vector<std::string>& texts) {
        if (m_tokenizer.supports_paired_input()) {
            return m_tokenizer.encode({query}, texts, m_tokenization_params);
        }

        std::vector<std::string> concatenated;
        concatenated.reserve(texts.size());

        if (m_is_qwen3) {
            const std::string_view instruction =
                m_config.instruction ? std::string_view{*m_config.instruction} : QWEN3_RERANK_INSTRUCTION;
            const std::string formatted_query = format_qwen3_query(query, instruction);
            for (const auto& text : texts) {
                concatenated.push_back(formatted_query + format_qwen3_document(text));
            }
            return encode_qwen3(concatenated);
        }

        for (const auto& text : texts) {
            concatenated.push_back(query + text);
        }

        return m_tokenizer.encode(concatenated, m_tokenization_params);
    }

    // The model scores the last token, so the suffix must survive truncation. Keep the suffix in the
    // prompt string and tokenize in a single pass whenever the content fits (so the token sequence
    // matches the reference exactly). Only when truncation is actually required do we reserve the suffix
    // budget, truncate the content, and re-append the suffix at the token level.
    TokenizedInputs encode_qwen3(const std::vector<std::string>& content_prompts) {
        OPENVINO_ASSERT(!m_config.padding_side || *m_config.padding_side == "left",
                        "Qwen3 rerankers require padding_side='left'");
        // pad_to_max_length without a max_length pads to the model's exported limit, leaving no room to re-append the
        // suffix after truncation; require an explicit limit so the suffix budget can be reserved.
        OPENVINO_ASSERT(m_config.max_length || !(m_config.pad_to_max_length && *m_config.pad_to_max_length),
                        "Qwen3 rerankers require max_length to be set when pad_to_max_length=true");

        ov::AnyMap params = m_tokenization_params;
        params[padding_side.name()] = std::string("left");

        std::vector<std::string> full_prompts;
        full_prompts.reserve(content_prompts.size());
        for (const auto& prompt : content_prompts) {
            full_prompts.push_back(prompt + std::string{QWEN3_RERANK_SUFFIX});
        }

        if (!m_config.max_length) {
            // No explicit budget: a single pass keeps the suffix in the string. The tokenizer still truncates to the
            // model's exported limit, which on a right-truncating export could drop the suffix, so verify it survived.
            const TokenizedInputs encoded = m_tokenizer.encode(full_prompts, params);
            if (qwen3_suffix_preserved(encoded)) {
                return encoded;
            }
            // The exported-limit truncation cut the suffix. Reserve its budget against that limit and re-append it.
            const size_t effective_limit = encoded.input_ids.get_shape()[1];
            OPENVINO_ASSERT(effective_limit > m_qwen3_suffix_ids.size(),
                            "The tokenizer's exported max_length (",
                            effective_limit,
                            ") must exceed the Qwen3 suffix length (",
                            m_qwen3_suffix_ids.size(),
                            ")");
            ov::AnyMap reserve_params = params;
            reserve_params[max_length.name()] = effective_limit - m_qwen3_suffix_ids.size();
            return append_qwen3_suffix(m_tokenizer.encode(content_prompts, reserve_params));
        }

        OPENVINO_ASSERT(*m_config.max_length > m_qwen3_suffix_ids.size(),
                        "max_length (",
                        *m_config.max_length,
                        ") must exceed the Qwen3 suffix length (",
                        m_qwen3_suffix_ids.size(),
                        ")");

        // Probe whether the prompt overflows without materializing the full untruncated sequence: truncating the
        // measurement to max_length + 1 tokens is enough to tell "fits" from "overflows" and caps the allocation,
        // so a single very long document can't exhaust memory before the real pass runs.
        if (*m_config.max_length == std::numeric_limits<size_t>::max()) {
            return m_tokenizer.encode(full_prompts, params);
        }
        ov::AnyMap measure_params = params;
        measure_params[max_length.name()] = *m_config.max_length + 1;
        measure_params.erase(pad_to_max_length.name());
        const TokenizedInputs measured = m_tokenizer.encode(full_prompts, measure_params);
        if (measured.input_ids.get_shape()[1] <= *m_config.max_length) {
            // measured already holds the fitted IDs and left-padding; re-encode only to pad to a fixed max_length.
            if (!m_config.pad_to_max_length.value_or(false)) {
                return measured;
            }
            return m_tokenizer.encode(full_prompts, params);
        }

        // Truncation needed: reserve the suffix budget on the content, then re-append the suffix tokens.
        params[max_length.name()] = *m_config.max_length - m_qwen3_suffix_ids.size();
        return append_qwen3_suffix(m_tokenizer.encode(content_prompts, params));
    }

    // Left padding puts the real tokens at the tail, so an intact suffix occupies the last suffix_len positions of
    // every row. A mismatch means the exported-limit truncation cut into it.
    bool qwen3_suffix_preserved(const TokenizedInputs& encoded) const {
        const size_t suffix_len = m_qwen3_suffix_ids.size();
        if (suffix_len == 0) {
            return true;
        }

        const ov::Shape shape = encoded.input_ids.get_shape();
        const size_t batch = shape[0];
        const size_t length = shape[1];
        if (length < suffix_len) {
            return false;
        }

        const int64_t* ids = encoded.input_ids.data<int64_t>();
        for (size_t b = 0; b < batch; ++b) {
            const int64_t* tail = ids + b * length + (length - suffix_len);
            if (!std::equal(m_qwen3_suffix_ids.begin(), m_qwen3_suffix_ids.end(), tail)) {
                return false;
            }
        }
        return true;
    }

    TokenizedInputs append_qwen3_suffix(const TokenizedInputs& encoded) const {
        const size_t suffix_len = m_qwen3_suffix_ids.size();
        if (suffix_len == 0) {
            return encoded;
        }

        const ov::Shape shape = encoded.input_ids.get_shape();
        const size_t batch = shape[0];
        const size_t length = shape[1];
        const size_t new_length = length + suffix_len;

        ov::Tensor input_ids(ov::element::i64, {batch, new_length});
        ov::Tensor attention_mask(ov::element::i64, {batch, new_length});

        const int64_t* src_ids = encoded.input_ids.data<int64_t>();
        const int64_t* src_mask = encoded.attention_mask.data<int64_t>();
        int64_t* dst_ids = input_ids.data<int64_t>();
        int64_t* dst_mask = attention_mask.data<int64_t>();

        for (size_t b = 0; b < batch; ++b) {
            std::copy_n(src_ids + b * length, length, dst_ids + b * new_length);
            std::copy(m_qwen3_suffix_ids.begin(), m_qwen3_suffix_ids.end(), dst_ids + b * new_length + length);
            std::copy_n(src_mask + b * length, length, dst_mask + b * new_length);
            std::fill_n(dst_mask + b * new_length + length, suffix_len, int64_t{1});
        }

        return {input_ids, attention_mask, encoded.token_type_ids};
    }
};

TextRerankPipeline::TextRerankPipeline(const std::filesystem::path& models_path,
                                       const std::string& device,
                                       const Config& config,
                                       const ov::AnyMap& properties)
    : m_impl{std::make_unique<TextRerankPipelineImpl>(models_path, device, config, properties)} {};

TextRerankPipeline::TextRerankPipeline(const std::filesystem::path& models_path,
                                       const std::string& device,
                                       const ov::AnyMap& properties)
    : m_impl{std::make_unique<TextRerankPipelineImpl>(models_path,
                                                      device,
                                                      Config(properties),
                                                      remove_config_properties(properties))} {};

std::vector<std::pair<size_t, float>> TextRerankPipeline::rerank(const std::string& query,
                                                                 const std::vector<std::string>& texts) {
    return m_impl->rerank(query, texts);
}

void TextRerankPipeline::start_rerank_async(const std::string& query, const std::vector<std::string>& texts) {
    m_impl->start_rerank_async(query, texts);
}

std::vector<std::pair<size_t, float>> TextRerankPipeline::wait_rerank() {
    return m_impl->wait_rerank();
}

TextRerankPipeline::~TextRerankPipeline() = default;

}  // namespace genai
}  // namespace ov
