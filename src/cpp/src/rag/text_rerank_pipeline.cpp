// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/text_rerank_pipeline.hpp"

#include <fstream>

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

        if (m_config.max_length) {
            m_tokenization_params.insert({max_length.name(), *m_config.max_length});
        }

        // qwen3 tokenizer doesn't support add_second_input(true)
        m_tokenizer = Tokenizer(models_path, ov::genai::add_second_input(!is_qwen3));

        ov::Core core = utils::singleton_core();

        auto model = core.read_model(models_path / "openvino_model.xml", {}, properties);

        m_has_position_ids = has_input(model, "position_ids");
        m_has_beam_idx = has_input(model, "beam_idx");

        if (is_qwen3) {
            const auto vocab = m_tokenizer.get_vocab();
            const auto token_true_id = vocab.at("yes");
            const auto token_false_id = vocab.at("no");
            model = apply_qwen3_postprocessing(model, {token_true_id, token_false_id});
        } else {
            model = apply_postprocessing(model);
        }

        ov::CompiledModel compiled_model = core.compile_model(model, device, properties);

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

    TokenizedInputs tokenize(const std::string& query, const std::vector<std::string>& texts) {
        if (m_tokenizer.supports_paired_input()) {
            return m_tokenizer.encode({query}, texts, m_tokenization_params);
        }

        std::vector<std::string> concatenated;
        concatenated.reserve(texts.size());

        for (auto& text : texts) {
            concatenated.push_back(query + text);
        }

        return m_tokenizer.encode(concatenated, m_tokenization_params);
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
