// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/text_rerank_pipeline.hpp"

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
        : m_config{config},
          m_tokenizer{models_path, ov::AnyMap{ov::genai::add_second_input(true)}} {
        ov::Core core = utils::singleton_core();

        auto model = core.read_model(models_path / "openvino_model.xml", {}, properties);

        model = apply_postprocessing(model);

        if (m_config.max_length) {
            m_tokenization_params.insert({max_length.name(), *m_config.max_length});
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
        const auto encoded = m_tokenizer.encode({query}, texts, m_tokenization_params);

        m_request.set_tensor("input_ids", encoded.input_ids);
        m_request.set_tensor("attention_mask", encoded.attention_mask);

        if (encoded.token_type_ids.has_value()) {
            m_request.set_tensor("token_type_ids", *encoded.token_type_ids);
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

        return results;
    }

private:
    Tokenizer m_tokenizer;
    InferRequest m_request;
    Config m_config;
    AnyMap m_tokenization_params;
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
