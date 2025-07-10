// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/text_rerank_pipeline.hpp"

#include "debug_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "utils.hpp"

namespace {
using namespace ov::genai;
using namespace ov;

ov::AnyMap remove_config_properties(const ov::AnyMap& properties) {
    auto properties_copy = properties;

    properties_copy.erase(top_n.name());

    return properties_copy;
}
}  // namespace

namespace ov {
namespace genai {
using utils::read_anymap_param;

TextRerankPipeline::Config::Config(const ov::AnyMap& properties) {
    read_anymap_param(properties, ov::genai::top_n.name(), top_n);
};

class TextRerankPipeline::TextRerankPipelineImpl {
public:
    TextRerankPipelineImpl(const std::filesystem::path& models_path,
                           const std::string& device,
                           const Config& config,
                           const ov::AnyMap& properties = {})
        : m_config{config},
          m_tokenizer{models_path} {
        ov::Core core = utils::singleton_core();

        auto model = core.read_model(models_path / "openvino_model.xml", {}, properties);

        ov::CompiledModel compiled_model = core.compile_model(model, device, properties);

        utils::print_compiled_model_properties(compiled_model, "text embedding model");
        m_request = compiled_model.create_infer_request();
    };

    std::vector<std::pair<size_t, float>> rerank(const std::string& query, const std::vector<std::string>& texts) {
        start_rerank_async(query, texts);
        return wait_rerank();
    }

    void start_rerank_async(const std::string& query, const std::vector<std::string>& texts) {
        const auto encoded = m_tokenizer.encode({query}, texts);

        m_request.set_tensor("input_ids", encoded.input_ids);
        m_request.set_tensor("attention_mask", encoded.attention_mask);

        if (encoded.token_type_ids.has_value()) {
            m_request.set_tensor("token_type_ids", *encoded.token_type_ids);
        }

        m_request.start_async();
    }

    std::vector<std::pair<size_t, float>> wait_rerank() {
        m_request.wait();

        auto logits = m_request.get_output_tensor(0);
        auto logits_shape = logits.get_shape();
        const size_t batch_size = logits_shape[0];
        auto logits_data = logits.data<float>();

        std::vector<float> scores;
        scores.reserve(batch_size);

        if (logits_shape[1] == 1) {
            // compute sigmoid
            for (size_t i = 0; i < batch_size; ++i) {
                scores.push_back(1.0f / (1.0f + std::exp(-logits_data[i])));
            }
        } else {
            // comptute softmax of class 1
            for (size_t batch = 0; batch < batch_size; ++batch) {
                const size_t batch_offset = batch * logits_shape[1];
                float exp_sum = 0.0f;
                for (size_t j = 0; j < logits_shape[1]; ++j) {
                    exp_sum += std::exp(logits_data[batch_offset + j]);
                }
                scores.push_back(std::exp(logits_data[batch_offset + 1]) / exp_sum);
            }
        }

        std::vector<std::pair<size_t, float>> results;
        for (size_t batch = 0; batch < batch_size; batch++) {
            results.emplace_back(batch, scores[batch]);
        }

        const size_t top_n = m_config.top_n;

        // partial sort to get top_n results
        std::partial_sort(results.begin(),
                          results.begin() + std::min(top_n, results.size()),
                          results.end(),
                          [](const auto& a, const auto& b) {
                              return a.second > b.second;
                          });

        // return only the top_n elements
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
                                       const ov::AnyMap& properties) {
    m_impl = std::make_unique<TextRerankPipelineImpl>(models_path, device, config, properties);
};

TextRerankPipeline::TextRerankPipeline(const std::filesystem::path& models_path,
                                       const std::string& device,
                                       const ov::AnyMap& properties) {
    const auto& plugin_properties = remove_config_properties(properties);

    m_impl = std::make_unique<TextRerankPipelineImpl>(models_path, device, Config(properties), plugin_properties);
};

std::vector<std::pair<size_t, float>> TextRerankPipeline::rerank(const std::string query,
                                                                 const std::vector<std::string>& texts) {
    return m_impl->rerank(query, texts);
}

void TextRerankPipeline::start_rerank_async(const std::string query, const std::vector<std::string>& texts) {
    m_impl->start_rerank_async(query, texts);
}

std::vector<std::pair<size_t, float>> TextRerankPipeline::wait_rerank() {
    return m_impl->wait_rerank();
}

TextRerankPipeline::~TextRerankPipeline() = default;

}  // namespace genai
}  // namespace ov
