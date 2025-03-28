// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag_components/embedding_pipeline.hpp"

#include "debug_utils.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

class TextEmbeddingPipeline::TextEmbeddingPipelineImpl {
public:
    TextEmbeddingPipelineImpl(const std::filesystem::path& models_path,
                              const std::string& device,
                              const std::optional<Config>& config,
                              const ov::AnyMap& properties = {})
        : m_tokenizer{models_path} {
        ov::Core core = utils::singleton_core();

        ov::CompiledModel compiled_model = core.compile_model(models_path / "openvino_model.xml", device, properties);
        ov::genai::utils::print_compiled_model_properties(compiled_model, "embedding model");
        m_request = compiled_model.create_infer_request();
    };

    std::vector<EmbeddingResult> embed_documents(const std::vector<std::string>& texts) {
        std::vector<std::string> copy{texts};
        const auto tokenized_inputs = m_tokenizer.encode(copy);

        m_request.set_tensor("input_ids", tokenized_inputs.input_ids);
        m_request.set_tensor("attention_mask", tokenized_inputs.attention_mask);

        m_request.infer();

        // [batch_size, seq_len, hidden_size]
        const Tensor last_hidden_state = m_request.get_tensor("last_hidden_state");

        // todo: implement mean_pooling
        // todo: implement l2 norm

        if (m_config.pooling_type == PoolingType::CLS) {
            return cls_pooling(last_hidden_state);
        } else {
            OPENVINO_THROW("Only PoolingType::CLS is supported");
        }
    };

    EmbeddingResult embed_query(const std::string& text) {
        return embed_documents({text})[0];
    };

private:
    Tokenizer m_tokenizer;
    InferRequest m_request;
    Config m_config;

    std::vector<EmbeddingResult> cls_pooling(Tensor last_hidden_state) {
        const auto last_hidden_state_data = last_hidden_state.data<float>();

        std::vector<EmbeddingResult> result;
        const auto shape = last_hidden_state.get_shape();
        const size_t batch_size = shape[0];
        const size_t seq_len = shape[1];
        const size_t hidden_size = shape[2];

        for (size_t batch = 0; batch < batch_size; batch++) {
            const auto batch_offset = batch * seq_len * hidden_size;
            const auto batch_data = last_hidden_state_data + batch_offset;
            const std::vector<float> batch_result(batch_data, batch_data + hidden_size);
            result.push_back(batch_result);
        }

        return result;
    }
};

TextEmbeddingPipeline::TextEmbeddingPipeline(const std::filesystem::path& models_path,
                                             const std::string& device,
                                             const std::optional<Config>& config,
                                             const ov::AnyMap& properties) {
    m_impl = std::make_unique<TextEmbeddingPipelineImpl>(models_path, device, config, properties);
};

std::vector<EmbeddingResult> TextEmbeddingPipeline::embed_documents(const std::vector<std::string>& texts) {
    return m_impl->embed_documents(texts);
}

EmbeddingResult TextEmbeddingPipeline::embed_query(const std::string& text) {
    return m_impl->embed_query(text);
}

TextEmbeddingPipeline::~TextEmbeddingPipeline() = default;

}  // namespace genai
}  // namespace ov
