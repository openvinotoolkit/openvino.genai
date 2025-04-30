// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/text_embedding_pipeline.hpp"

#include "debug_utils.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace {
using namespace ov::genai;
using namespace ov;

ov::AnyMap remove_config_properties(const ov::AnyMap& properties) {
    auto properties_copy = properties;

    properties_copy.erase(max_length.name());
    properties_copy.erase(pooling_type.name());
    properties_copy.erase(normalize.name());
    properties_copy.erase(embed_instruction.name());
    properties_copy.erase(query_instruction.name());

    return properties_copy;
}

/**
 * CLS pooling slices first element from seq_length dimension
 * [batch_size, seq_length, hidden_size] -> [batch_size, seq_length[0], hidden_size]
 * [10, 5, 768] -> [10, 768]
 */
std::shared_ptr<op::Op> get_cls_pooling_op(const ov::Output<ov::Node>& last_hidden_state_node) {
    auto start = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto stop = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto step = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});

    auto slice = std::make_shared<op::v8::Slice>(last_hidden_state_node, start, stop, step, axis);

    auto squeeze_axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    return std::make_shared<op::v15::Squeeze>(slice, squeeze_axis);
}

std::shared_ptr<op::Op> get_mean_pooling_op(std::shared_ptr<Model> model,
                                            const ov::Output<ov::Node>& last_hidden_state_node) {
    auto shape_of = std::make_shared<op::v3::ShapeOf>(last_hidden_state_node);

    auto attention_mask = model->input("attention_mask").get_node()->outputs()[0];

    auto unsqueze_axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    auto unsqueze = std::make_shared<op::v0::Unsqueeze>(attention_mask, unsqueze_axis);

    auto input_mask_expanded = std::make_shared<op::v3::Broadcast>(unsqueze, shape_of);

    auto input_mask_expanded_convert =
        std::make_shared<op::v0::Convert>(input_mask_expanded, last_hidden_state_node.get_element_type());

    auto last_hidden_node_with_applied_attention_mask =
        std::make_shared<op::v1::Multiply>(last_hidden_state_node, input_mask_expanded_convert->outputs()[0]);

    auto axis_1 = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto sum_hidden_state = std::make_shared<op::v1::ReduceSum>(last_hidden_node_with_applied_attention_mask, axis_1);

    // f32 overflow possible
    // ReduceMean might help with overflow but its precision diverges from LlamaIndex
    auto sum_expanded_mask = std::make_shared<op::v1::ReduceSum>(input_mask_expanded_convert, axis_1);

    auto nearest_to_zero =
        std::make_shared<op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{1e-12});
    auto max_expanded_mask = std::make_shared<op::v1::Maximum>(sum_expanded_mask, nearest_to_zero);

    // shape: [batch_size, hidden_state_size]
    return std::make_shared<op::v1::Divide>(sum_hidden_state, max_expanded_mask);
}

std::shared_ptr<Model> apply_postprocessing(std::shared_ptr<Model> model, const TextEmbeddingPipeline::Config& config) {
    ov::preprocess::PrePostProcessor processor(model);

    processor.output().postprocess().custom([model, &config](const ov::Output<ov::Node>& node) {
        if (config.pooling_type == TextEmbeddingPipeline::PoolingType::CLS) {
            return get_cls_pooling_op(node);
        } else if (config.pooling_type == TextEmbeddingPipeline::PoolingType::MEAN) {
            return get_mean_pooling_op(model, node);
        }

        OPENVINO_THROW("Pooling type is not supported");
    });

    if (config.normalize) {
        processor.output().postprocess().custom([](const ov::Output<ov::Node>& node) {
            auto axis_const = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{1});
            return std::make_shared<op::v0::NormalizeL2>(node, axis_const, 1e-12, op::EpsMode::ADD);
        });
    }

    return processor.build();
}
}  // namespace

namespace ov {
namespace genai {
using utils::read_anymap_param;

TextEmbeddingPipeline::Config::Config(const ov::AnyMap& properties) {
    read_anymap_param(properties, ov::genai::max_length.name(), max_length);
    read_anymap_param(properties, ov::genai::pooling_type.name(), pooling_type);
    read_anymap_param(properties, ov::genai::normalize.name(), normalize);
    read_anymap_param(properties, ov::genai::embed_instruction.name(), embed_instruction);
    read_anymap_param(properties, ov::genai::query_instruction.name(), query_instruction);
};

class TextEmbeddingPipeline::TextEmbeddingPipelineImpl {
public:
    TextEmbeddingPipelineImpl(const std::filesystem::path& models_path,
                              const std::string& device,
                              const Config& config,
                              const ov::AnyMap& properties = {})
        : m_config{config},
          m_tokenizer{models_path} {
        ov::Core core = utils::singleton_core();

        auto model = core.read_model(models_path / "openvino_model.xml", {}, properties);

        model = apply_postprocessing(model, m_config);
        if (m_config.max_length) {
            m_tokenization_params.insert({max_length.name(), *m_config.max_length});
        }

        ov::CompiledModel compiled_model = core.compile_model(model, device, properties);

        utils::print_compiled_model_properties(compiled_model, "text embedding model");
        m_request = compiled_model.create_infer_request();
    };

    std::vector<EmbeddingResult> embed_documents(const std::vector<std::string>& texts) {
        start_embed_documents_async(texts);
        return wait_embed_documents();
    };

    void start_embed_documents_async(const std::vector<std::string>& texts) {
        auto formatted_texts = format_texts(texts);
        start_embed_async(formatted_texts);
    };

    std::vector<EmbeddingResult> wait_embed_documents() {
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
        return wait_embed()[0];
    };

private:
    Tokenizer m_tokenizer;
    InferRequest m_request;
    Config m_config;
    AnyMap m_tokenization_params;

    void start_embed_async(std::vector<std::string>& texts) {
        const auto encoded = m_tokenizer.encode(texts, m_tokenization_params);

        m_request.set_tensor("input_ids", encoded.input_ids);
        m_request.set_tensor("attention_mask", encoded.attention_mask);

        // fill token_type_ids
        // todo: pass token_type_ids from tokenizer
        for (auto& input : m_request.get_compiled_model().inputs()) {
            if (input.get_any_name() == "token_type_ids") {
                ov::Tensor token_type_ids{ov::element::i64, encoded.input_ids.get_shape()};
                std::fill_n(token_type_ids.data<int64_t>(), encoded.input_ids.get_size(), 0);
                m_request.set_tensor("token_type_ids", token_type_ids);
                break;
            }
        }

        m_request.start_async();
    };

    std::vector<EmbeddingResult> wait_embed() {
        m_request.wait();

        // [batch_size, hidden_size]
        const Tensor last_hidden_state = m_request.get_tensor("last_hidden_state");

        return to_embedding_result(last_hidden_state);
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

    std::string format_query(const std::string& text) {
        if (!m_config.query_instruction) {
            return text;
        }

        return *m_config.query_instruction + text;
    }

    std::vector<EmbeddingResult> to_embedding_result(const Tensor& last_hidden_state) {
        const auto last_hidden_state_data = last_hidden_state.data<float>();

        std::vector<EmbeddingResult> result;
        const auto shape = last_hidden_state.get_shape();

        const size_t batch_size = shape[0];
        const size_t hidden_size = shape[1];

        for (size_t batch = 0; batch < batch_size; batch++) {
            const auto batch_offset = batch * hidden_size;
            const auto batch_data = last_hidden_state_data + batch_offset;
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
    const auto& plugin_properties = remove_config_properties(properties);

    m_impl = std::make_unique<TextEmbeddingPipelineImpl>(models_path, device, Config(properties), plugin_properties);
};

std::vector<EmbeddingResult> TextEmbeddingPipeline::embed_documents(const std::vector<std::string>& texts) {
    return m_impl->embed_documents(texts);
}

void TextEmbeddingPipeline::start_embed_documents_async(const std::vector<std::string>& texts) {
    return m_impl->start_embed_documents_async(texts);
}

std::vector<EmbeddingResult> TextEmbeddingPipeline::wait_embed_documents() {
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

TextEmbeddingPipeline::~TextEmbeddingPipeline() = default;

}  // namespace genai
}  // namespace ov
