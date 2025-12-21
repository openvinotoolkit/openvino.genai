// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/text_embedding_pipeline.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "json_utils.hpp"
#include "logger.hpp"
#include "openvino/core/except.hpp"
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
    properties_copy.erase(pad_to_max_length.name());
    properties_copy.erase(batch_size.name());
    properties_copy.erase(pooling_type.name());
    properties_copy.erase(normalize.name());
    properties_copy.erase(embed_instruction.name());
    properties_copy.erase(query_instruction.name());
    properties_copy.erase(padding_side.name());

    return properties_copy;
}

template <typename T>
bool has_token_type_ids_input(const T& inputs) {
    for (const auto& input : inputs) {
        if (input.get_any_name() == "token_type_ids") {
            return true;
        }
    }
    return false;
}

void set_node_name(std::shared_ptr<ov::Node> node, const std::string& name) {
    node->set_friendly_name(name);
    node->get_output_tensor(0).set_names({name});
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

std::shared_ptr<op::Op> get_mean_pooling_op(const ov::Output<ov::Node>& last_hidden_state_node,
                                            const ov::Output<ov::Node>& attention_mask) {
    auto shape_of = std::make_shared<op::v3::ShapeOf>(last_hidden_state_node);

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

std::shared_ptr<op::Op> get_last_token_pooling_op(const ov::Output<ov::Node>& last_hidden_state_node,
                                                  const ov::Output<ov::Node>& attention_mask,
                                                  const TextEmbeddingPipeline::Config& config) {
    const auto left_padding = config.padding_side.has_value() && config.padding_side.value() == "left";

    // shortcut for left padding. We can slice last token directly
    if (left_padding) {
        auto start = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
        auto stop = std::make_shared<op::v0::Constant>(ov::element::i64,
                                                       ov::Shape{1},
                                                       std::vector<int64_t>{std::numeric_limits<int64_t>::max()});
        auto step = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        auto axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});

        auto slice = std::make_shared<op::v8::Slice>(last_hidden_state_node, start, stop, step, axis);

        auto squeeze_axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        return std::make_shared<op::v15::Squeeze>(slice, squeeze_axis);
    }

    auto axis_1 = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto reduce_sum = std::make_shared<op::v1::ReduceSum>(attention_mask, axis_1);
    auto subtract_1 = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto subtract = std::make_shared<op::v1::Subtract>(reduce_sum, subtract_1);

    return std::make_shared<op::v8::Gather>(last_hidden_state_node, subtract, axis_1, 1);
}

std::shared_ptr<op::Op> create_post_ops(const ov::Output<ov::Node>& input,
                                        const ov::Output<ov::Node>& attention_mask,
                                        const TextEmbeddingPipeline::Config& config) {
    if (config.pooling_type == TextEmbeddingPipeline::PoolingType::CLS) {
        return get_cls_pooling_op(input);
    } else if (config.pooling_type == TextEmbeddingPipeline::PoolingType::MEAN) {
        return get_mean_pooling_op(input, attention_mask);
    } else if (config.pooling_type == TextEmbeddingPipeline::PoolingType::LAST_TOKEN) {
        return get_last_token_pooling_op(input, attention_mask, config);
    }

    OPENVINO_THROW("Pooling type is not supported");
}

std::shared_ptr<op::Op> create_normalize_ops(const ov::Output<ov::Node>& input,
                                            const TextEmbeddingPipeline::Config& config) {
    if (config.normalize) {
        auto axis_const = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{1});
        return std::make_shared<op::v0::NormalizeL2>(input, axis_const, 1e-12, op::EpsMode::MAX);
    }
    return std::dynamic_pointer_cast<op::Op>(input.get_node_shared_ptr());
}

std::shared_ptr<Model> apply_postprocessing(std::shared_ptr<Model> model, const TextEmbeddingPipeline::Config& config) {
    ov::preprocess::PrePostProcessor processor(model);

    processor.output().postprocess().custom([model, &config](const ov::Output<ov::Node>& node) {
        auto attention_mask = model->input("attention_mask").get_node()->outputs()[0];
        return create_post_ops(node, attention_mask, config);
    });

    if (config.normalize) {
        processor.output().postprocess().custom([&config](const ov::Output<ov::Node>& node) {
            return create_normalize_ops(node, config);
        });
    }

    return processor.build();
}

std::shared_ptr<ov::Model> create_post_model(std::shared_ptr<ov::Model> model,
                                            const TextEmbeddingPipeline::Config& config,
                                            ov::Dimension::value_type max_prompt_size) {
    auto output_node = model->outputs()[0];
    auto output_shape = output_node.get_partial_shape();
    auto input_param =
        std::make_shared<ov::op::v0::Parameter>(output_node.get_element_type(), ov::PartialShape{1, max_prompt_size, output_shape[2]});
    set_node_name(input_param, "embedding_hidden_state");

    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, max_prompt_size});
    set_node_name(attention_mask, "attention_mask");

    auto post_output = create_post_ops(input_param, attention_mask, config);
    auto post_normalize_output = create_normalize_ops(post_output, config);
    OPENVINO_ASSERT(post_normalize_output != nullptr);

    auto result_node = std::make_shared<ov::op::v0::Result>(post_normalize_output);
    set_node_name(result_node, "last_hidden_state");
    auto post_model =
        std::make_shared<ov::Model>(ov::OutputVector{result_node}, ov::ParameterVector{input_param, attention_mask});
    post_model->set_friendly_name(model->get_friendly_name() + "_post_process");
    post_model->validate_nodes_and_infer_types();
    return post_model;
}

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

        auto model = core.read_model(models_path / "openvino_model.xml", {}, properties);

        bool is_seq_len_fixed = true;
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
        }

        bool should_reshape_non_npu =
            (device != "NPU" && (m_config.batch_size.has_value() || m_config.max_length.has_value()));
        bool should_reshape_npu = (device == "NPU" && m_config.batch_size.has_value() && is_seq_len_fixed);
        if (should_reshape_non_npu || should_reshape_npu) {
            reshape_model(model);
        }

        ov::CompiledModel compiled_model;
        if (device == "NPU" && model->is_dynamic()) {
            OPENVINO_ASSERT(m_config.max_length.has_value(), "The parameter max_length is not set");

            bool is_padding_on_left = m_config.padding_side.has_value() && m_config.padding_side.value() == "left";
            if (is_padding_on_left && is_seq_len_fixed &&
                config.pooling_type != TextEmbeddingPipeline::PoolingType::MEAN) {
                OPENVINO_THROW("Padding on left is only supported for the MEAN pooling type");
            }

            auto kv_pos = ov::genai::utils::get_kv_axes_pos(model);
            utils::KVDesc kv_desc;
            std::tie(compiled_model, kv_desc) =
                utils::compile_decoder_for_npu_text_embedding(model, properties, kv_pos, m_config);

            auto post_model = create_post_model(model, m_config, m_config.max_length.value());
            auto post_compiled_model = core.compile_model(post_model, "CPU");
            m_post_request = post_compiled_model.create_infer_request();
        } else {
            model = apply_postprocessing(model, m_config);
            compiled_model = core.compile_model(model, device, properties);
        }

        utils::print_compiled_model_properties(compiled_model, "text embedding model");
        m_request = compiled_model.create_infer_request();
    };

    EmbeddingResults embed_documents(const std::vector<std::string>& texts) {
        start_embed_documents_async(texts);
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

private:
    Tokenizer m_tokenizer;
    InferRequest m_request;
    InferRequest m_post_request;
    Config m_config;
    AnyMap m_tokenization_params;
    std::optional<size_t> m_max_position_embeddings;
    ov::Tensor m_attention_mask;

    void reshape_model(std::shared_ptr<Model>& model) {
        ov::PartialShape target_shape{ov::Dimension::dynamic(), ov::Dimension::dynamic()};

        if (m_config.batch_size.has_value()) {
            target_shape[0] = ov::Dimension(*m_config.batch_size);
        }

        if (m_config.max_length.has_value()) {
            if (m_max_position_embeddings.has_value() && *m_config.max_length > *m_max_position_embeddings) {
                std::stringstream message;
                message << "max_length is set to " << *m_config.max_length
                        << " which is greater than models max_position_embeddings (" << *m_max_position_embeddings
                        << ")."
                        << " Some models may fail with such configuration."
                        << " Remove max_position_embeddings from config.json to silence this warning.";
                GENAI_WARN(message.str());
            }

            if (m_config.pad_to_max_length.has_value() && *m_config.pad_to_max_length) {
                target_shape[1] = ov::Dimension(*m_config.max_length);
            } else {
                target_shape[1] = ov::Dimension{1, static_cast<int64_t>(*m_config.max_length)};
            }
        }

        std::map<std::string, ov::PartialShape> input_name_to_shape;
        input_name_to_shape["input_ids"] = target_shape;
        input_name_to_shape["attention_mask"] = target_shape;

        if (has_token_type_ids_input(model->inputs())) {
            input_name_to_shape["token_type_ids"] = target_shape;
        }

        model->reshape(input_name_to_shape);
    }

    ov::Tensor post_model_infer(ov::Tensor input) {
        if (m_post_request) {
            m_post_request.set_tensor("embedding_hidden_state", input);

            auto attention_mask_tensor = m_post_request.get_tensor("attention_mask");
            std::copy_n(m_attention_mask.data<int64_t>(),
                    m_attention_mask.get_size(),
                    attention_mask_tensor.data<int64_t>());

            if (m_attention_mask.get_size() < attention_mask_tensor.get_size()) {
                std::fill_n(attention_mask_tensor.data<int64_t>() + m_attention_mask.get_size(),
                            attention_mask_tensor.get_size() - m_attention_mask.get_size(),
                            0);
            }

            m_post_request.infer();
            return m_post_request.get_tensor("last_hidden_state");
        }

        return input;
    }

    void start_embed_async(std::vector<std::string>& texts) {
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
        if (has_token_type_ids_input(m_request.get_compiled_model().inputs())) {
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
    const auto& plugin_properties = remove_config_properties(properties);

    m_impl = std::make_unique<TextEmbeddingPipelineImpl>(models_path, device, Config(properties), plugin_properties);
};

EmbeddingResults TextEmbeddingPipeline::embed_documents(const std::vector<std::string>& texts) {
    return m_impl->embed_documents(texts);
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

TextEmbeddingPipeline::~TextEmbeddingPipeline() = default;

}  // namespace genai
}  // namespace ov
