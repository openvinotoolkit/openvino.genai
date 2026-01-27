// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/text_rerank_pipeline.hpp"

#include <fstream>

#include "debug_utils.hpp"
#include "json_utils.hpp"
#include "lora/helper.hpp"
#include "openvino/core/except.hpp"
#include "openvino/genai/lora_adapter.hpp"
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
    properties_copy.erase(padding_side.name());

    return properties_copy;
}

struct ModelMetadata {
    std::optional<std::string> model_type;
    bool is_qwen3 = false;
};
ModelMetadata read_model_type(const std::filesystem::path& models_path) {
    ModelMetadata info;
    const std::filesystem::path& json_path = models_path / "config.json";
    if (!std::filesystem::exists(json_path)) {
        return info;
    }

    using ov::genai::utils::read_json_param;

    std::ifstream f(json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '", json_path);

    nlohmann::json data = nlohmann::json::parse(f);

    read_json_param(data, "model_type", info.model_type);
    if (info.model_type.has_value()) {
        info.is_qwen3 = (info.model_type.value() == "qwen3");
    }
    return info;
}

bool has_input(const std::shared_ptr<Model>& model, const std::string& name) {
    const auto& inputs = model->inputs();
    return std::any_of(inputs.begin(), inputs.end(), [&](const auto& input) {
        return input.get_any_name() == name;
    });
}

std::shared_ptr<Model> apply_postprocessing(std::shared_ptr<Model> model) {
    PartialShape output_shape = model->get_output_partial_shape(0);

    ov::preprocess::PrePostProcessor processor(model);

    processor.output().postprocess().custom(
        [&output_shape](const ov::Output<ov::Node>& node) -> std::shared_ptr<ov::Node> {
            if (output_shape[1] == 1) {
                return std::make_shared<op::v0::Sigmoid>(node);
            }

            const auto softmax = std::make_shared<op::v8::Softmax>(node, 1);

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
};

void TextRerankPipeline::Config::validate() const {
    if (max_length.has_value()) {
        OPENVINO_ASSERT(max_length.value() > 0, "max_length should be greater than 0");
    }

    OPENVINO_ASSERT(top_n > 0, "top_n should be greater than 0");
}

class TextRerankPipeline::TextRerankPipelineImpl {
public:
    TextRerankPipelineImpl(const std::filesystem::path& models_path,
                           const std::string& device,
                           const Config& config,
                           const ov::AnyMap& properties = {})
        : m_config{config},
          m_models_path{models_path} {
        m_config.validate();
        const auto model_info = read_model_type(models_path);
        const bool is_qwen3 = model_info.is_qwen3;

        if (m_config.max_length) {
            m_tokenization_params.insert({max_length.name(), *m_config.max_length});
        }

        if (m_config.pad_to_max_length) {
            m_tokenization_params.insert({pad_to_max_length.name(), *m_config.pad_to_max_length});
        }

        if (m_config.padding_side) {
            m_tokenization_params.insert({padding_side.name(), *m_config.padding_side});
        }

        m_tokenizer = Tokenizer(models_path, ov::genai::add_second_input(!is_qwen3));

        ov::Core core = utils::singleton_core();

        auto model = core.read_model(models_path / "openvino_model.xml", {}, properties);

        auto filtered_properties = extract_adapters_from_properties(properties, &m_adapters);
        if (m_adapters.has_value()) {
            setup_lora(model, device);
        }

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

        const ov::AnyMap& compile_props = *filtered_properties;
        m_compiled_model = core.compile_model(model, device, compile_props);

        utils::print_compiled_model_properties(m_compiled_model, "text rerank model");
        m_request = m_compiled_model.create_infer_request();

        if (m_adapters.has_value() && m_adapter_controller.has_value()) {
            m_adapter_controller->apply(m_request, *m_adapters);
        }
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

    void set_adapters(const std::optional<AdapterConfig>& adapters) {
        OPENVINO_ASSERT(m_adapter_controller.has_value(),
                        "Cannot set adapters: pipeline was not constructed with LoRA support. "
                        "Provide adapters in the constructor properties to enable LoRA.");
        if (adapters.has_value()) {
            m_adapter_controller->apply(m_request, *adapters);
        } else {
            m_adapter_controller->apply(m_request, std::nullopt);
        }
    }
    bool has_adapters() const {
        return m_adapters.has_value();
    }
private:
    Tokenizer m_tokenizer;
    ov::CompiledModel m_compiled_model;
    InferRequest m_request;
    Config m_config;
    AnyMap m_tokenization_params;
    std::filesystem::path m_models_path;
    bool m_has_position_ids = false;
    bool m_has_beam_idx = false;
    std::optional<AdapterConfig> m_adapters;
    std::optional<AdapterController> m_adapter_controller;

    /**
     * @brief Setup LoRA adapters for the model
     * 
     * Uses the tensor name prefix from AdapterConfig if set by user,
     * otherwise uses empty string (no prefix stripping).
     * 
     * If your LoRA adapter was trained with PEFT and has "base_model.model" prefix,
     * set it explicitly: adapter_config.set_tensor_name_prefix("base_model.model")
     */
    void setup_lora(std::shared_ptr<Model>& model, const std::string& device) {
        OPENVINO_ASSERT(m_adapters.has_value(), "setup_lora called without adapters");

        // Use user-specified prefix, or empty string if not set
        if (!m_adapters->get_tensor_name_prefix().has_value()) {
            m_adapters->set_tensor_name_prefix("");
        }

        m_adapter_controller = AdapterController(model, *m_adapters, device);
    }

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

void TextRerankPipeline::set_adapters(const std::optional<AdapterConfig>& adapters) {
    m_impl->set_adapters(adapters);
}
bool TextRerankPipeline::has_adapters() const {
    return m_impl->has_adapters();
}
TextRerankPipeline::~TextRerankPipeline() = default;

}  // namespace genai
}  // namespace ov
