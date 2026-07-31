// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/qwen3_text_encoder.hpp"

#include <fstream>
#include <cstring>

#include "json_utils.hpp"
#include "lora/helper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

std::filesystem::path get_tokenizer_path_by_text_encoder(const std::filesystem::path& text_encoder_path);

Qwen3TextEncoder::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "hidden_size", hidden_size);
    read_json_param(data, "num_hidden_layers", num_hidden_layers);
    read_json_param(data, "hidden_states_layers", hidden_states_layers);
}

Qwen3TextEncoder::Qwen3TextEncoder(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json"),
      m_tokenizer(get_tokenizer_path_by_text_encoder(root_dir)) {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

Qwen3TextEncoder::Qwen3TextEncoder(const std::filesystem::path& root_dir,
                                   const std::string& device,
                                   const ov::AnyMap& properties)
    : Qwen3TextEncoder(root_dir) {
    compile(device, properties);
}

Qwen3TextEncoder::Qwen3TextEncoder(const Qwen3TextEncoder&) = default;

std::shared_ptr<Qwen3TextEncoder> Qwen3TextEncoder::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "Qwen3TextEncoder must have exactly one of m_model or m_request initialized");

    std::shared_ptr<Qwen3TextEncoder> cloned = std::make_shared<Qwen3TextEncoder>(*this);

    if (m_model) {
        cloned->m_model = m_model->clone();
    } else {
        cloned->m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

Qwen3TextEncoder& Qwen3TextEncoder::reshape(const int batch_size, const int max_sequence_length) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");
    OPENVINO_ASSERT(batch_size >= 1 && batch_size <= 2,
                    "Qwen3TextEncoder supports batch_size 1 or 2 (for classifier-free guidance)");

    std::map<std::string, ov::PartialShape> name_to_shape;
    for (auto&& input : m_model->inputs()) {
        std::string input_name = input.get_any_name();
        name_to_shape[input_name] = input.get_partial_shape();
        if (input_name == "input_ids" || input_name == "attention_mask") {
            name_to_shape[input_name] = {batch_size, max_sequence_length};
        }
    }

    m_model->reshape(name_to_shape);
    return *this;
}

Qwen3TextEncoder& Qwen3TextEncoder::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
    if (adapters) {
        adapters->set_tensor_name_prefix(adapters->get_tensor_name_prefix().value_or("text_encoder"));
        m_adapter_controller = AdapterController(m_model, *adapters, device);
    }
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *filtered_properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Qwen3 text encoder model");
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

ov::Tensor Qwen3TextEncoder::infer(const std::string& pos_prompt, const std::string& neg_prompt, const bool do_classifier_free_guidance, const int& max_sequence_length) {
    OPENVINO_ASSERT(m_request, "Qwen3 text encoder model must be compiled first. Cannot infer non-compiled model");

    const size_t text_embedding_batch_size = do_classifier_free_guidance ? 2 : 1;
    const size_t seq_len = static_cast<size_t>(max_sequence_length);

    // Use element type from the compiled model's inputs
    const ov::element::Type input_type = m_request.get_compiled_model().input("input_ids").get_element_type();
    const int64_t pad_token_id = m_tokenizer.get_pad_token_id();

    auto tokenize_prompt = [&](const std::string& prompt, size_t batch_idx,
                               ov::Tensor& input_ids, ov::Tensor& attention_mask) {
        // Apply chat template as diffusers does
        ChatHistory history = {{{"role", "user"}, {"content", prompt}}};
        JsonContainer extra_context = JsonContainer::from_json_string("{\"enable_thinking\": false}");
        std::string formatted_prompt = m_tokenizer.apply_chat_template(history, true, {}, std::nullopt, extra_context);

        auto tokenizer_output = m_tokenizer.encode(formatted_prompt);
        ov::Tensor input_ids_token = tokenizer_output.input_ids;
        ov::Tensor attention_mask_token = tokenizer_output.attention_mask;

        const size_t token_len = input_ids_token.get_shape()[1];
        const size_t actual_len = std::min(token_len, seq_len);

        if (input_type == ov::element::i32) {
            int32_t* ids_row = input_ids.data<int32_t>() + batch_idx * seq_len;
            int32_t* mask_row = attention_mask.data<int32_t>() + batch_idx * seq_len;
            std::fill_n(ids_row, seq_len, static_cast<int32_t>(pad_token_id));
            std::fill_n(mask_row, seq_len, static_cast<int32_t>(0));
            std::copy_n(input_ids_token.data<int64_t>(), actual_len, ids_row);
            std::copy_n(attention_mask_token.data<int64_t>(), actual_len, mask_row);
        } else {
            int64_t* ids_row = input_ids.data<int64_t>() + batch_idx * seq_len;
            int64_t* mask_row = attention_mask.data<int64_t>() + batch_idx * seq_len;
            std::fill_n(ids_row, seq_len, pad_token_id);
            std::fill_n(mask_row, seq_len, static_cast<int64_t>(0));
            std::copy_n(input_ids_token.data<int64_t>(), actual_len, ids_row);
            std::copy_n(attention_mask_token.data<int64_t>(), actual_len, mask_row);
        }
    };

    // Prepare batched input tensors
    ov::Tensor input_ids(input_type, {text_embedding_batch_size, seq_len});
    ov::Tensor attention_mask(input_type, {text_embedding_batch_size, seq_len});

    size_t current_batch_idx = 0;
    if (do_classifier_free_guidance) {
        tokenize_prompt(neg_prompt, current_batch_idx, input_ids, attention_mask);
        ++current_batch_idx;
    }
    tokenize_prompt(pos_prompt, current_batch_idx, input_ids, attention_mask);

    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.infer();

    // Gather hidden states from selected layers and concatenate along channel dimension
    const size_t num_layers = m_config.hidden_states_layers.size();
    const size_t hidden_size = m_config.hidden_size;
    const size_t output_dim = num_layers * hidden_size;

    // Output shape: (text_embedding_batch_size, seq_len, num_layers * hidden_size)
    ov::Tensor result(ov::element::f32, {text_embedding_batch_size, seq_len, output_dim});
    float* result_data = result.data<float>();

    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const size_t layer_num = m_config.hidden_states_layers[layer_idx];
        const std::string output_name = "hidden_states." + std::to_string(layer_num);

        ov::Tensor hidden_state = m_request.get_tensor(output_name);
        const float* hs_data = hidden_state.data<float>();

        // hidden_state shape: (text_embedding_batch_size, seq_len, hidden_size)
        // result layout: (batch, seq_len, [layer0_hidden | layer1_hidden | ...])
        for (size_t b = 0; b < text_embedding_batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                std::memcpy(result_data + (b * seq_len + s) * output_dim + layer_idx * hidden_size,
                            hs_data + (b * seq_len + s) * hidden_size,
                            hidden_size * sizeof(float));
            }
        }
    }

    return result;
}

void Qwen3TextEncoder::set_adapters(const std::optional<AdapterConfig>& adapters) {
    OPENVINO_ASSERT(m_request, "Text encoder model must be compiled first");
    if (adapters) {
        m_adapter_controller.apply(m_request, *adapters);
    }
}

const Qwen3TextEncoder::Config& Qwen3TextEncoder::get_config() const {
    return m_config;
}

}  // namespace genai
}  // namespace ov
