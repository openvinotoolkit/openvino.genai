// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/qwen3_vl_for_conditional_generation.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <type_traits>

#include "json_utils.hpp"
#include "lora/helper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

namespace {

// The exporter stores the Qwen3-VL processor (and its tokenizer) in 'processor', while models converted with
// the diffusers layout keep it in 'tokenizer'.
std::filesystem::path get_qwen_image21_tokenizer_path(const std::filesystem::path& text_encoder_path) {
    const std::filesystem::path root_dir = text_encoder_path.parent_path();
    for (const std::string& subfolder : {"processor", "tokenizer"}) {
        const std::filesystem::path candidate = root_dir / subfolder;
        if (std::filesystem::exists(candidate / "openvino_tokenizer.xml")) {
            return candidate;
        }
    }
    OPENVINO_THROW("Failed to find 'openvino_tokenizer.xml' neither in '",
                   root_dir / "processor", "' nor in '", root_dir / "tokenizer", "'");
}

}  // namespace

// The prompt is built as a raw template string instead of going through the chat template: the two tokenize
// differently and the checkpoint expects this one.
const std::string Qwen3VLForConditionalGeneration::SYSTEM_PREFIX =
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";

const std::string Qwen3VLForConditionalGeneration::PROMPT_TEMPLATE =
    Qwen3VLForConditionalGeneration::SYSTEM_PREFIX + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";

Qwen3VLForConditionalGeneration::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "hidden_size", hidden_size);
}

Qwen3VLForConditionalGeneration::Qwen3VLForConditionalGeneration(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json"),
      m_tokenizer(get_qwen_image21_tokenizer_path(root_dir)),
      m_system_prefix_length(
          m_tokenizer.encode(SYSTEM_PREFIX, ov::genai::add_special_tokens(false)).input_ids.get_shape()[1]) {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

Qwen3VLForConditionalGeneration::Qwen3VLForConditionalGeneration(const std::filesystem::path& root_dir,
                                                                 const std::string& device,
                                                                 const ov::AnyMap& properties)
    : Qwen3VLForConditionalGeneration(root_dir) {
    compile(device, properties);
}

Qwen3VLForConditionalGeneration::Qwen3VLForConditionalGeneration(const Qwen3VLForConditionalGeneration&) = default;

std::shared_ptr<Qwen3VLForConditionalGeneration> Qwen3VLForConditionalGeneration::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "Qwen3VLForConditionalGeneration must have exactly one of m_model or m_request initialized");

    std::shared_ptr<Qwen3VLForConditionalGeneration> cloned =
        std::make_shared<Qwen3VLForConditionalGeneration>(*this);

    if (m_model) {
        cloned->m_model = m_model->clone();
    } else {
        cloned->m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

Qwen3VLForConditionalGeneration& Qwen3VLForConditionalGeneration::compile(const std::string& device,
                                                                          const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
    if (adapters) {
        adapters->set_tensor_name_prefix(adapters->get_tensor_name_prefix().value_or("text_encoder"));
        m_adapter_controller = AdapterController(m_model, *adapters, device);
    }
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *filtered_properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "QwenImage 2.1 text encoder model");
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

ov::Tensor Qwen3VLForConditionalGeneration::infer(const std::string& prompt, const int max_sequence_length) {
    OPENVINO_ASSERT(m_request, "QwenImage 2.1 text encoder model must be compiled first. Cannot infer non-compiled model");
    OPENVINO_ASSERT(max_sequence_length > 0, "'max_sequence_length' must be positive, got ", max_sequence_length);

    std::string formatted_prompt = PROMPT_TEMPLATE;
    const std::string placeholder = "{}";
    const size_t placeholder_pos = formatted_prompt.find(placeholder);
    OPENVINO_ASSERT(placeholder_pos != std::string::npos, "Prompt template must contain '{}'");
    formatted_prompt.replace(placeholder_pos, placeholder.length(), prompt);

    const ov::Tensor token_ids =
        m_tokenizer.encode(formatted_prompt, ov::genai::add_special_tokens(false)).input_ids;
    const size_t token_count = token_ids.get_shape()[1];

    OPENVINO_ASSERT(token_count > m_system_prefix_length,
                    "Tokenized prompt length (", token_count, ") must be greater than the system prefix length (",
                    m_system_prefix_length, ")");
    const size_t prompt_length = token_count - m_system_prefix_length;
    OPENVINO_ASSERT(prompt_length <= static_cast<size_t>(max_sequence_length),
                    "Tokenized prompt length (", prompt_length, ") exceeds 'max_sequence_length' (",
                    max_sequence_length, ")");

    const ov::element::Type input_type = m_request.get_compiled_model().input("input_ids").get_element_type();
    ov::Tensor input_ids(input_type, {1, token_count});
    ov::Tensor attention_mask(input_type, {1, token_count});

    if (input_type == ov::element::i32) {
        std::copy_n(token_ids.data<const int64_t>(), token_count, input_ids.data<int32_t>());
        std::fill_n(attention_mask.data<int32_t>(), token_count, int32_t{1});
    } else {
        std::copy_n(token_ids.data<const int64_t>(), token_count, input_ids.data<int64_t>());
        std::fill_n(attention_mask.data<int64_t>(), token_count, int64_t{1});
    }

    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.infer();

    const ov::Tensor hidden_states = m_request.get_output_tensor();
    const size_t hidden_size = hidden_states.get_shape()[2];

    ov::Tensor prompt_embeds(ov::element::f32, {1, prompt_length, hidden_size});
    std::memcpy(prompt_embeds.data<float>(),
                hidden_states.data<const float>() + m_system_prefix_length * hidden_size,
                prompt_length * hidden_size * sizeof(float));

    return prompt_embeds;
}

void Qwen3VLForConditionalGeneration::set_adapters(const std::optional<AdapterConfig>& adapters) {
    OPENVINO_ASSERT(m_request, "Text encoder model must be compiled first");
    if (adapters) {
        m_adapter_controller.apply(m_request, *adapters);
    }
}

const Qwen3VLForConditionalGeneration::Config& Qwen3VLForConditionalGeneration::get_config() const {
    return m_config;
}

}  // namespace genai
}  // namespace ov
