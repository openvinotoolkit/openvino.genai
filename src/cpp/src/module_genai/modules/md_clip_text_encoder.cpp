// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_clip_text_encoder.hpp"

#include "module_genai/module_factory.hpp"
#include "utils.hpp"
#include "module_genai/utils/tensor_utils.hpp"
#include <iostream>
#include "openvino/genai/tokenizer.hpp"
#include "tokenizer/tokenizer_impl.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(ClipTextEncoderModule);

void ClipTextEncoderModule::print_static_config() {
    std::cout << R"(
  clip_text_encoder:                       # Module Name
    type: "ClipTextEncoderModule"
    description: "Encode positive prompt and negative prompt"
    device: "GPU"
    inputs:
      - name: "prompt"
        type: "String"            # [Optional] Support DataType: [String]
        source: "ParentModuleName.OutputPortName"
      - name: "prompts"
        type: "VecString"         # [Optional] Support DataType: [VecString]
        source: "ParentModuleName.OutputPortName"
      - name: "negative_prompt"
        type: "String"            # [Optional] Support DataType: [String]
        source: "ParentModuleName.OutputPortName"
      - name: "negative_prompts"
        type: "VecString"         # [Optional] Support DataType: [VecString]
        source: "ParentModuleName.OutputPortName"
      - name: "guidance_scale"
        type: "Float"             # [Optional] Support DataType: [Float]
        source: "ParentModuleName.OutputPortName"
      - name: "max_sequence_length"
        type: "Int"               # [Optional] Support DataType: [Int]
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "prompt_embeds"
        type: "VecOVTensor"       # Support DataType: [VecOVTensor]
      - name: "negative_prompt_embeds"
        type: "VecOVTensor"       # [Optional] Support DataType: [VecOVTensor]
    params:
      model_path: "model_dir/"  # model directory
    )" << std::endl;
}

ClipTextEncoderModule::ClipTextEncoderModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    if (!initialize()) {
        GENAI_ERR("Failed to initialize ClipTextEncoderModule");
    }
}

ClipTextEncoderModule::~ClipTextEncoderModule() {}

bool ClipTextEncoderModule::initialize() {
    const auto& params = module_desc->params;
    auto it_path = params.find("model_path");
    if (it_path == params.end()) {
        GENAI_ERR("ClipTextEncoderModule[" + module_desc->name + "]: 'model_path' not found in params")
        return false;
    }
    
    std::filesystem::path root_dir = module_desc->get_full_path(it_path->second);
    std::string device = module_desc->device.empty() ? "CPU" : module_desc->device;
    ov::AnyMap properties = {};

    m_encoder_config = utils::from_config_json_if_exists<TransformerConfig>(
        root_dir, "text_encoder/config.json");

    try {
        auto tokenizer_path = root_dir / "tokenizer";
        m_tokenizer_impl = std::make_shared<Tokenizer::TokenizerImpl>(tokenizer_path, properties);
    } catch (const std::exception& e) {
        GENAI_ERR("ClipTextEncoderModule[" + module_desc->name + "]: Failed to load tokenizer: " + e.what());
        return false;
    }

    try {
        auto text_encoder_path = root_dir / "text_encoder/openvino_model.xml";
        auto model = utils::singleton_core().read_model(text_encoder_path);
        auto compiled_model = utils::singleton_core().compile_model(
            model,
            device,
            ov::AnyMap{});
        m_request = compiled_model.create_infer_request();
    } catch (const std::exception& e) {
        GENAI_ERR("ClipTextEncoderModule[" + module_desc->name + "]: Failed to initiate text tokenizer: " + e.what());
        return false;
    }
    
    return true;
}

void ClipTextEncoderModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    
    prepare_inputs();
    std::vector<std::string> m_prompts = {};
    std::vector<std::string> m_negative_prompts = {};
    
    if (exists_input("prompts")) {
        m_prompts = this->inputs["prompts"].data.as<std::vector<std::string>>();
    }
    if (exists_input("prompt")) {
        std::string single_prompt = this->inputs["prompt"].data.as<std::string>();
        m_prompts.insert(m_prompts.begin(), single_prompt);
    }

    if (exists_input("negative_prompts")) {
        m_negative_prompts = this->inputs["negative_prompts"].data.as<std::vector<std::string>>();
    }
    if (exists_input("negative_prompt")) {
        std::string single_negative_prompt = this->inputs["negative_prompt"].data.as<std::string>();
        m_negative_prompts.insert(m_negative_prompts.begin(), single_negative_prompt);
    }

    if (!m_prompts.empty() && !m_negative_prompts.empty() && m_negative_prompts.size() != m_prompts.size()) {
        GENAI_ERR("Either prompts or negative_prompts size is 0, or they are both equal");
    }

    ImageGenerationConfig generation_config {};
    if (exists_input("guidance_scale")) {
        generation_config.guidance_scale = this->inputs["guidance_scale"].data.as<float>();
    } else {
        generation_config.guidance_scale = 0.0f;
    }
    if (exists_input("max_sequence_length")) {
        generation_config.max_sequence_length = this->inputs["max_sequence_length"].data.as<int>();
    } else {
        generation_config.max_sequence_length = 512;
    }

    auto [prompt_embeds, negative_prompt_embeds] = run(m_prompts, m_negative_prompts, generation_config);

    if (prompt_embeds) {
        this->outputs["prompt_embeds"].data = tensor_utils::split(prompt_embeds);
    } else {
        // Set empty vector instead of uninitialized tensor
        this->outputs["prompt_embeds"].data = std::vector<ov::Tensor>{};
    }
    if (negative_prompt_embeds) {
        this->outputs["negative_prompt_embeds"].data = tensor_utils::split(negative_prompt_embeds);
    } else {
        // When CFG is disabled (guidance_scale <= 1.0), set empty vector instead of uninitialized tensor
        this->outputs["negative_prompt_embeds"].data = std::vector<ov::Tensor>{};
    }
}

std::pair<ov::Tensor, ov::Tensor> ClipTextEncoderModule::run(
        const std::vector<std::string>& prompts,
        const std::vector<std::string>& negative_prompts,
        const ImageGenerationConfig &generation_config) {
    const bool use_cfg = do_classifier_free_guidance(generation_config.guidance_scale);
    
    ov::Tensor prompt_embed, negative_prompt_embed;
    if (!prompts.empty()) {
        prompt_embed = encode_prompt(prompts, generation_config);
    }
    if (use_cfg) {
        negative_prompt_embed = encode_prompt(negative_prompts, generation_config);
    }
    return {prompt_embed, negative_prompt_embed};
}

bool ClipTextEncoderModule::do_classifier_free_guidance(float guidance_scale) {
    return guidance_scale > 1.0;
}

ov::Tensor ClipTextEncoderModule::encode_prompt(
        const std::vector<std::string>& prompts,
        const ImageGenerationConfig &generation_config) {
    std::vector<std::string> templated_prompts = {};
    templated_prompts.reserve(prompts.size());
    for (const auto& s : prompts) {
        ChatHistory history({{{"role", "user"}, {"content", s}}});
        bool add_generation_prompt = true;
        auto templated_s = m_tokenizer_impl->apply_chat_template(history, add_generation_prompt, {}, std::nullopt, std::nullopt);
        templated_prompts.push_back(templated_s);
    }

    ov::AnyMap tokenization_params = {};
    auto text_inputs = m_tokenizer_impl->encode(templated_prompts, tokenization_params);
    m_request.set_tensor("input_ids", text_inputs.input_ids);
    m_request.set_tensor("attention_mask", text_inputs.attention_mask);
    m_request.infer();

    size_t idx = m_encoder_config.num_hidden_layers;
    ov::Tensor prompt_embed = m_request.get_output_tensor(idx);
    ov::Tensor prompt_embed_out = ov::Tensor(prompt_embed.get_element_type(), prompt_embed.get_shape());
    prompt_embed.copy_to(prompt_embed_out);
    return prompt_embed_out;
}

}  // namespace module
}  // namespace genai
}  // namespace ov
