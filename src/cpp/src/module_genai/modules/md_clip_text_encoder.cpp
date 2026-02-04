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
#include "module_genai/utils/profiler.hpp"

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
      - name: "num_images_per_prompt"
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

    // Determine text encoder XML path
    auto text_encoder_path = root_dir / "text_encoder/openvino_model.xml";
    DiffusionModelType model_type = to_diffusion_model_type(module_desc->model_type);
    if (model_type == DiffusionModelType::WAN_2_1) {
        text_encoder_path = root_dir / "text_encoder/text_encoder.xml";
    }

    // Create cache key hash from model XML content and device
    std::size_t cache_key = compute_cache_key(text_encoder_path, device);

    // Check if resources already exist in cache
    bool is_cached = ClipTextEncoderResourceCache::instance().exists(cache_key);
    if (is_cached) {
        GENAI_INFO("ClipTextEncoderModule: Reusing cached model for: " + root_dir.string() + " on device: " + device);
    }

    // Get or create shared resources
    m_shared_resources = ClipTextEncoderResourceCache::instance().get_or_create(
        cache_key,
        [this, &root_dir, &device, &text_encoder_path, model_type]() -> std::shared_ptr<ClipTextEncoderSharedResources> {
            GENAI_INFO("ClipTextEncoderModule: Loading model from: " + root_dir.string() + " to device: " + device);

            auto resources = std::make_shared<ClipTextEncoderSharedResources>();
            resources->model_type = model_type;

            ov::AnyMap properties = {};

            resources->encoder_config = utils::from_config_json_if_exists<TransformerConfig>(
                root_dir, "text_encoder/config.json");

            // Load tokenizer
            try {
                auto tokenizer_path = root_dir / "tokenizer";
                resources->tokenizer_impl = std::make_shared<Tokenizer::TokenizerImpl>(tokenizer_path, properties);
            } catch (const std::exception& e) {
                GENAI_ERR("ClipTextEncoderModule: Failed to load tokenizer: " + std::string(e.what()));
                return nullptr;
            }

            // Load and compile text encoder model
            try {
                auto model = utils::singleton_core().read_model(text_encoder_path);
                resources->compiled_model = utils::singleton_core().compile_model(
                    model,
                    device,
                    ov::AnyMap{});
            } catch (const std::exception& e) {
                GENAI_ERR("ClipTextEncoderModule: Failed to load text encoder model: " + std::string(e.what()));
                return nullptr;
            }

            resources->minja_template = std::make_shared<minja::chat_template>(
                resources->tokenizer_impl->get_chat_template(),
                resources->tokenizer_impl->m_bos_token,
                resources->tokenizer_impl->m_eos_token);

            return resources;
        });

    if (!m_shared_resources) {
        GENAI_ERR("ClipTextEncoderModule[" + module_desc->name + "]: Failed to initialize shared resources");
        return false;
    }

    // Each module instance creates its own InferRequest from the shared compiled model
    try {
        m_request = m_shared_resources->compiled_model.create_infer_request();
    } catch (const std::exception& e) {
        GENAI_ERR("ClipTextEncoderModule[" + module_desc->name + "]: Failed to create infer request: " + std::string(e.what()));
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

    if (m_shared_resources->model_type == DiffusionModelType::WAN_2_1) {
        if (m_negative_prompts.empty()) {
            for (size_t i = 0; i < m_prompts.size(); i++) {
                m_negative_prompts.push_back("");
            }
        }
        if (m_prompts.size() > 1 && m_negative_prompts.size() == 1) {
            std::string negative_prompt = m_negative_prompts[0];
            m_negative_prompts.clear();
            for (size_t i = 0; i < m_prompts.size(); i++) {
                m_negative_prompts.push_back(negative_prompt);
            }
        }
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
    if (exists_input("num_images_per_prompt")) {
        generation_config.num_images_per_prompt = this->inputs["num_images_per_prompt"].data.as<int>();
    } else {
        generation_config.num_images_per_prompt = 1;
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
        if (m_shared_resources->model_type == DiffusionModelType::WAN_2_1) {
            templated_prompts.push_back(s);
        } else {
            ChatHistory history({{{"role", "user"}, {"content", s}}});
            bool add_generation_prompt = true;
            auto templated_s = m_shared_resources->tokenizer_impl->apply_chat_template(history, add_generation_prompt, {}, std::nullopt, std::nullopt, m_shared_resources->minja_template);
            templated_prompts.push_back(templated_s);
        }
    }

    ov::AnyMap tokenization_params = {};
    if (m_shared_resources->model_type == DiffusionModelType::WAN_2_1) {
        tokenization_params["add_special_tokens"] = true;
        tokenization_params["max_length"] = generation_config.max_sequence_length;
        tokenization_params["pad_to_max_length"] = true;
    }


    auto text_inputs = m_shared_resources->tokenizer_impl->encode(templated_prompts, tokenization_params);
    if (m_shared_resources->model_type == DiffusionModelType::ZIMAGE) {
        m_request.set_tensor("input_ids", text_inputs.input_ids);
        m_request.set_tensor("attention_mask", text_inputs.attention_mask);
        {
            PROFILE(pm, "infer");
            m_request.infer();
        }

        size_t idx = m_shared_resources->encoder_config.num_hidden_layers;
        ov::Tensor prompt_embed = m_request.get_output_tensor(idx);
        ov::Tensor prompt_embed_out = ov::Tensor(prompt_embed.get_element_type(), prompt_embed.get_shape());
        prompt_embed.copy_to(prompt_embed_out);
        return prompt_embed_out;
    } else if (m_shared_resources->model_type == DiffusionModelType::WAN_2_1) {
        m_request.set_tensor("input_ids", text_inputs.input_ids);
        {
            PROFILE(pm, "infer");
            m_request.infer();
        }

        ov::Tensor prompt_embed = m_request.get_output_tensor();
        size_t batch_size = prompt_embed.get_shape()[0];
        std::vector<int64_t> seq_lens;
        auto attention_mask_data = text_inputs.attention_mask.data<int64_t>();
        for (size_t i = 0; i < batch_size; i++) {
            int64_t sum = 0;
            for (size_t j = 0; j < text_inputs.attention_mask.get_shape()[1]; j++) {
                sum += attention_mask_data[i * text_inputs.attention_mask.get_shape()[1] + j];
            }
            seq_lens.push_back(sum);
        }

        ov::Shape output_shape = prompt_embed.get_shape();
        output_shape[0] = batch_size * generation_config.num_images_per_prompt;
        output_shape[1] = generation_config.max_sequence_length;
        ov::Tensor prompt_embed_out = ov::Tensor(prompt_embed.get_element_type(), output_shape);
        auto prompt_embed_data = prompt_embed.data<float>();
        auto prompt_embed_out_data = prompt_embed_out.data<float>();
        size_t embed_size = output_shape[1] * output_shape[2];
        std::memset(prompt_embed_out_data, 0, prompt_embed_out.get_byte_size());
        for (size_t i = 0; i < generation_config.num_images_per_prompt; i++) {
            for (size_t j = 0; j < batch_size; j++) {
                size_t seq_len = seq_lens[j];
                std::memcpy(prompt_embed_out_data + (i * (embed_size * batch_size) + j * embed_size),
                            prompt_embed_data + (j * embed_size),
                            seq_len * output_shape[2] * sizeof(float));
            }
        }
        return prompt_embed_out;
    } else {
        OPENVINO_THROW("Unsupported model type in ClipTextEncoderModule: " + module_desc->model_type);
    }
}

}  // namespace module
}  // namespace genai
}  // namespace ov
