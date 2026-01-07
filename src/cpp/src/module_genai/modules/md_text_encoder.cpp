// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_text_encoder.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "tokenizer/tokenizer_impl.hpp"

#include <chrono>
#include <thread>

namespace ov {
namespace genai {
namespace module {

const std::string NATIVE_TAG = "<|vision_start|><|image_pad|><|vision_end|>";
const std::string NATIVE_VIDEO_TAG = "<|vision_start|><|video_pad|><|vision_end|>";

void TextEncoderModule::print_static_config() {
    std::cout << R"(
  prompt_encoder:                       # Module Name
    type: "TextEncoderModule"
    description: "Encode prompt to prompt ids."
    device: "GPU"
    inputs:
      - name: "prompt"
        type: "String"            # [Optional] Support DataType: [String]
        source: "ParentModuleName.OutputPortName"
      - name: "prompts"
        type: "VecString"         # [Optional] Support DataType: [VecString]
        source: "ParentModuleName.OutputPortName"
      - name: "encoded_image"
        type: "OVTensor"          # [Optional] Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "encoded_images"
        type: "VecOVTensor"       # [Optional] Support DataType: [VecOVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "source_size"
        type: "VecInt"            # [Optional] Support DataType: [VecInt]
        source: "ParentModuleName.OutputPortName"
      - name: "source_sizes"
        type: "VecVecInt"         # [Optional] Support DataType: [VecVecInt]
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "input_ids"
        type: "OVTensor"     # Support DataType: [OVTensor, OVRemoteTensor]
      - name: "mask"
        type: "OVTensor"     # Support DataType: [OVTensor, OVRemoteTensor]
      - name: "images_sequence"
        type: "VecInt"             # Support DataType: [VecInt]
    params:
      model_path: "models/text_encoder.xml"  # Optional. OpenVINO IR
    )" << std::endl;
}

TextEncoderModule::TextEncoderModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    if (!initialize()) {
        GENAI_ERR("Failed to initiate TextEncoderModule");
    }
}

TextEncoderModule::~TextEncoderModule() {}

bool TextEncoderModule::initialize() {
    const auto& params = module_desc->params;
    auto it_path = params.find("model_path");
    if (it_path == params.end()) {
        std::cerr << "TextEncoderModule[" << module_desc->name << "]: 'model_path' not found in params" << std::endl;
        return false;
    }
    
    std::filesystem::path tokenizer_path = module_desc->get_full_path(it_path->second);

    m_tokenizer_impl = std::make_shared<Tokenizer::TokenizerImpl>(tokenizer_path, m_tokenization_params);
    OPENVINO_ASSERT(m_tokenizer_impl->m_ireq_queue_tokenizer != nullptr, std::string("Load tokenizer model fail: ") + tokenizer_path.c_str());
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(tokenizer_path, "config.json");
    m_processor_config = utils::from_config_json_if_exists<ProcessorConfig>(tokenizer_path, "preprocessor_config.json");
    m_merge_length = std::pow(m_processor_config.merge_size, 2);
    return true;
}

void TextEncoderModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    
    prepare_inputs();
    std::vector<std::string> m_prompts = {};
    if (this->inputs.find("prompts") != this->inputs.end()) {
        m_prompts = this->inputs["prompts"].data.as<std::vector<std::string>>();
    }
    if (this->inputs.find("prompt") != this->inputs.end()) {
        std::string single_prompt = this->inputs["prompt"].data.as<std::string>();
        m_prompts.insert(m_prompts.begin(), single_prompt);
    }
    std::vector<ov::Tensor> encoded_images = {};
    std::vector<std::vector<int>> source_sizes = {};
    bool has_encoded_image = false;
    if (this->inputs.find("encoded_image") != this->inputs.end()) {
        ov::Tensor encoded_image = this->inputs["encoded_image"].data.as<ov::Tensor>();
        encoded_images.push_back(encoded_image);
        has_encoded_image = true;
    }
    if (this->inputs.find("encoded_images") != this->inputs.end()) {
        encoded_images = this->inputs["encoded_images"].data.as<std::vector<ov::Tensor>>();
        has_encoded_image = true;
    }
    if (this->inputs.find("source_size") != this->inputs.end()) {
        source_sizes.push_back(this->inputs["source_size"].data.as<std::vector<int>>());
    }
    if (this->inputs.find("source_sizes") != this->inputs.end()) {
        source_sizes = this->inputs["source_sizes"].data.as<std::vector<std::vector<int>>>();
    }

    if (has_encoded_image) {
        VLMModelType model_type = to_vlm_model_type(module_desc->model_type);
        if (model_type != VLMModelType::QWEN2_VL && model_type != VLMModelType::QWEN2_5_VL) {
            GENAI_ERR("TextEncoderModule[" + module_desc->name + "]: Unsupported model type: " + module_desc->model_type);
            return;
        }
    }

    auto [encoded, images_sequence] = run(m_prompts, encoded_images, source_sizes, has_encoded_image);

    this->outputs["input_ids"].data = encoded.input_ids;
    this->outputs["mask"].data = encoded.attention_mask;
    if (images_sequence.size() > 0) {
        this->outputs["images_sequence"].data = images_sequence;
    }
}

std::pair<TokenizedInputs, std::vector<int>> TextEncoderModule::run(const std::vector<std::string>& prompts, 
                        const std::vector<ov::Tensor>& encoded_images,
                        const std::vector<std::vector<int>>& source_sizes,
                        bool has_encoded_image) {
    OPENVINO_ASSERT(m_tokenizer_impl, "TextEncoderModule is not initialized. Call initialize() first.");
    check_arguments(m_tokenization_params, {ov::genai::add_special_tokens.name(),
                                            ov::genai::max_length.name(),
                                            ov::genai::pad_to_max_length.name(),
                                            ov::genai::padding_side.name()});
    
    if (has_encoded_image) {
        std::vector<std::string> unified_prompts = {};
        std::vector<int> g_images_sequence = {};
        for (const auto &prompt : prompts) {
            // Hard code base image/video id and encoded images/videos
            auto [unified_prompt, images_sequence, videos_sequence] = normalize_prompt(prompt, 0, 0, encoded_images, {}, source_sizes);
            ChatHistory history = {{{"role", "user"}, {"content", unified_prompt}}};
            auto templated_prompt = m_tokenizer_impl->apply_chat_template(history, true, {}, std::nullopt, std::nullopt);
            unified_prompts.push_back(templated_prompt);
            g_images_sequence = std::vector<int>(images_sequence.begin(), images_sequence.end());
        }
        return {m_tokenizer_impl->encode(unified_prompts, m_tokenization_params), g_images_sequence};
    } else {
        return {m_tokenizer_impl->encode(prompts, m_tokenization_params), {}};
    }
}

NormalizedPrompt TextEncoderModule::normalize_prompt(const std::string& prompt,
                                      size_t base_image_id,
                                      size_t base_video_id,
                                      const std::vector<ov::Tensor>& encoded_images,
                                      const std::vector<ov::Tensor>& encoded_videos,
                                      const std::vector<std::vector<int>>& source_sizes) {
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG, base_image_id, encoded_images.size());
    std::vector<std::array<size_t, 3>> images_grid_thw;
    images_grid_thw.reserve(encoded_images.size());
    for (const auto& source_size : source_sizes) {
        size_t grid_t = 1;
        size_t grid_h = source_size[0];
        size_t grid_w = source_size[1];
        images_grid_thw.push_back({grid_t, grid_h, grid_w});
    }

    for (size_t new_image_id : images_sequence) {
        auto [grid_t, grid_h, grid_w] = images_grid_thw.at(new_image_id - base_image_id);
        const size_t num_image_pad_tokens = calc_tokens_num(grid_t, grid_h, grid_w);

        std::string expanded_tag;
        expanded_tag.reserve(m_vlm_config.vision_start_token.length() +
                             m_vlm_config.image_pad_token.length() * num_image_pad_tokens +
                             m_vlm_config.vision_end_token.length());
        expanded_tag.append(m_vlm_config.vision_start_token);
        for (int i = 0; i < num_image_pad_tokens; ++i) {
            expanded_tag.append(m_vlm_config.image_pad_token);
        }
        expanded_tag.append(m_vlm_config.vision_end_token);

        unified_prompt.replace(unified_prompt.find(NATIVE_TAG), NATIVE_TAG.length(), expanded_tag);
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

std::pair<std::string, std::vector<size_t>> TextEncoderModule::normalize(
    const std::string& prompt,
    const std::string& native_tag,
    const std::string& automatic_tag,
    size_t base_id,
    size_t n_images) {
    size_t pos = prompt.find(native_tag);
    auto [image_prompt, image_sequence] = universal_to_native(prompt, [&](std::ostream& os, size_t) {
        os << automatic_tag;
    });
    if (!image_sequence.empty()) {
        OPENVINO_ASSERT(pos == std::string::npos, "Prompt can contain only one type of image tags.");
        verify_ids(image_sequence, base_id, n_images);
        return {std::move(image_prompt), std::move(image_sequence)};
    }
    // Restore ids from native tags
    while (pos != std::string::npos) {
        image_sequence.push_back(base_id + image_sequence.size());
        pos = prompt.find(native_tag, pos + native_tag.length());
    }
    if (!image_sequence.empty()) {
        OPENVINO_ASSERT(image_sequence.size() == n_images, "The number of native image tags and provided images must match because it's ambiguous which image should be ignored.");
        return {std::move(image_prompt), std::move(image_sequence)};
    }
    // Prepend automatic tags
    std::stringstream stream;
    for (size_t relative_id = 0; relative_id < n_images; relative_id++) {
        image_sequence.push_back(base_id + relative_id);
        stream << automatic_tag;
    }
    stream << prompt;
    return {stream.str(), std::move(image_sequence)};
}

size_t TextEncoderModule::calc_tokens_num(size_t grid_t, size_t grid_h, size_t grid_w) const {
    return grid_t * grid_h * grid_w / m_merge_length;
}

}  // namespace module
}  // namespace genai
}  // namespace ov
