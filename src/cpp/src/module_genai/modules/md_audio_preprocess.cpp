// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_audio_preprocess.hpp"

#include <chrono>
#include <thread>

#include "module_genai/module_factory.hpp"
#include "module_genai/utils/tensor_utils.hpp"

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(AudioPreprocessModule);

void AudioPreprocessModule::print_static_config() {
    std::cout << R"(
  audio_preprocessor:           # Module Name
    type: "AudioPreprocessModule"
    device: "CPU"               # Optional, default to CPU
    description: "Audio preprocessing."
    inputs:
      - name: "audio"           # [optional]
        type: "OVTensor"        # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "audios"          # [Optional] multiple audios
        type: "VecOVTensor"     # Support DataType: [VecOVTensor]
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "input_features"  # Output port name.
        type: "OVTensor"        # Support DataType: [OVTensor]
      - name: "feature_attention_mask"  # Output port name
        type: "OVTensor"                # Support DataType: [OVTensor]   
      - name: "vec_input_features"  # Output port name.
        type: "VecOVTensor"        # Support DataType: [VecOVTensor]
      - name: "vec_feature_attention_mask"  # Output port name
        type: "VecOVTensor"                # Support DataType: [VecOVTensor]     
    params:
      model_path: "models_path"
    )" << std::endl;
}

AudioPreprocessModule::AudioPreprocessModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    std::string model_path = desc->get_full_path(desc->params["model_path"]);

    m_feature_extractor_ptr = std::make_shared<WhisperFeatureExtractor>(model_path);
    OPENVINO_ASSERT(m_feature_extractor_ptr != nullptr, "Failed to create WhisperFeatureExtractor with model path: " + model_path);
}

AudioPreprocessModule::~AudioPreprocessModule() {}

void AudioPreprocessModule::preprocess_audio(const bool& has_audios_input) {
    std::vector<ov::Tensor> audio_tensors;
    if (has_audios_input) {
        audio_tensors = get_input("audios").as<std::vector<ov::Tensor>>();
    } else {
        audio_tensors.push_back(get_input("audio").as<ov::Tensor>());
    }

    ov::TensorVector vec_input_features;
    ov::TensorVector vec_attention_masks;
    for (const auto& tensor : audio_tensors) {
        auto outputs = m_feature_extractor_ptr->extract(tensor, 16000, true);
        vec_input_features.push_back(std::move(outputs.input_features));
        if (outputs.attention_mask.has_value()) {
            vec_attention_masks.push_back(std::move(outputs.attention_mask.value()));
        }
    }

    if (has_audios_input) {
        this->outputs["input_features"].dt_type = DataType::VecOVTensor;
        this->outputs["input_features"].data = vec_input_features;
        this->outputs["feature_attention_mask"].dt_type = DataType::VecOVTensor;
        this->outputs["feature_attention_mask"].data = vec_attention_masks;
    } else {
        this->outputs["input_features"].dt_type = DataType::OVTensor;
        this->outputs["input_features"].data = vec_input_features[0];
        OPENVINO_ASSERT(!vec_attention_masks.empty(),
                        "AudioPreprocessModule: attention mask generation failed or was disabled, "
                        "but a single-audio attention mask output is required.");
        this->outputs["feature_attention_mask"].dt_type = DataType::OVTensor;
        this->outputs["feature_attention_mask"].data = vec_attention_masks[0];
    }
}

void AudioPreprocessModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();

    bool has_audios_input = exists_input("audios");
    bool has_audio_input = exists_input("audio");
    bool has_audio = has_audios_input || has_audio_input;
    if (has_audio) {
        OPENVINO_ASSERT(
            !(has_audios_input && has_audio_input),
            "AudioPreprocessModule: Both 'audio' and 'audios' inputs exist. Please provide only one of them.");
    }

    if (has_audios_input || has_audio_input) {
        preprocess_audio(has_audios_input);
    } else {
        OPENVINO_THROW("AudioPreprocessModule[" + module_desc->name +
                       "]: No valid input found. Please provide one of the following inputs: 'audio', 'audios'.");
    }
}

}  // namespace module
}  // namespace genai
}  // namespace ov
