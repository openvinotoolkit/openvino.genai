// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/classification/classification_pipeline.hpp"

#include "classification/activation.hpp"
#include "classification/vlm_classifier_backend.hpp"
#include "openvino/core/except.hpp"
#include "utils.hpp"

namespace {

using namespace ov::genai;

std::unique_ptr<ClassifierBackend> create_backend(const std::filesystem::path& models_path,
                                                  const std::string& device,
                                                  const ov::AnyMap& properties,
                                                  const ClassifierConfig& config) {
    return std::make_unique<VLMClassifierBackend>(models_path, device, properties, config);
}

ov::AnyMap remove_config_properties(const ov::AnyMap& properties) {
    ov::AnyMap plugin_properties = properties;
    plugin_properties.erase(activation.name());
    return plugin_properties;
}

}  // namespace

namespace ov {
namespace genai {

ClassificationPipeline::Config::Config(const ov::AnyMap& properties) {
    utils::read_anymap_param(properties, ov::genai::activation.name(), activation);
}

class ClassificationPipeline::ClassificationPipelineImpl {
public:
    ClassificationPipelineImpl(const std::filesystem::path& models_path,
                               const std::string& device,
                               const Config& config,
                               const ov::AnyMap& properties) {
        const ClassifierConfig classifier_config{models_path};
        m_backend = create_backend(models_path, device, properties, classifier_config);
        m_activation = config.activation.value_or(
            default_activation(classifier_config.problem_type.value_or(""), m_backend->num_labels()));
    }

    ClassificationResult classify(const ClassificationInputs& inputs) {
        ClassificationResult result;
        result.logits = m_backend->infer(inputs);
        result.scores = apply_activation(result.logits, m_activation);
        return result;
    }

private:
    std::unique_ptr<ClassifierBackend> m_backend;
    ClassificationActivation m_activation;
};

ClassificationPipeline::ClassificationPipeline(const std::filesystem::path& models_path,
                                               const std::string& device,
                                               const Config& config,
                                               const ov::AnyMap& properties)
    : m_impl{std::make_unique<ClassificationPipelineImpl>(models_path, device, config, properties)} {}

ClassificationPipeline::ClassificationPipeline(const std::filesystem::path& models_path,
                                               const std::string& device,
                                               const ov::AnyMap& properties)
    : m_impl{std::make_unique<ClassificationPipelineImpl>(models_path,
                                                          device,
                                                          Config(properties),
                                                          remove_config_properties(properties))} {}

ClassificationResult ClassificationPipeline::classify(const std::string& text, const std::vector<ov::Tensor>& images) {
    return m_impl->classify({text, images});
}

ClassificationPipeline::~ClassificationPipeline() = default;

}  // namespace genai
}  // namespace ov
