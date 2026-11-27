// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

/**
 * @brief Activation applied to the classifier logits.
 */
enum class ClassificationActivation {
    NONE = 0,     // Raw logits.
    SOFTMAX = 1,  // Single-label classification: scores of an item sum to 1.
    SIGMOID = 2,  // Multi-label classification or a single relevance logit: scores are independent.
};

/**
 * @brief Result of classifying one item.
 */
struct OPENVINO_GENAI_EXPORTS ClassificationResult {
    // Raw logits [1, num_labels] in label ID order.
    ov::Tensor logits;
    // Logits after the activation. scores[0][i] is the score of label ID i.
    std::vector<std::vector<float>> scores;
};

/**
 * @brief Classifies a prompt with optional images using a vision-language model exported as
 * `*ForSequenceClassification` (text-classification task).
 *
 * The scores are computed from the logits of the last prompt token. No text is generated, and no chat template or
 * special tokens are added: the caller builds the complete model prompt.
 */
class OPENVINO_GENAI_EXPORTS ClassificationPipeline {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        /**
         * @brief Activation applied to the logits. If not set, it is derived from `problem_type` in config.json:
         * NONE for `regression`, SIGMOID for `multi_label_classification` or a single label, SOFTMAX otherwise.
         */
        std::optional<ClassificationActivation> classification_activation;

        Config() = default;

        /**
         * @param properties e.g. {{"classification_activation", ClassificationActivation::SIGMOID}}
         */
        explicit Config(const ov::AnyMap& properties);
    };

    /**
     * @param models_path Directory with the exported model, tokenizer and config.json
     * @param device Device
     * @param config Pipeline configuration
     * @param properties Optional plugin properties for ov::Core::compile_model()
     */
    ClassificationPipeline(const std::filesystem::path& models_path,
                           const std::string& device,
                           const Config& config,
                           const ov::AnyMap& properties = {});

    /**
     * @param models_path Directory with the exported model, tokenizer and config.json
     * @param device Device
     * @param properties Plugin properties for ov::Core::compile_model() and configuration properties
     * (ov::genai::classification_activation)
     */
    ClassificationPipeline(const std::filesystem::path& models_path,
                           const std::string& device,
                           const ov::AnyMap& properties = {});

    /**
     * @param models_path Directory with the exported model, tokenizer and config.json
     * @param device Device
     * @param properties Plugin and configuration properties, e.g.
     * ov::genai::classification_activation(ClassificationActivation::SIGMOID)
     */
    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    ClassificationPipeline(const std::filesystem::path& models_path,
                           const std::string& device,
                           Properties&&... properties)
        : ClassificationPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    /**
     * @brief Classifies a prompt.
     *
     * @param text Complete model prompt. Images are referenced by native or zero-based <ov_genai_image_N> tags.
     * @param images RGB uint8 tensors [H, W, 3] or [1, H, W, 3], one image per tensor.
     * @return Logits and activated scores, both in label ID order.
     */
    ClassificationResult classify(const std::string& text, const std::vector<ov::Tensor>& images = {});

    ~ClassificationPipeline();

private:
    class ClassificationPipelineImpl;
    std::unique_ptr<ClassificationPipelineImpl> m_impl;
};

/**
 * @brief Activation applied to the classifier logits.
 */
static constexpr ov::Property<ClassificationActivation> classification_activation{"classification_activation"};

}  // namespace genai
}  // namespace ov
