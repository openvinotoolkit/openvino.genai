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
    // Owned raw logits of shape [1, num_labels] in label ID order, before activation.
    ov::Tensor logits;
    // Scores after the activation, one per label in label ID order: scores[0][i] belongs to label ID i.
    std::vector<std::vector<float>> scores;
};

/**
 * @brief Classification pipeline for vision-language models with a sequence-classification head.
 *
 * Uses models exported with the text-classification task as vision, text embeddings and language
 * model parts. The language model must end in a classification head, not a token-generation head.
 *
 * The pipeline pools the logits of the last prompt token and applies the configured activation.
 * It does not generate text, apply a chat template or add extra special tokens.
 * Model-specific prompt construction and input preparation are the caller's responsibility.
 */
class OPENVINO_GENAI_EXPORTS ClassificationPipeline {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        /**
         * @brief Activation applied to the logits. When not set, it is derived from config.json:
         * NONE for `problem_type == regression`, SIGMOID for `multi_label_classification` or a single label,
         * SOFTMAX otherwise.
         */
        std::optional<ClassificationActivation> activation;

        /**
         * @brief Constructs classification pipeline configuration
         */
        Config() = default;

        /**
         * @brief Constructs classification pipeline configuration
         *
         * @param properties configuration options, e.g. {{"activation", ClassificationActivation::SIGMOID}}
         */
        explicit Config(const ov::AnyMap& properties);
    };

    /**
     * @brief Constructs a pipeline from xml/bin files, tokenizer and configuration in the same directory.
     *
     * @param models_path Directory with the exported model, tokenizer and config.json
     * @param device Device
     * @param config Pipeline configuration
     * @param properties Optional plugin properties to pass to ov::Core::compile_model().
     */
    ClassificationPipeline(const std::filesystem::path& models_path,
                           const std::string& device,
                           const Config& config,
                           const ov::AnyMap& properties = {});

    /**
     * @brief Constructs a pipeline from xml/bin files, tokenizer and configuration in the same directory.
     *
     * @param models_path Directory with the exported model, tokenizer and config.json
     * @param device Device
     * @param properties Optional plugin properties to pass to ov::Core::compile_model() mixed with
     * configuration properties (ov::genai::activation).
     */
    ClassificationPipeline(const std::filesystem::path& models_path,
                           const std::string& device,
                           const ov::AnyMap& properties = {});

    /**
     * @brief Constructs a pipeline using configuration properties.
     *
     * @param models_path Directory with the exported model, tokenizer and config.json
     * @param device Device
     * @param properties Configuration and plugin properties, e.g.
     * ov::genai::activation(ClassificationActivation::SIGMOID)
     */
    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    ClassificationPipeline(const std::filesystem::path& models_path,
                           const std::string& device,
                           Properties&&... properties)
        : ClassificationPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    /**
     * @brief Classifies one already-formatted prompt.
     *
     * @param text Complete model prompt, including any suffix required by the model. No chat template or extra
     * special tokens are added. Images may use native tags or zero-based <ov_genai_image_N> tags.
     * @param images Optional RGB uint8 tensors of shape [H, W, 3] or [N, H, W, 3].
     * The number, order and content of the images must match the model's input requirements.
     * @return Owned logits [1, num_labels] and activated scores [1][num_labels], both in label ID order.
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
static constexpr ov::Property<ClassificationActivation> activation{"activation"};

}  // namespace genai
}  // namespace ov
