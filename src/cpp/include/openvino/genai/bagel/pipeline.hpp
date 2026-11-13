// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

struct BagelImageGenerationConfig {
    // Output side for text-to-image; image editing keeps the VAE resolution of the input image
    size_t size = 512;
    size_t num_inference_steps = 50;
    float timestep_shift = 3.0f;
    float cfg_text_scale = 4.0f;
    // Image editing only
    float cfg_img_scale = 2.0f;
    // Defaults to (0.4, 1.0] for text-to-image and (0.0, 1.0] for image editing
    std::optional<std::pair<float, float>> cfg_interval;
    float cfg_renorm_min = 0.0f;
    // Initial noise matches numpy.random.RandomState(seed).randn
    uint32_t seed = 0;
};

struct BagelPerfMetrics {
    // compile_model per model, plus first-use compile of each NPUW_BAGEL static variant on NPU
    std::vector<std::pair<std::string, float>> compile_ms;
    // Per call, variant compile excluded
    std::vector<std::pair<std::string, float>> infer_ms;
    // Sum of first calls of new variants (compile + one inference)
    float first_call_ms = 0.0f;
};

/**
 * BAGEL-7B-MoT pipeline for visual question answering, text-to-image and image editing.
 * Images are u8 RGB tensors with [1, H, W, 3] or [H, W, 3] layout.
 * On NPU the language, vision and VAE models run through NPUW_BAGEL; text embeddings run on CPU.
 */
class OPENVINO_GENAI_EXPORTS BagelPipeline {
public:
    /**
     * @param models_dir directory with the BAGEL IRs and bagel_ov_config.json, or an optimum-intel export
     *                   (openvino_*_model.xml with config.json)
     * @param device inference device
     * @param tokenizer_dir directory with openvino_tokenizer.xml / openvino_detokenizer.xml; models_dir if empty
     * @param properties applied on top of the default per-model configuration
     */
    BagelPipeline(const std::filesystem::path& models_dir,
                  const std::string& device,
                  const std::filesystem::path& tokenizer_dir = {},
                  const ov::AnyMap& properties = {});
    ~BagelPipeline();

    /**
     * @brief Greedy text generation, optionally about an image
     * @param prompt user prompt
     * @param image optional u8 RGB image
     * @param max_new_tokens maximum number of generated tokens
     * @param system_prompt optional system prompt prefilled before the image and the prompt
     * @return generated text
     */
    std::string generate_text(const std::string& prompt,
                              const std::optional<ov::Tensor>& image = std::nullopt,
                              size_t max_new_tokens = 256,
                              const std::string& system_prompt = {});

    /**
     * @brief Text-to-image generation
     * @param prompt image description
     * @param config generation parameters
     * @return u8 [1, size, size, 3] RGB image
     */
    ov::Tensor generate_image(const std::string& prompt, const BagelImageGenerationConfig& config = {});

    /**
     * @brief Image editing
     * @param image u8 RGB image to edit
     * @param prompt edit instruction
     * @param config generation parameters, `size` is ignored
     * @return u8 [1, H, W, 3] RGB image at the VAE resolution of the input image
     */
    ov::Tensor edit_image(const ov::Tensor& image,
                          const std::string& prompt,
                          const BagelImageGenerationConfig& config = {});

    /// @brief Compile and inference timings accumulated since construction or the last reset_perf_metrics()
    const BagelPerfMetrics& get_perf_metrics() const;
    void reset_perf_metrics();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ov::genai
