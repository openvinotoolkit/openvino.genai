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
    // Output side for text-to-image; 0 selects 512 capped to the largest side the model supports.
    // Image editing keeps the VAE resolution of the input image
    size_t size = 0;
    size_t num_inference_steps = 50;
    float timestep_shift = 3.0f;
    float cfg_text_scale = 4.0f;
    // Image editing only; applied on top of text guidance, so it needs cfg_text_scale > 1
    float cfg_img_scale = 2.0f;
    // Guidance is applied for timesteps t with low < t <= high; defaults to (0.4, 1.0] for text-to-image and
    // (0.0, 1.0] for image editing
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
     * @param models_path directory with the BAGEL IRs and bagel_ov_config.json, or an optimum-intel export
     *                    (openvino_*_model.xml with config.json)
     * @param device inference device
     * @param tokenizer_path directory with openvino_tokenizer.xml / openvino_detokenizer.xml; models_path if empty
     * @param properties applied on top of the default per-model configuration
     */
    BagelPipeline(const std::filesystem::path& models_path,
                  const std::string& device,
                  const std::filesystem::path& tokenizer_path = {},
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
     * @brief Greedy text generation about several images
     * @param prompt user prompt
     * @param images u8 RGB images, prefilled in order before the prompt
     * @param max_new_tokens maximum number of generated tokens
     * @param system_prompt optional system prompt prefilled before the images and the prompt
     * @return generated text
     */
    std::string generate_text(const std::string& prompt,
                              const std::vector<ov::Tensor>& images,
                              size_t max_new_tokens = 256,
                              const std::string& system_prompt = {});

    /**
     * @brief Text-to-image generation
     * @param prompt image description
     * @param config generation parameters; defaults if not set
     * @return u8 [1, size, size, 3] RGB image
     */
    ov::Tensor generate_image(const std::string& prompt,
                              const std::optional<BagelImageGenerationConfig>& config = std::nullopt);

    /**
     * @brief Image editing
     * @param image u8 RGB image to edit
     * @param prompt edit instruction
     * @param config generation parameters, `size` is ignored; defaults if not set
     * @return u8 [1, H, W, 3] RGB image at the VAE resolution of the input image
     */
    ov::Tensor edit_image(const ov::Tensor& image,
                          const std::string& prompt,
                          const std::optional<BagelImageGenerationConfig>& config = std::nullopt);

    /**
     * @brief Image editing or composition from several images
     * @param images u8 RGB input images, in the order the prompt refers to them
     * @param prompt edit instruction
     * @param config generation parameters, `size` is ignored; defaults if not set
     * @return u8 [1, H, W, 3] RGB image at the VAE resolution of the last input image
     */
    ov::Tensor edit_image(const std::vector<ov::Tensor>& images,
                          const std::string& prompt,
                          const std::optional<BagelImageGenerationConfig>& config = std::nullopt);

    /**
     * @brief Predicts the next frames of a video, as a world model. Frame k is generated from a window of the
     * input frames' count: it then replaces the oldest frame of the window. Generated frames join the context as VAE
     * latents only, without a decode / re-encode, and all of them are decoded at the end, in one VAE decoder call
     * (one call per frame on NPU)
     * @param frames u8 RGB frames, oldest first
     * @param prompt instruction for the next frames, e.g. the task and the action of a robot
     * @param num_frames number of frames to generate
     * @param config generation parameters, `size` is ignored; frame k starts from the noise of seed + k
     * @return u8 [num_frames, H, W, 3] RGB frames at the VAE resolution of the last input frame
     */
    ov::Tensor generate_frames(const std::vector<ov::Tensor>& frames,
                               const std::string& prompt,
                               size_t num_frames,
                               const std::optional<BagelImageGenerationConfig>& config = std::nullopt);

    /// @brief Snapshot of compile and inference timings since construction or the last reset_perf_metrics();
    /// safe to call while another thread generates
    BagelPerfMetrics get_perf_metrics() const;
    void reset_perf_metrics();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ov::genai
