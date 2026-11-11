// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "openvino/genai/video_generation/generation_config.hpp"
#include "openvino/genai/video_generation/text2video_pipeline.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

class VideoToVideoPipelineImpl;

/**
 * Video to video pipeline: transforms an input video into an output video, e.g. video super-resolution with FlashVSR
 * (one-step diffusion, x4 upscaling).
 *
 * The video is processed in chunks with the model states (key / value cache of the transformer, memories of the
 * decoder) carried from chunk to chunk, so it can be fed incrementally with start_stream() / push_frames() /
 * end_stream(), or at once with generate().
 */
class OPENVINO_GENAI_EXPORTS VideoToVideoPipeline {
public:
    /**
     * Initializes the pipeline from a folder with models (model_index.json and the model subdirectories).
     * @param models_path A models path to read models and config files from
     */
    explicit VideoToVideoPipeline(const std::filesystem::path& models_path);

    /**
     * Initializes the pipeline from a folder with models and compiles the models.
     * @param models_path A models path to read models and config files from
     * @param device A single device used for all models
     * @param properties Properties passed to 'compile_model'
     */
    VideoToVideoPipeline(const std::filesystem::path& models_path,
                         const std::string& device,
                         const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    VideoToVideoPipeline(const std::filesystem::path& models_path,
                         const std::string& device,
                         Properties&&... properties)
        : VideoToVideoPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    /**
     * Returns the default generation config.
     */
    const VideoGenerationConfig& get_generation_config() const;

    /**
     * Sets the default generation config.
     */
    void set_generation_config(const VideoGenerationConfig& generation_config);

    /**
     * Compiles the models for a given device.
     */
    void compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<void, Properties...> compile(const std::string& device, Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * Transforms a whole video.
     * @param video Input frames, uint8 tensor [num_frames, height, width, 3] (NHWC, RGB).
     * @param properties Generation parameters (see VideoGenerationConfig): 'generator' for the initial noise,
     *        'sparse_attention_ratio', 'color_correction'.
     * @returns VideoGenerationResult with the video tensor [1, num_frames, out_height, out_width, 3] uint8.
     */
    VideoGenerationResult generate(const ov::Tensor& video, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<VideoGenerationResult, Properties...> generate(const ov::Tensor& video,
                                                                                  Properties&&... properties) {
        return generate(video, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * Starts processing a video given frame by frame. Resets the model states.
     * @param properties Generation parameters (see generate()).
     */
    void start_stream(const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<void, Properties...> start_stream(Properties&&... properties) {
        return start_stream(ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * Adds input frames to the started stream and processes every chunk that has enough input frames.
     * @param frames uint8 tensor [num_frames, height, width, 3] or [height, width, 3]; all frames of a stream must
     *        have the same size.
     * @returns The output frames that became available, uint8 tensor [n, out_height, out_width, 3] (n may be 0).
     */
    ov::Tensor push_frames(const ov::Tensor& frames);

    /**
     * Finishes the stream: processes the remaining input frames (the last frame is repeated to complete the last
     * chunk).
     * @returns The remaining output frames, uint8 tensor [n, out_height, out_width, 3]. Over the whole stream the
     *          number of output frames equals the number of input frames.
     */
    ov::Tensor end_stream();

    VideoGenerationPerfMetrics get_performance_metrics();

    ~VideoToVideoPipeline();

private:
    std::shared_ptr<VideoToVideoPipelineImpl> m_impl;
    float m_load_time_ms = 0.0f;
};

}  // namespace ov::genai
