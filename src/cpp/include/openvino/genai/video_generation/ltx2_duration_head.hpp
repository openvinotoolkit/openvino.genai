// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

#include "openvino/genai/visibility.hpp"

namespace ov::genai {

/**
 * LTX-2.5 duration head: predicts a video length in seconds from the text connector outputs, so that a
 * pipeline can choose the number of frames for a prompt.
 */
class OPENVINO_GENAI_EXPORTS LTX2DurationHead {
public:
    explicit LTX2DurationHead(const std::filesystem::path& root_dir);

    LTX2DurationHead(const std::filesystem::path& root_dir,
                     const std::string& device,
                     const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    LTX2DurationHead(const std::filesystem::path& root_dir, const std::string& device, Properties&&... properties)
        : LTX2DurationHead(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    LTX2DurationHead(const LTX2DurationHead&);

    LTX2DurationHead clone();

    LTX2DurationHead& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<LTX2DurationHead&, Properties...> compile(const std::string& device,
                                                                             Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    LTX2DurationHead& reshape(int64_t batch_size);

    /**
     * @brief Predicts the duration of a video in seconds
     * @param video_tokens The video text embeddings of the connectors [B, S, D]
     * @param audio_tokens The audio text embeddings of the connectors [B, S, D_audio]
     * @returns The duration of the first video in seconds
     */
    float infer(const ov::Tensor& video_tokens, const ov::Tensor& audio_tokens);

    /**
     * @brief Converts a duration into a number of frames on the causal VAE grid (k * temporal_compression_ratio + 1),
     * as diffusers' LTX2DurationHead.predict_num_frames does: the frame count is clamped to
     * [round(min_seconds * frame_rate), round(max_seconds * frame_rate)] first and snapped down to the grid, and up
     * by one grid step if that falls below the minimum. Throws if no grid point lies within the bounds.
     */
    static size_t duration_to_num_frames(float seconds,
                                         float frame_rate,
                                         size_t temporal_compression_ratio,
                                         float min_seconds = 1.0f,
                                         float max_seconds = 20.0f);

private:
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
};

}  // namespace ov::genai
