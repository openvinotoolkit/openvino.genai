// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/ltx2_duration_head.hpp"

#include <algorithm>
#include <cmath>

#include "utils.hpp"

using namespace ov::genai;

namespace {

// Python's round(): halves to even, which std::nearbyint does in the default rounding mode
int64_t round_half_to_even(double value) {
    return static_cast<int64_t>(std::nearbyint(value));
}

}  // namespace

LTX2DurationHead::LTX2DurationHead(const std::filesystem::path& root_dir) {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

LTX2DurationHead::LTX2DurationHead(const std::filesystem::path& root_dir,
                                   const std::string& device,
                                   const ov::AnyMap& properties)
    : LTX2DurationHead(root_dir) {
    compile(device, properties);
}

LTX2DurationHead::LTX2DurationHead(const LTX2DurationHead&) = default;

LTX2DurationHead LTX2DurationHead::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "LTX2DurationHead must have exactly one of m_model or m_request initialized");

    LTX2DurationHead cloned = *this;

    if (m_model) {
        cloned.m_model = m_model->clone();
    } else {
        cloned.m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

LTX2DurationHead& LTX2DurationHead::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "LTX2 duration head model");
    m_request = compiled_model.create_infer_request();
    m_model.reset();

    return *this;
}

LTX2DurationHead& LTX2DurationHead::reshape(int64_t batch_size) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    std::map<std::string, ov::PartialShape> name_to_shape;
    for (auto&& input : m_model->inputs()) {
        ov::PartialShape shape = input.get_partial_shape();
        shape[0] = batch_size;
        name_to_shape[input.get_any_name()] = shape;
    }
    m_model->reshape(name_to_shape);

    return *this;
}

float LTX2DurationHead::infer(const ov::Tensor& video_tokens, const ov::Tensor& audio_tokens) {
    OPENVINO_ASSERT(m_request, "Duration head model must be compiled first. Cannot infer non-compiled model");

    m_request.set_tensor("video_tokens", video_tokens);
    m_request.set_tensor("audio_tokens", audio_tokens);
    m_request.infer();

    const ov::Tensor duration = m_request.get_tensor("duration");
    OPENVINO_ASSERT(duration.get_size() >= 1, "Duration head returned an empty prediction");
    return duration.data<const float>()[0];
}

size_t LTX2DurationHead::duration_to_num_frames(float seconds,
                                                float frame_rate,
                                                size_t temporal_compression_ratio,
                                                float min_seconds,
                                                float max_seconds) {
    OPENVINO_ASSERT(temporal_compression_ratio > 0, "The VAE's temporal compression ratio must be positive");
    const auto ratio = static_cast<int64_t>(temporal_compression_ratio);
    // Floored at 1 so the grid arithmetic below cannot go negative
    const int64_t min_frames = std::max<int64_t>(1, round_half_to_even(min_seconds * frame_rate));
    const int64_t max_frames = round_half_to_even(max_seconds * frame_rate);
    const int64_t clamped_frames =
        std::max(min_frames, std::min(round_half_to_even(static_cast<double>(seconds) * frame_rate), max_frames));

    int64_t num_frames = (clamped_frames - 1) / ratio * ratio + 1;
    if (num_frames < min_frames) {
        // Flooring undershot the lower bound; the next grid point up is the in-bounds choice
        num_frames += ratio;
        OPENVINO_ASSERT(num_frames <= max_frames,
                        "Duration bounds [",
                        min_seconds,
                        "s, ",
                        max_seconds,
                        "s] at ",
                        frame_rate,
                        " fps admit no frame count on the VAE's temporal grid (k * ",
                        ratio,
                        " + 1)");
    }
    return static_cast<size_t>(num_frames);
}
