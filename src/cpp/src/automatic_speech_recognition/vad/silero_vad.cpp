// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/vad/silero_vad.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils.hpp"

namespace {

using ov::genai::silero_vad::CONTEXT_SAMPLES;
using ov::genai::silero_vad::STATE_SIZE;
using ov::genai::silero_vad::WINDOW_SAMPLES;

constexpr size_t FRAME_INPUT_SAMPLES = CONTEXT_SAMPLES + WINDOW_SAMPLES;
const ov::Shape STATE_SHAPE{1, 1, STATE_SIZE};

void validate_io_contract(const std::shared_ptr<ov::Model>& model, const std::filesystem::path& model_path) {
    OPENVINO_ASSERT(model->inputs().size() == 3,
                    "Silero VAD model ",
                    model_path,
                    " must have 3 inputs, got ",
                    model->inputs().size());
    for (const char* name : {"h", "c"}) {
        model->input(name);
    }
    for (const char* name : {"speech_probs", "hn", "cn"}) {
        model->output(name);
    }
    const ov::PartialShape frames = model->input("input").get_partial_shape();
    OPENVINO_ASSERT(frames.rank().is_static() && frames.rank().get_length() == 2 && frames[0].is_dynamic(),
                    "Silero VAD model ",
                    model_path,
                    " must have a rank-2 'input' with a dynamic sequence dimension, got ",
                    frames);
}

void validate(const ov::genai::SileroVADConfig& config) {
    OPENVINO_ASSERT(std::isfinite(config.threshold) && config.threshold > 0.0 && config.threshold < 1.0,
                    "SileroVADConfig.threshold must be in (0, 1), got ",
                    config.threshold);
    if (config.neg_threshold.has_value()) {
        const double value = *config.neg_threshold;
        OPENVINO_ASSERT(std::isfinite(value) && value > 0.0 && value <= config.threshold,
                        "SileroVADConfig.neg_threshold must be in (0, threshold], got ",
                        value,
                        " with threshold ",
                        config.threshold);
    }
    OPENVINO_ASSERT(!std::isnan(config.max_speech_duration_s) && config.max_speech_duration_s > 0.0,
                    "SileroVADConfig.max_speech_duration_s must be positive or infinity, got ",
                    config.max_speech_duration_s);
}

ov::genai::SileroVADConfig get_config(const ov::AnyMap& properties) {
    ov::genai::SileroVADConfig config;
    for (const auto& [name, value] : properties) {
        OPENVINO_ASSERT(name == ov::genai::silero_vad_config.name(),
                        "Unsupported Silero VAD property '",
                        name,
                        "'");
        config = value.as<ov::genai::SileroVADConfig>();
    }
    validate(config);
    return config;
}

}  // namespace

namespace ov::genai::silero_vad {

SileroVAD::SileroVAD(const std::filesystem::path& models_path,
                     const std::string& device,
                     const ov::AnyMap& properties) {
    const std::filesystem::path model_path = models_path / SEQUENCE_MODEL_FILE;
    OPENVINO_ASSERT(std::filesystem::is_regular_file(model_path),
                    "Silero VAD model file ",
                    SEQUENCE_MODEL_FILE,
                    " not found in ",
                    models_path);

    ov::Core& core = utils::singleton_core();
    const std::shared_ptr<ov::Model> model = core.read_model(model_path);
    validate_io_contract(model, model_path);

    ov::AnyMap compile_properties = properties;
    compile_properties.emplace(ov::hint::inference_precision.name(), ov::element::f32);
    ov::CompiledModel compiled = core.compile_model(model, device, compile_properties);
    utils::print_compiled_model_properties(compiled, "Silero VAD model");
    m_request = compiled.create_infer_request();
}

std::vector<SpeechSegment> SileroVAD::get_speech_segments(const std::vector<float>& audio,
                                                          const ov::AnyMap& properties) {
    const SileroVADConfig config = get_config(properties);
    return segments_from_probabilities(get_speech_probabilities(audio), audio.size(), config);
}

// Adapted from frame_blocks() and SileroVADSequence.audio_forward() in Silero VAD v6.3
// (MIT License, Copyright (c) 2020-present Silero Team):
// https://github.com/snakers4/silero-vad/blob/63d4a2a7bfce8cd9bb39ed8fa0afcb5236d73e01/src/silero_vad/sequence_vad.py#L34-L149
std::vector<float> SileroVAD::get_speech_probabilities(const std::vector<float>& audio) {
    const size_t frame_count = (audio.size() + WINDOW_SAMPLES - 1) / WINDOW_SAMPLES;
    std::vector<float> probabilities;
    probabilities.reserve(frame_count);

    ov::Tensor hidden(ov::element::f32, STATE_SHAPE);
    ov::Tensor cell(ov::element::f32, STATE_SHAPE);
    std::fill_n(hidden.data<float>(), hidden.get_size(), 0.0f);
    std::fill_n(cell.data<float>(), cell.get_size(), 0.0f);
    std::array<float, CONTEXT_SAMPLES> context{};

    for (size_t first_frame = 0; first_frame < frame_count; first_frame += MAX_BLOCK_FRAMES) {
        const size_t block_frames = std::min(MAX_BLOCK_FRAMES, frame_count - first_frame);
        ov::Tensor input(ov::element::f32, {block_frames, FRAME_INPUT_SAMPLES});
        float* rows = input.data<float>();
        for (size_t frame = 0; frame < block_frames; ++frame) {
            float* row = rows + frame * FRAME_INPUT_SAMPLES;
            std::copy(context.begin(), context.end(), row);
            float* window = row + CONTEXT_SAMPLES;
            const size_t begin = (first_frame + frame) * WINDOW_SAMPLES;
            const size_t available = std::min(WINDOW_SAMPLES, audio.size() - begin);
            std::copy_n(audio.data() + begin, available, window);
            std::fill(window + available, window + WINDOW_SAMPLES, 0.0f);
            std::copy(window + WINDOW_SAMPLES - CONTEXT_SAMPLES, window + WINDOW_SAMPLES, context.begin());
        }

        m_request.set_tensor("input", input);
        m_request.set_tensor("h", hidden);
        m_request.set_tensor("c", cell);
        m_request.infer();

        const ov::Tensor speech_probs = m_request.get_tensor("speech_probs");
        OPENVINO_ASSERT(speech_probs.get_size() == block_frames,
                        "Silero VAD model returned ",
                        speech_probs.get_size(),
                        " probabilities for ",
                        block_frames,
                        " frames");
        const float* values = speech_probs.data<const float>();
        probabilities.insert(probabilities.end(), values, values + block_frames);

        m_request.get_tensor("hn").copy_to(hidden);
        m_request.get_tensor("cn").copy_to(cell);
    }
    return probabilities;
}

}  // namespace ov::genai::silero_vad
