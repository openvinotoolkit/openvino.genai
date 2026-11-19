// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstring>
#include <filesystem>
#include <functional>
#include <list>
#include <map>
#include <mutex>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/pass/make_stateful.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace qwen3_tts {

// Snapshot of a decoding session's state: the state tensors and the number of frames decoded so far.
struct StreamDecoderSnapshot {
    std::vector<ov::Tensor> tensors;
    size_t frames_decoded = 0;
};

// Explicit-state ("stateful") streaming codec decoder, exported by optimum-intel as
// `openvino_codec_decoder_stream.xml`.
//
// The codec decoder is causal, so instead of re-decoding left context for every chunk, each call consumes a
// fixed block of new code frames together with the decoder's carried state (convolution histories,
// transposed-convolution tails, attention key/value cache) and returns exactly the block's new samples. Run in
// order from a zero state, the blocks reproduce the offline full decode up to float rounding.
//
// The IR has a dynamic frame count and is reshaped to a static `block_frames` at load time (NPU needs static
// shapes). The state is handled in one of two ways:
//   * CPU / GPU: `ov::pass::MakeStateful` folds the `state.X` / `new_state.X` pairs into variables, so the runtime
//     keeps the state device-resident and copies nothing.
//   * NPU: the state stays explicit graph inputs / outputs and is carried by the session on the host.
// Every stream owns an infer request (and thus its own state), so concurrent streams do not interfere.
class StreamDecoder : public std::enable_shared_from_this<StreamDecoder> {
public:
    // Frames decoded per call. Streaming granularity is one block (320 ms of audio at 4 frames), and the first
    // audio is available after one block. Blocks must be full, except for the last one of an utterance, which is
    // zero-padded (that is safe because the decoder is causal and the padded output is dropped).
    static constexpr size_t BLOCK_FRAMES = 4;

    StreamDecoder(const std::filesystem::path& model_path,
                  const std::string& device,
                  const ov::AnyMap& properties,
                  size_t num_quantizers,
                  size_t samples_per_frame)
        : m_num_quantizers(num_quantizers),
          m_samples_per_frame(samples_per_frame),
          m_explicit_state(device.find("NPU") != std::string::npos) {
        ov::Core& core = ov::genai::utils::singleton_core();
        auto model = core.read_model(model_path);

        std::map<std::string, std::string> state_pairs;
        for (const auto& input : model->inputs()) {
            const std::string name = input.get_any_name();
            if (name.rfind("state.", 0) == 0) {
                state_pairs[name] = "new_" + name;
                m_state_inputs.push_back(name);
            }
        }
        OPENVINO_ASSERT(!m_state_inputs.empty(),
                        "Streaming codec decoder '",
                        model_path.string(),
                        "' has no 'state.*' inputs; re-export the model with a recent optimum-intel.");

        model->reshape({{"audio_codes", ov::PartialShape{1, static_cast<int64_t>(num_quantizers), static_cast<int64_t>(BLOCK_FRAMES)}}});
        for (const auto& name : m_state_inputs) {
            m_state_shapes.push_back(model->input(name).get_shape());
        }

        if (!m_explicit_state) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::MakeStateful>(state_pairs);
            manager.run_passes(model);
        }

        m_compiled = core.compile_model(model, device, properties);
        ov::genai::utils::print_compiled_model_properties(m_compiled, "codec_decoder_stream");
    }

    bool uses_explicit_state() const {
        return m_explicit_state;
    }

    ov::CompiledModel& compiled_model() {
        return m_compiled;
    }

    size_t num_quantizers() const {
        return m_num_quantizers;
    }

    size_t samples_per_frame() const {
        return m_samples_per_frame;
    }

    const std::vector<std::string>& state_inputs() const {
        return m_state_inputs;
    }

    const std::vector<ov::Shape>& state_shapes() const {
        return m_state_shapes;
    }

    // Cache of decoder states after priming with voice-clone reference codes. The state depends only on the
    // reference codes, so every request that clones the same voice can restore it instead of re-decoding the
    // reference (25 calls for 100 frames). Keyed by a hash of the primed codes; keeps the last few references.
    std::shared_ptr<const StreamDecoderSnapshot> find_primed(uint64_t key) const {
        std::lock_guard<std::mutex> lock(m_primed_mutex);
        for (auto it = m_primed.begin(); it != m_primed.end(); ++it) {
            if (it->first == key) {
                m_primed.splice(m_primed.begin(), m_primed, it);  // most recently used first
                return it->second;
            }
        }
        return nullptr;
    }

    void store_primed(uint64_t key, std::shared_ptr<const StreamDecoderSnapshot> snapshot) const {
        std::lock_guard<std::mutex> lock(m_primed_mutex);
        m_primed.remove_if([key](const auto& entry) { return entry.first == key; });
        m_primed.emplace_front(key, std::move(snapshot));
        while (m_primed.size() > MAX_PRIMED_REFERENCES) {
            m_primed.pop_back();
        }
    }

    // FNV-1a over the primed frame-major codes and the frame count.
    static uint64_t primed_key(const int64_t* codes, size_t count, size_t frames) {
        uint64_t hash = 1469598103934665603ull;
        auto mix = [&hash](uint64_t value) {
            for (int i = 0; i < 8; ++i) {
                hash ^= (value >> (8 * i)) & 0xffu;
                hash *= 1099511628211ull;
            }
        };
        mix(frames);
        for (size_t i = 0; i < count; ++i) {
            mix(static_cast<uint64_t>(codes[i]));
        }
        return hash;
    }

private:
    size_t m_num_quantizers;
    size_t m_samples_per_frame;
    bool m_explicit_state;
    std::vector<std::string> m_state_inputs;
    std::vector<ov::Shape> m_state_shapes;
    ov::CompiledModel m_compiled;

    static constexpr size_t MAX_PRIMED_REFERENCES = 4;
    mutable std::mutex m_primed_mutex;
    mutable std::list<std::pair<uint64_t, std::shared_ptr<const StreamDecoderSnapshot>>> m_primed;
};

// Decoding state of ONE utterance. Not thread-safe; use from one thread at a time.
class StreamDecoderSession {
public:
    // `on_infer_ms` (optional) receives the duration of every infer() when perf reporting is enabled.
    StreamDecoderSession(std::shared_ptr<StreamDecoder> decoder, std::function<void(double)> on_infer_ms = {})
        : m_decoder(std::move(decoder)),
          m_on_infer_ms(std::move(on_infer_ms)),
          m_request(m_decoder->compiled_model().create_infer_request()),
          m_codes(ov::element::i64, ov::Shape{1, m_decoder->num_quantizers(), StreamDecoder::BLOCK_FRAMES}),
          m_position(ov::element::i64, ov::Shape{}) {
        m_request.set_tensor("audio_codes", m_codes);
        m_request.set_tensor("position", m_position);
        if (m_decoder->uses_explicit_state()) {
            const auto& names = m_decoder->state_inputs();
            for (size_t i = 0; i < names.size(); ++i) {
                m_states.emplace_back(ov::element::f32, m_decoder->state_shapes()[i]);
                m_request.set_tensor(names[i], m_states.back());
            }
        }
        reset();
    }

    static constexpr size_t block_frames() {
        return StreamDecoder::BLOCK_FRAMES;
    }

    // Back to the zero state at position 0 (start of an utterance).
    void reset() {
        m_frames_decoded = 0;
        if (m_decoder->uses_explicit_state()) {
            for (auto& state : m_states) {
                std::memset(state.data(), 0, state.get_byte_size());
            }
        } else {
            m_request.reset_state();
        }
    }

    size_t frames_decoded() const {
        return m_frames_decoded;
    }

    // Copies out / restores the decoding state (deep copies: the session keeps decoding afterwards).
    std::shared_ptr<StreamDecoderSnapshot> snapshot() {
        auto result = std::make_shared<StreamDecoderSnapshot>();
        result->frames_decoded = m_frames_decoded;
        if (m_decoder->uses_explicit_state()) {
            for (const auto& state : m_states) {
                ov::Tensor copy(state.get_element_type(), state.get_shape());
                state.copy_to(copy);
                result->tensors.push_back(copy);
            }
        } else {
            for (auto& variable : m_request.query_state()) {
                const ov::Tensor value = variable.get_state();
                ov::Tensor copy(value.get_element_type(), value.get_shape());
                value.copy_to(copy);
                result->tensors.push_back(copy);
            }
        }
        return result;
    }

    void restore(const StreamDecoderSnapshot& snapshot) {
        if (m_decoder->uses_explicit_state()) {
            OPENVINO_ASSERT(snapshot.tensors.size() == m_states.size(), "Streaming decoder snapshot does not match the model");
            for (size_t i = 0; i < m_states.size(); ++i) {
                snapshot.tensors[i].copy_to(m_states[i]);
            }
        } else {
            auto variables = m_request.query_state();
            OPENVINO_ASSERT(snapshot.tensors.size() == variables.size(), "Streaming decoder snapshot does not match the model");
            for (size_t i = 0; i < variables.size(); ++i) {
                variables[i].set_state(snapshot.tensors[i]);
            }
        }
        m_frames_decoded = snapshot.frames_decoded;
    }

    std::shared_ptr<const StreamDecoderSnapshot> find_primed(uint64_t key) const {
        return m_decoder->find_primed(key);
    }

    void store_primed(uint64_t key, std::shared_ptr<const StreamDecoderSnapshot> snapshot) const {
        m_decoder->store_primed(key, std::move(snapshot));
    }

    static uint64_t primed_key(const int64_t* codes, size_t count, size_t frames) {
        return StreamDecoder::primed_key(codes, count, frames);
    }

    // Decodes the next BLOCK_FRAMES frames. `frame_major_codes` points at [BLOCK_FRAMES, num_quantizers] int64
    // codes (one row per frame). Returns BLOCK_FRAMES * samples_per_frame samples.
    std::vector<float> decode_block(const int64_t* frame_major_codes) {
        const size_t quantizers = m_decoder->num_quantizers();
        const size_t block = StreamDecoder::BLOCK_FRAMES;
        int64_t* dst = m_codes.data<int64_t>();
        for (size_t t = 0; t < block; ++t) {
            for (size_t q = 0; q < quantizers; ++q) {
                dst[q * block + t] = frame_major_codes[t * quantizers + q];
            }
        }
        m_position.data<int64_t>()[0] = static_cast<int64_t>(m_frames_decoded);

        const bool timed = qwen_perf_enabled() && m_on_infer_ms;
        const auto t0 = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        m_request.infer();
        if (timed) {
            m_on_infer_ms(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }

        if (m_decoder->uses_explicit_state()) {
            const auto& names = m_decoder->state_inputs();
            for (size_t i = 0; i < names.size(); ++i) {
                const ov::Tensor out = m_request.get_tensor("new_" + names[i]);
                OPENVINO_ASSERT(out.get_element_type() == ov::element::f32 && out.get_byte_size() == m_states[i].get_byte_size(),
                                "Unexpected shape or type of streaming decoder state output new_", names[i]);
                std::memcpy(m_states[i].data(), out.data(), out.get_byte_size());
            }
        }
        m_frames_decoded += block;

        const ov::Tensor waveform = m_request.get_tensor("waveform");
        OPENVINO_ASSERT(waveform.get_element_type() == ov::element::f32 &&
                            waveform.get_size() == block * m_decoder->samples_per_frame(),
                        "Streaming codec decoder produced an unexpected waveform size");
        const float* ptr = waveform.data<const float>();
        return std::vector<float>(ptr, ptr + waveform.get_size());
    }

private:
    std::shared_ptr<StreamDecoder> m_decoder;
    std::function<void(double)> m_on_infer_ms;
    ov::InferRequest m_request;
    ov::Tensor m_codes;
    ov::Tensor m_position;
    std::vector<ov::Tensor> m_states;  // explicit-state mode only
    size_t m_frames_decoded = 0;
};

}  // namespace qwen3_tts
}  // namespace genai
}  // namespace ov
