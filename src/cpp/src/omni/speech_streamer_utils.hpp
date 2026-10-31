// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include <functional>
#include <memory>
#include <type_traits>
#include <variant>

#include "logger.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

inline StreamingStatus invoke_speech_streamer(const OmniSpeechStreamerVariant& streamer, const ov::Tensor& chunk) {
    return std::visit(
        [&chunk](auto&& arg) -> StreamingStatus {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::function<StreamingStatus(const ov::Tensor&)>>) {
                return arg(chunk);
            } else if constexpr (std::is_same_v<T, std::shared_ptr<OmniSpeechStreamerBase>>) {
                return arg->write(chunk);
            } else {
                return StreamingStatus::RUNNING;
            }
        },
        streamer);
}

inline void end_speech_streamer(const OmniSpeechStreamerVariant& streamer) {
    std::visit(
        [](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::shared_ptr<OmniSpeechStreamerBase>>) {
                arg->end();
            }
        },
        streamer);
}

inline bool is_speech_streamer_active(const OmniSpeechStreamerVariant& streamer) {
    // A C++ caller can hold the shared_ptr alternative with a null pointer, which write() and
    // end() would dereference. Python cannot reach this: an explicit None lands in monostate.
    if (const auto* holder = std::get_if<std::shared_ptr<OmniSpeechStreamerBase>>(&streamer)) {
        return *holder != nullptr;
    }
    return !std::holds_alternative<std::monostate>(streamer);
}

/// @brief Calls end() on an active speech streamer exactly once, on every exit path.
/// Speech generation can throw after streaming has started — speaker lookup, talker input,
/// inference, waveform conversion — and a consumer blocked on the stream must not be left
/// waiting. Call end() directly to close at a chosen point; the destructor covers the rest.
class SpeechStreamerGuard {
public:
    SpeechStreamerGuard(const OmniSpeechStreamerVariant& streamer, bool active)
        : m_streamer(streamer),
          m_active(active) {}

    SpeechStreamerGuard(const SpeechStreamerGuard&) = delete;
    SpeechStreamerGuard& operator=(const SpeechStreamerGuard&) = delete;

    ~SpeechStreamerGuard() {
        try {
            end();
        } catch (const std::exception& e) {
            // end() runs during unwinding here, so letting a user streamer throw would terminate.
            GENAI_WARN("Speech: streamer end() threw during cleanup: %s", e.what());
        } catch (...) {
            GENAI_WARN("Speech: streamer end() threw an unknown exception during cleanup");
        }
    }

    void end() {
        if (m_active) {
            m_active = false;
            end_speech_streamer(m_streamer);
        }
    }

private:
    const OmniSpeechStreamerVariant& m_streamer;
    bool m_active;
};

}  // namespace ov::genai
