// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

namespace ov {
namespace genai {
namespace qwen3_tts {

// Single-dispatch fused subcode predictor for Qwen3-TTS (`openvino_code_predictor_fused.xml`).
//
// Replaces the per-frame 15-step code predictor loop + 30 embedding lookups (45 infer calls/frame)
// with a single OpenVINO InferRequest call that predicts all remaining code groups and returns
// the summed `next_embed` ready to feed back into the talker backbone.
//
// Expected IR contract:
//   Inputs:
//     - `past_hidden`:   [-1, 1, H] (f32) — last hidden state from talker
//     - `first_code`:    [-1, 1] (i64)    — sampled group-0 codec token
//     - `tts_pad_embed`: [1, 1, H] (f32)  — optional additive text conditioning (zeros for sum-only)
//     - `uniform_noise`: [-1, G-1, V_sub] (f32) — optional in-graph Gumbel-max sampling noise
//     - `temperature`:   [1] (f32)        — optional sampling temperature (<= 0 for greedy)
//   Outputs:
//     - `codes`:         [-1, G] (i64)    — full frame codec groups [c_0, ..., c_{G-1}]
//                                           (or [-1, G-1] residual groups; first_code is prepended)
//     - `next_embed`:    [-1, 1, H] (f32) — summed group embeddings (+ tts_pad_embed if present)
class Qwen3TTSFusedCodePredictor {
public:
    struct SingleResult {
        std::vector<int64_t> codes;
        ov::Tensor next_embed;
    };

    struct BatchResult {
        std::vector<std::vector<int64_t>> codes;
        ov::Tensor next_embeds;  // [B, 1, H] f32
    };

    bool initialize(const std::filesystem::path& model_path,
                    const std::string& device,
                    const ov::AnyMap& properties,
                    size_t num_code_groups,
                    size_t hidden_size);

    bool initialize_from_model(const std::shared_ptr<ov::Model>& model,
                               const std::string& device,
                               const ov::AnyMap& properties,
                               size_t num_code_groups,
                               size_t hidden_size);

    bool is_available() const {
        return m_available;
    }

    bool has_tts_pad_embed() const {
        return m_has_tts_pad_embed;
    }

    bool supports_sampling() const {
        return m_has_uniform_noise && m_has_temperature;
    }

    bool uses_remote_next_embed() const {
        return m_use_remote_next_embed;
    }

    const std::string& remote_context_device_name() const {
        return m_remote_context_device_name;
    }

    size_t sub_vocab_size() const {
        return m_sub_vocab_size;
    }

    const std::string& target_device() const {
        return m_device;
    }

    // Single-sequence fused prediction (batch size 1).
    // When `allow_device_zero_copy` is true and `m_use_remote_next_embed` is active (e.g. GPU
    // Talker + GPU fused predictor in non_streaming_mode), `next_embed` is returned as a ping-pong
    // ov::RemoteTensor to avoid host round-trips before feeding back into the talker.
    SingleResult predict_single(const ov::Tensor& past_hidden,
                                int64_t first_code,
                                const ov::Tensor& additive_embed,
                                bool additive_is_tts_pad,
                                bool do_sample,
                                float temperature,
                                std::mt19937& rng,
                                bool allow_device_zero_copy,
                                std::unordered_map<std::string, double>& perf_ms,
                                std::unordered_map<std::string, int64_t>& perf_calls);

    // Batched fused prediction (batch size B >= 1) for continuous batching talker steps.
    // `per_row_additive_embeds` (size B): if non-empty, each non-null pointer points to H floats
    // to add to row b's next_embed (passing zeros to the IR's `tts_pad_embed` when rows differ).
    BatchResult predict_batch(const ov::Tensor& past_hidden_batch,
                              const std::vector<int64_t>& first_codes,
                              const ov::Tensor& shared_tts_pad_embed,
                              const std::vector<const float*>& per_row_additive_embeds,
                              bool do_sample,
                              float temperature,
                              std::mt19937& rng,
                              std::unordered_map<std::string, double>& perf_ms,
                              std::unordered_map<std::string, int64_t>& perf_calls);

private:
    void prepare_sampling_inputs(size_t batch_size, bool do_sample, float temperature, std::mt19937& rng);
    void init_remote_next_embed_buffers();

    bool m_available = false;
    bool m_has_tts_pad_embed = false;
    bool m_has_uniform_noise = false;
    bool m_has_temperature = false;
    bool m_codes_include_first = true;
    bool m_use_remote_next_embed = false;

    size_t m_num_code_groups = 16;
    size_t m_hidden_size = 1024;
    size_t m_sub_vocab_size = 2048;
    std::string m_device;

    ov::CompiledModel m_compiled;
    ov::InferRequest m_request;

    // Reusable host tensors for batch=1 fast path.
    ov::Tensor m_single_past_hidden;
    ov::Tensor m_single_first_code;
    ov::Tensor m_zero_tts_pad;
    ov::Tensor m_temperature_tensor;

    // Ping-pong device RemoteTensors for zero-copy next_embed -> talker inputs_embeds on GPU.
    std::string m_remote_context_device_name;
    std::array<ov::Tensor, 2> m_remote_next_embeds;
    size_t m_ping_pong_idx = 0;
};

}  // namespace qwen3_tts
}  // namespace genai
}  // namespace ov
