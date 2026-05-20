// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen3_omni/audio_encoder.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include "openvino/openvino.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {
// Qwen3-Omni declares "feature_extractor_type": "WhisperFeatureExtractor" in its
// HF preprocessor_config.json. These match the HF WhisperFeatureExtractor defaults.
constexpr size_t WHISPER_SAMPLING_RATE = 16000;
constexpr size_t WHISPER_N_FFT = 400;
constexpr size_t WHISPER_HOP_LENGTH = 160;
}  // namespace

AudioEncoderQwen3Omni::AudioEncoderQwen3Omni(const std::filesystem::path& model_dir,
                                             const VLMConfig& config,
                                             const std::string& device,
                                             const ov::AnyMap& properties)
    : m_config(config),
      m_feature_extractor(config.audio_config_num_mel_bins,
                          WHISPER_SAMPLING_RATE,
                          WHISPER_N_FFT,
                          WHISPER_HOP_LENGTH) {
    // Resolve canonical paths to prevent path traversal via attacker-controlled model_dir.
    const auto canonical_model_dir = std::filesystem::weakly_canonical(model_dir);
    auto model_path = model_dir / "openvino_audio_encoder_model.xml";
    auto canonical_model_path = std::filesystem::weakly_canonical(model_path);
    OPENVINO_ASSERT(canonical_model_path.string().rfind(canonical_model_dir.string(), 0) == 0,
                    "Refusing to load audio encoder file outside model_dir: ",
                    canonical_model_path.string());
    // Audio encoder is optional - model works as text+vision VLM without it
    if (!std::filesystem::exists(model_path)) {
        return;
    }

    auto model = utils::singleton_core().read_model(model_path);
    auto compiled = utils::singleton_core().compile_model(model, device, properties);
    m_ireq_queue = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled.get_property(ov::optimal_number_of_infer_requests),
        [&compiled] {
            return compiled.create_infer_request();
        });
}

void AudioEncoderQwen3Omni::validate_audio_shape(const ov::Shape& shape, ov::element::Type element_type) {
    OPENVINO_ASSERT(element_type == ov::element::f32,
                    "Audio input must be float32 PCM, got ",
                    element_type);
    OPENVINO_ASSERT(shape.size() == 1,
                    "Audio input must be a 1-D tensor of PCM samples, got rank ",
                    shape.size());
    const size_t audio_len = ov::shape_size(shape);
    OPENVINO_ASSERT(audio_len > 0,
                    "Audio input is empty (0 samples). Provide at least n_fft/2 + 1 samples.");
    OPENVINO_ASSERT(audio_len <= MAX_AUDIO_SAMPLES,
                    "Audio input too large: got ",
                    audio_len,
                    " samples, max is ",
                    MAX_AUDIO_SAMPLES,
                    " (~30 min @ 16 kHz)");
}

void AudioEncoderQwen3Omni::validate_audio_input(const ov::Tensor& audio_raw) {
    validate_audio_shape(audio_raw.get_shape(), audio_raw.get_element_type());
}

std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> AudioEncoderQwen3Omni::preprocess_audio(
    const ov::Tensor& audio_raw) {
    // Reject malformed or oversized audio before any allocation.
    validate_audio_input(audio_raw);

    const auto* audio_data = audio_raw.data<float>();
    const auto audio_len = audio_raw.get_size();
    std::vector<float> raw_speech(audio_data, audio_data + audio_len);

    auto features = m_feature_extractor.extract(raw_speech);
    const auto& mel_data = features.data;
    const size_t n_frames = features.n_frames;
    OPENVINO_ASSERT(n_frames > 0, "Audio input too short to produce mel spectrogram frames");

    const auto num_mel_bins = m_config.audio_config_num_mel_bins;
    const auto n_window_infer = m_config.audio_config_n_window_infer;
    const auto n_window = m_config.audio_config_n_window;

    // Chunk the mel spectrogram into windows of (n_window_infer * 2) frames
    // Each window overlaps: window i covers frames [i * n_window_infer, i * n_window_infer + 2 * n_window_infer)
    const size_t chunk_size = n_window_infer * 2;
    const size_t num_chunks = (n_frames + n_window_infer - 1) / n_window_infer;

    ov::Tensor padded_feature(ov::element::f32, {num_chunks, num_mel_bins, chunk_size});
    auto* pf_data = padded_feature.data<float>();
    std::fill(pf_data, pf_data + padded_feature.get_size(), 0.0f);

    std::vector<size_t> chunk_frame_lens(num_chunks);
    for (size_t c = 0; c < num_chunks; c++) {
        const size_t start = c * n_window_infer;
        const size_t end = std::min(start + chunk_size, n_frames);
        chunk_frame_lens[c] = end - start;

        // Copy mel data for this chunk: mel_data is [num_mel_bins, n_frames] row-major
        for (size_t mel = 0; mel < num_mel_bins; mel++) {
            for (size_t f = 0; f < chunk_frame_lens[c]; f++) {
                pf_data[c * num_mel_bins * chunk_size + mel * chunk_size + f] = mel_data[mel * n_frames + start + f];
            }
        }
    }

    const auto max_aftercnn_len = get_feat_extract_output_length(chunk_size);
    ov::Tensor aftercnn_lens(ov::element::i64, {num_chunks});
    auto* acl_data = aftercnn_lens.data<int64_t>();

    ov::Tensor padded_mask_after_cnn(ov::element::boolean, {num_chunks, max_aftercnn_len});
    auto* mask_data = padded_mask_after_cnn.data<bool>();
    std::fill(mask_data, mask_data + padded_mask_after_cnn.get_size(), false);

    for (size_t c = 0; c < num_chunks; c++) {
        const auto cnn_len = get_feat_extract_output_length(chunk_frame_lens[c]);
        acl_data[c] = static_cast<int64_t>(cnn_len);
        for (size_t i = 0; i < cnn_len; i++) {
            mask_data[c * max_aftercnn_len + i] = true;
        }
    }

    // Compute cu_seqlens for chunked attention (must be done outside the model
    // because it uses data-dependent loops that can't be traced)
    const size_t window_aftercnn = max_aftercnn_len * (n_window_infer / (n_window * 2));
    std::vector<int32_t> cu_chunk_lens = {0};
    OPENVINO_ASSERT(window_aftercnn > 0,
                    "Audio encoder: window_aftercnn is zero — check config values n_window_infer and n_window");
    for (size_t c = 0; c < num_chunks; c++) {
        const auto cnn_len = static_cast<size_t>(acl_data[c]);
        const size_t full_windows = cnn_len / window_aftercnn;
        for (size_t w = 0; w < full_windows; w++) {
            cu_chunk_lens.push_back(static_cast<int32_t>(window_aftercnn));
        }
        const size_t remainder = cnn_len % window_aftercnn;
        if (remainder != 0) {
            cu_chunk_lens.push_back(static_cast<int32_t>(remainder));
        }
    }

    ov::Tensor cu_seqlens(ov::element::i32, {cu_chunk_lens.size()});
    auto* cu_data = cu_seqlens.data<int32_t>();
    cu_data[0] = cu_chunk_lens[0];
    for (size_t i = 1; i < cu_chunk_lens.size(); i++) {
        cu_data[i] = cu_data[i - 1] + cu_chunk_lens[i];
    }

    return {padded_feature, padded_mask_after_cnn, aftercnn_lens, cu_seqlens};
}

ov::Tensor AudioEncoderQwen3Omni::encode(const ov::Tensor& audio_raw) {
    OPENVINO_ASSERT(m_ireq_queue, "Audio encoder not initialized (model not found)");

    auto [padded_feature, padded_mask, aftercnn_lens, cu_seqlens] = preprocess_audio(audio_raw);

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(m_ireq_queue.get());
    auto& ireq = infer_request_guard.get();
    ireq.set_tensor("padded_feature", padded_feature);
    ireq.set_tensor("padded_mask_after_cnn", padded_mask);
    ireq.set_tensor("aftercnn_lens", aftercnn_lens);
    ireq.set_tensor("cu_seqlens", cu_seqlens);
    ireq.infer();

    const auto audio_features = ireq.get_tensor("audio_features");

    // Output is padded 3D [N_chunks, aftercnn_time, dim] -- extract valid tokens using mask
    const auto feat_shape = audio_features.get_shape();
    const auto num_chunks = feat_shape[0];
    const auto max_aftercnn_len = feat_shape[1];
    const auto dim = feat_shape[2];

    const auto* mask_data = padded_mask.data<const bool>();
    const auto* feat_data = audio_features.data<const float>();

    size_t total_valid = 0;
    for (size_t i = 0; i < num_chunks * max_aftercnn_len; i++) {
        if (mask_data[i])
            total_valid++;
    }

    ov::Tensor result(ov::element::f32, {total_valid, dim});
    auto* dst = result.data<float>();
    for (size_t c = 0; c < num_chunks; c++) {
        for (size_t t = 0; t < max_aftercnn_len; t++) {
            if (mask_data[c * max_aftercnn_len + t]) {
                std::memcpy(dst, feat_data + (c * max_aftercnn_len + t) * dim, dim * sizeof(float));
                dst += dim;
            }
        }
    }

    return result;
}

}  // namespace ov::genai
