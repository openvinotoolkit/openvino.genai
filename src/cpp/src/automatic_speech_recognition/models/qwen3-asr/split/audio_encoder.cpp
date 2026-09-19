#include "audio_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "utils.hpp"

namespace ov::genai {

AudioEncoderQwen3ASR::AudioEncoderQwen3ASR(const VLMConfig& config,
                                           const std::shared_ptr<ov::Model>& audio_model,
                                           const std::filesystem::path& config_dir,
                                           const std::string& device,
                                           const ov::AnyMap& properties)
    : m_config(config),
      m_feature_extractor(config_dir / "preprocessor_config.json") {
    OPENVINO_ASSERT(
        config.audio_config_n_window > 0 && config.audio_config_n_window <= config.audio_config_n_window_infer / 2,
        "Qwen3-ASR requires 0 < 2 * n_window <= n_window_infer");
    OPENVINO_ASSERT(config.audio_config_num_mel_bins == m_feature_extractor.feature_size,
                    "Qwen3-ASR audio config and preprocessor feature sizes must match");
    m_request =
        utils::singleton_core()
            .compile_model(audio_model, device, utils::get_model_properties(properties, "audio_encoder", device))
            .create_infer_request();
}

std::vector<EncodedAudio> AudioEncoderQwen3ASR::encode(const std::vector<ov::Tensor>& audios) {
    std::lock_guard<std::mutex> lock(m_audio_mutex);
    std::vector<EncodedAudio> encoded;
    encoded.reserve(audios.size());
    for (const auto& audio : audios) {
        validate_audio_input(audio);
        encoded.push_back(audio.get_size() == 0 ? EncodedAudio{} : encode_audio(audio));
    }
    return encoded;
}

void AudioEncoderQwen3ASR::validate_audio_input(const ov::Tensor& audio) const {
    OPENVINO_ASSERT(audio.get_shape().size() == 1 && audio.get_element_type() == ov::element::f32,
                    "Qwen3-ASR audio must be a 1-D float32 PCM tensor");
    const size_t sample_count = audio.get_size();
    if (sample_count == 0) {
        return;
    }
    OPENVINO_ASSERT(sample_count > m_feature_extractor.n_fft / 2,
                    "Qwen3-ASR audio must contain more than n_fft / 2 samples");
    OPENVINO_ASSERT(sample_count <= 1200 * m_feature_extractor.sampling_rate,
                    "Qwen3-ASR audio chunk must not exceed 1200 seconds");
    const float* samples = audio.data<const float>();
    OPENVINO_ASSERT(std::all_of(samples,
                                samples + sample_count,
                                [](float sample) {
                                    return std::isfinite(sample);
                                }),
                    "Qwen3-ASR audio samples must be finite");
}

/**
 * @brief Extract mel features and prepare padded chunks and metadata for encoder inference.
 *
 * Example: 250 mel frames, 128 mel bins, n_window=50, n_window_infer=800.
 *
 * 1. Extract mel features from the PCM waveform:
 * @code
 *    PCM [sample_count] -> mel features [128, 250]
 * @endcode
 *
 * 2. Split time into non-overlapping chunks of at most 2 * n_window = 100 frames.
 *    Zero-pad to the longest actual chunk. A recording shorter than 100 frames
 *    keeps its actual width instead of being padded to 100.
 * @code
 *    Chunk 0: frames [  0, 100) -> [100 valid frames                 ]
 *    Chunk 1: frames [100, 200) -> [100 valid frames                 ]
 *    Chunk 2: frames [200, 250) -> [ 50 valid frames | 50 zero frames]
 *
 *    padded_feature shape: [3, 128, 100] = [chunks, mel_bins, padded_frames]
 * @endcode
 *
 * 3. Predict output lengths for three stride-2 convolutions, each using
 *    (length + 1) / 2 integer division: 100 -> 50 -> 25 -> 13;
 *    50 -> 25 -> 13 -> 7. The CNN is not run during preprocessing.
 *    Mark each chunk's valid output prefix in the boolean mask:
 * @code
 *    Chunk       Valid frames    Valid tokens    Mask (T=true, F=false)
 *      0             100             13           TTTTTTTTTTTTT
 *      1             100             13           TTTTTTTTTTTTT
 *      2              50              7           TTTTTTTFFFFFF
 *
 *    padded_mask_after_cnn shape: [3, 13]
 *    chunk_token_counts:          [13, 13, 7]
 *    total_tokens:                33
 *    aftercnn_lens:               tensor containing [33], shape [1]
 * @endcode
 *
 * 4. Build cumulative attention-window offsets into the compacted token sequence.
 *    Window capacity is padded_tokens * (n_window_infer / (2 * n_window)),
 *    using integer division. Include zero and end at total_tokens:
 * @code
 *    Window capacity: 13 * (800 / 100) = 104 tokens
 *    33 tokens:  cu_seqlens = [0, 33]
 *    130 tokens: cu_seqlens = [0, 104, 130]
 * @endcode
 *    The current export ignores this attention-window metadata.
 *
 * After preprocessing, encode_audio() runs inference to obtain [3, 13, hidden_size]
 * and copies only each chunk's valid prefix, producing [33, hidden_size].
 */
AudioEncoderQwen3ASR::PreprocessedAudio AudioEncoderQwen3ASR::preprocess_audio(const ov::Tensor& audio) {
    const size_t sample_count = audio.get_size();
    const float* samples = audio.data<const float>();
    const auto features = m_feature_extractor.extract(std::vector<float>(samples, samples + sample_count), false);
    OPENVINO_ASSERT(features.n_frames > 0, "Qwen3-ASR audio must produce at least one mel frame");

    const size_t chunk_frames = 2 * m_config.audio_config_n_window;
    const size_t padded_frames = std::min(chunk_frames, features.n_frames);
    const size_t num_chunks = (features.n_frames + chunk_frames - 1) / chunk_frames;
    const size_t padded_tokens = get_feat_extract_output_length(padded_frames);
    ov::Tensor padded_feature(ov::element::f32, {num_chunks, features.feature_size, padded_frames});
    ov::Tensor padded_mask(ov::element::boolean, {num_chunks, padded_tokens});
    float* feature_data = padded_feature.data<float>();
    bool* mask_data = padded_mask.data<bool>();
    std::fill_n(feature_data, padded_feature.get_size(), 0.0f);
    std::fill_n(mask_data, padded_mask.get_size(), false);

    std::vector<size_t> chunk_token_counts;
    chunk_token_counts.reserve(num_chunks);
    size_t total_tokens = 0;
    for (size_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
        const size_t frame_offset = chunk_index * chunk_frames;
        const size_t valid_frames = std::min(chunk_frames, features.n_frames - frame_offset);
        const size_t valid_tokens = get_feat_extract_output_length(valid_frames);
        chunk_token_counts.push_back(valid_tokens);
        total_tokens += valid_tokens;
        std::fill_n(mask_data + chunk_index * padded_tokens, valid_tokens, true);
        for (size_t mel_index = 0; mel_index < features.feature_size; ++mel_index) {
            std::memcpy(feature_data + (chunk_index * features.feature_size + mel_index) * padded_frames,
                        features.data.data() + mel_index * features.n_frames + frame_offset,
                        valid_frames * sizeof(float));
        }
    }

    OPENVINO_ASSERT(total_tokens <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                    "Qwen3-ASR audio token count exceeds int32 attention offsets");
    const size_t window_tokens = padded_tokens * (m_config.audio_config_n_window_infer / chunk_frames);
    const size_t num_windows = (total_tokens + window_tokens - 1) / window_tokens;
    ov::Tensor aftercnn_lens(ov::element::i64, {1});
    aftercnn_lens.data<int64_t>()[0] = static_cast<int64_t>(total_tokens);
    ov::Tensor cu_seqlens(ov::element::i32, {num_windows + 1});
    for (size_t window_index = 0; window_index <= num_windows; ++window_index) {
        cu_seqlens.data<int32_t>()[window_index] =
            static_cast<int32_t>(std::min(window_index * window_tokens, total_tokens));
    }

    return {std::move(padded_feature),
            std::move(padded_mask),
            std::move(aftercnn_lens),
            std::move(cu_seqlens),
            std::move(chunk_token_counts),
            total_tokens};
}

EncodedAudio AudioEncoderQwen3ASR::encode_audio(const ov::Tensor& audio) {
    const auto preprocessed = preprocess_audio(audio);
    const size_t num_chunks = preprocessed.chunk_token_counts.size();
    const size_t padded_tokens = preprocessed.padded_mask_after_cnn.get_shape()[1];
    m_request.set_tensor("padded_feature", preprocessed.padded_feature);
    m_request.set_tensor("padded_mask_after_cnn", preprocessed.padded_mask_after_cnn);
    m_request.set_tensor("aftercnn_lens", preprocessed.aftercnn_lens);
    m_request.set_tensor("cu_seqlens", preprocessed.cu_seqlens);
    m_request.infer();
    const auto encoded = m_request.get_tensor("audio_features");
    OPENVINO_ASSERT(encoded.get_shape() == ov::Shape({num_chunks, padded_tokens, m_config.hidden_size}),
                    "Qwen3-ASR audio encoder must return [chunks, padded_tokens, hidden_size], got ",
                    encoded.get_shape());
    ov::Tensor compacted(ov::element::f32, {preprocessed.total_tokens, m_config.hidden_size});
    float* destination = compacted.data<float>();
    const float* source = encoded.data<const float>();
    for (size_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
        const size_t valid_elements = preprocessed.chunk_token_counts[chunk_index] * m_config.hidden_size;
        std::memcpy(destination,
                    source + chunk_index * padded_tokens * m_config.hidden_size,
                    valid_elements * sizeof(float));
        destination += valid_elements;
    }
    return {std::move(compacted), preprocessed.total_tokens};
}

}  // namespace ov::genai
