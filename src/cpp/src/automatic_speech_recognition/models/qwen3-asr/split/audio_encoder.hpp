#pragma once

#include <mutex>

#include "openvino/runtime/infer_request.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/vlm_config.hpp"
#include "whisper/feature_extractor.hpp"

namespace ov::genai {

class AudioEncoderQwen3ASR {
public:
    AudioEncoderQwen3ASR(const VLMConfig& config,
                         const std::shared_ptr<ov::Model>& audio_model,
                         const std::filesystem::path& config_dir,
                         const std::string& device,
                         const ov::AnyMap& properties);

    std::vector<EncodedAudio> encode(const std::vector<ov::Tensor>& audios);

private:
    struct PreprocessedAudio {
        ov::Tensor padded_feature;
        ov::Tensor padded_mask_after_cnn;
        ov::Tensor aftercnn_lens;
        ov::Tensor cu_seqlens;
        std::vector<size_t> chunk_token_counts;
        size_t total_tokens;
    };

    void validate_audio_input(const ov::Tensor& audio) const;
    /**
     * @brief Prepare one recording for audio encoder inference without running the encoder.
     * @param audio Validated, nonempty 1-D float32 PCM tensor at the configured sampling rate.
     * @return Mel chunks, CNN-output validity mask, recording length, attention offsets,
     *         and valid-token counts used to compact the encoder output.
     */
    PreprocessedAudio preprocess_audio(const ov::Tensor& audio);
    EncodedAudio encode_audio(const ov::Tensor& audio);

    static constexpr size_t get_feat_extract_output_length(size_t input_length) {
        size_t length = input_length;
        length = (length + 1) / 2;
        length = (length + 1) / 2;
        length = (length + 1) / 2;
        return length;
    }

    const VLMConfig m_config;
    WhisperFeatureExtractor m_feature_extractor;
    ov::InferRequest m_request;
    std::mutex m_audio_mutex;
};

}  // namespace ov::genai
