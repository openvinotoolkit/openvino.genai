#pragma once

#include "audio_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

class InputsEmbedderQwen3ASR : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderQwen3ASR(const VLMConfig& config,
                          const std::filesystem::path& model_dir,
                          const Tokenizer& tokenizer,
                          const std::string& device,
                          const ov::AnyMap& properties);
    InputsEmbedderQwen3ASR(const VLMConfig& config,
                          const ModelsMap& models_map,
                          const Tokenizer& tokenizer,
                          const std::filesystem::path& config_dir,
                          const std::string& device,
                          const ov::AnyMap& properties);

    std::vector<EncodedAudio> encode_audios(const std::vector<ov::Tensor>& audios) override;
    std::vector<EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base_id,
                                      const std::vector<EncodedImage>& images) const override;
    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t image_base_id,
                                      size_t video_base_id,
                                      size_t audio_base_id,
                                      const std::vector<EncodedImage>& images,
                                      const std::vector<EncodedVideo>& videos,
                                      const std::vector<EncodedAudio>& audios) const override;

    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                const std::vector<EncodedImage>& images,
                                VLMPerfMetrics& metrics,
                                bool recalculate_merged_embeddings = true,
                                const std::vector<size_t>& image_sequence = {}) override;
    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                const std::vector<EncodedImage>& images,
                                const std::vector<EncodedVideo>& videos,
                                const std::vector<EncodedAudio>& audios,
                                VLMPerfMetrics& metrics,
                                bool recalculate_merged_embeddings,
                                const std::vector<size_t>& image_sequence,
                                const std::vector<size_t>& videos_sequence,
                                const std::vector<size_t>& audios_sequence,
                                size_t base_audio_id,
                                const std::vector<std::pair<size_t, size_t>>& history_vision_count) override;

    std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(size_t inputs_embeds_size,
                                                                size_t history_size) override;
    void start_chat(const std::string& system_message) override;

private:
    InputsEmbedderQwen3ASR(const VLMConfig& config,
                          const Tokenizer& tokenizer,
                          EmbeddingsModel::Ptr embedding,
                          const std::shared_ptr<ov::Model>& audio_model,
                          const std::filesystem::path& config_dir,
                          const std::string& device,
                          const ov::AnyMap& properties);

    void merge_audio_embeddings(ov::Tensor& inputs_embeds,
                                const ov::Tensor& input_ids,
                                const std::vector<EncodedAudio>& audios,
                                const std::vector<size_t>& audios_sequence,
                                size_t base_audio_id) const;

    AudioEncoderQwen3ASR m_audio_encoder;
};

}  // namespace ov::genai
