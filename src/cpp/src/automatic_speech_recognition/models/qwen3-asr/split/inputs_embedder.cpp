#include "inputs_embedder.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>

#include "openvino/genai/visual_language/perf_metrics.hpp"

namespace ov::genai {
namespace {

const std::string AUDIO_PAD = "<|audio_pad|>";
const std::string AUDIO_START = "<|audio_start|>";
const std::string AUDIO_END = "<|audio_end|>";
const std::string AUDIO_TAG = AUDIO_START + AUDIO_PAD + AUDIO_END;

const EncodedAudio& get_audio(const std::vector<EncodedAudio>& audios, size_t audio_id, size_t base_audio_id) {
    OPENVINO_ASSERT(audio_id >= base_audio_id && audio_id - base_audio_id < audios.size(),
                    "Qwen3-ASR audio index ",
                    audio_id,
                    " is outside the supplied audio range");
    return audios[audio_id - base_audio_id];
}

void expand_audio_tags_in_prompt(std::string& prompt,
                                 const std::vector<EncodedAudio>& audios,
                                 const std::vector<size_t>& audios_sequence,
                                 size_t base_audio_id) {
    size_t offset = 0;
    for (size_t audio_id : audios_sequence) {
        const auto& audio = get_audio(audios, audio_id, base_audio_id);
        const size_t tag_offset = prompt.find(AUDIO_TAG, offset);
        OPENVINO_ASSERT(tag_offset != std::string::npos, "Qwen3-ASR prompt is missing a native audio tag");
        std::string expanded = AUDIO_START;
        expanded.reserve(AUDIO_START.size() + AUDIO_PAD.size() * audio.num_audio_tokens + AUDIO_END.size());
        for (size_t token_index = 0; token_index < audio.num_audio_tokens; ++token_index) {
            expanded += AUDIO_PAD;
        }
        expanded += AUDIO_END;
        prompt.replace(tag_offset, AUDIO_TAG.size(), expanded);
        offset = tag_offset + expanded.size();
    }
    OPENVINO_ASSERT(prompt.find(AUDIO_TAG, offset) == std::string::npos,
                    "Qwen3-ASR prompt has audio tags without matching audio inputs");
}

}  // namespace

InputsEmbedderQwen3ASR::InputsEmbedderQwen3ASR(const VLMConfig& config,
                                               const std::filesystem::path& model_dir,
                                               const Tokenizer& tokenizer,
                                               const std::string& device,
                                               const ov::AnyMap& properties)
    : InputsEmbedderQwen3ASR(config,
                             tokenizer,
                             EmbeddingsModel::create(model_dir, config.scale_emb, device, properties),
                             utils::singleton_core().read_model(model_dir / "openvino_audio_encoder_model.xml"),
                             model_dir,
                             device,
                             properties) {}

InputsEmbedderQwen3ASR::InputsEmbedderQwen3ASR(const VLMConfig& config,
                                               const ModelsMap& models_map,
                                               const Tokenizer& tokenizer,
                                               const std::filesystem::path& config_dir,
                                               const std::string& device,
                                               const ov::AnyMap& properties)
    : InputsEmbedderQwen3ASR(
          config,
          tokenizer,
          EmbeddingsModel::create(utils::get_model_weights_pair(models_map, "text_embeddings").first,
                                  utils::get_model_weights_pair(models_map, "text_embeddings").second,
                                  config.scale_emb,
                                  device,
                                  properties),
          utils::singleton_core().read_model(utils::get_model_weights_pair(models_map, "audio_encoder").first,
                                             utils::get_model_weights_pair(models_map, "audio_encoder").second),
          config_dir,
          device,
          properties) {}

InputsEmbedderQwen3ASR::InputsEmbedderQwen3ASR(const VLMConfig& config,
                                               const Tokenizer& tokenizer,
                                               EmbeddingsModel::Ptr embedding,
                                               const std::shared_ptr<ov::Model>& audio_model,
                                               const std::filesystem::path& config_dir,
                                               const std::string& device,
                                               const ov::AnyMap& properties)
    : IInputsEmbedder(config, tokenizer, std::move(embedding)),
      m_audio_encoder(config, audio_model, config_dir, device, properties) {
    const auto vocab = tokenizer.get_vocab();
    const auto audio_token = vocab.find(AUDIO_PAD);
    OPENVINO_ASSERT(audio_token != vocab.end() && audio_token->second == config.audio_token_id,
                    "Qwen3-ASR audio_token_id must match tokenizer token <|audio_pad|>");
    set_add_special_tokens(false);
}

std::vector<EncodedAudio> InputsEmbedderQwen3ASR::encode_audios(const std::vector<ov::Tensor>& audios) {
    return m_audio_encoder.encode(audios);
}

std::vector<EncodedImage> InputsEmbedderQwen3ASR::encode_images(const std::vector<ov::Tensor>& images) {
    OPENVINO_ASSERT(images.empty(), "Qwen3-ASR does not support images");
    return {};
}

NormalizedPrompt InputsEmbedderQwen3ASR::normalize_prompt(const std::string& prompt,
                                                          size_t base_id,
                                                          const std::vector<EncodedImage>& images) const {
    return normalize_prompt(prompt, base_id, 0, 0, images, {}, {});
}

NormalizedPrompt InputsEmbedderQwen3ASR::normalize_prompt(const std::string& prompt,
                                                          size_t image_base_id,
                                                          size_t video_base_id,
                                                          size_t audio_base_id,
                                                          const std::vector<EncodedImage>& images,
                                                          const std::vector<EncodedVideo>& videos,
                                                          const std::vector<EncodedAudio>& audios) const {
    OPENVINO_ASSERT(images.empty() && videos.empty(), "Qwen3-ASR does not support images or videos");
    auto [normalized, sequence] =
        normalize_media_tags(prompt, AUDIO_TAG, AUDIO_TAG, audio_base_id, audios.size(), ModalityType::AUDIO);
    return {std::move(normalized), {}, {}, std::move(sequence)};
}

ov::Tensor InputsEmbedderQwen3ASR::get_inputs_embeds(const std::string& prompt,
                                                     const std::vector<EncodedImage>& images,
                                                     VLMPerfMetrics& metrics,
                                                     bool recalculate_merged_embeddings,
                                                     const std::vector<size_t>& image_sequence) {
    return get_inputs_embeds(prompt,
                             images,
                             {},
                             {},
                             metrics,
                             recalculate_merged_embeddings,
                             image_sequence,
                             {},
                             {},
                             0,
                             {});
}

ov::Tensor InputsEmbedderQwen3ASR::get_inputs_embeds(
    const std::string& prompt,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos,
    const std::vector<EncodedAudio>& audios,
    VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<size_t>& audios_sequence,
    size_t base_audio_id,
    const std::vector<std::pair<size_t, size_t>>& history_vision_count) {
    OPENVINO_ASSERT(images.empty() && videos.empty() && image_sequence.empty() && videos_sequence.empty(),
                    "Qwen3-ASR does not support images or videos");
    std::string formatted_prompt = prompt;

    if (m_apply_chat_template) {
        const auto template_start = std::chrono::steady_clock::now();
        std::string system_context;
        auto audio_content = JsonContainer::array();
        size_t offset = 0;
        size_t tag_offset = prompt.find(AUDIO_TAG);
        while (tag_offset != std::string::npos) {
            system_context.append(prompt, offset, tag_offset - offset);
            audio_content.push_back(JsonContainer{{"type", "audio"}});
            offset = tag_offset + AUDIO_TAG.size();
            tag_offset = prompt.find(AUDIO_TAG, offset);
        }
        system_context.append(prompt, offset, std::string::npos);
        const ChatHistory history(
            {{{"role", "system"}, {"content", system_context}}, {{"role", "user"}, {"content", audio_content}}});
        formatted_prompt = m_tokenizer.apply_chat_template(history, true);
        PerfMetrics::emplace_duration(metrics.raw_metrics.chat_template_durations, template_start);
    }

    expand_audio_tags_in_prompt(formatted_prompt, audios, audios_sequence, base_audio_id);

    const auto tokenization_start = std::chrono::steady_clock::now();
    const auto input_ids = m_tokenizer.encode(formatted_prompt, ov::genai::add_special_tokens(false)).input_ids;
    PerfMetrics::emplace_duration(metrics.raw_metrics.tokenization_durations, tokenization_start);
    OPENVINO_ASSERT(input_ids.get_size() > 0, "Qwen3-ASR prompt must not be empty");

    CircularBufferQueueElementGuard<EmbeddingsRequest> request_guard(m_embedding->get_request_queue().get());
    const auto text_embeddings = get_text_embedding(request_guard.get(), input_ids, metrics);
    OPENVINO_ASSERT(text_embeddings.get_shape() == ov::Shape({1, input_ids.get_size(), m_vlm_config.hidden_size}),
                    "Qwen3-ASR text embeddings must have shape [1, sequence_length, hidden_size]");

    ov::Tensor inputs_embeds(text_embeddings.get_element_type(), text_embeddings.get_shape());
    text_embeddings.copy_to(inputs_embeds);

    merge_audio_embeddings(inputs_embeds, input_ids, audios, audios_sequence, base_audio_id);
    m_prev_hist_length = m_cache_state.get_state().size();
    m_cache_state.add_inputs(input_ids);
    
    return inputs_embeds;
}

void InputsEmbedderQwen3ASR::merge_audio_embeddings(ov::Tensor& inputs_embeds,
                                                    const ov::Tensor& input_ids,
                                                    const std::vector<EncodedAudio>& audios,
                                                    const std::vector<size_t>& audios_sequence,
                                                    size_t base_audio_id) const {
    const int64_t* tokens = input_ids.data<const int64_t>();
    size_t token_offset = 0;
    for (size_t audio_id : audios_sequence) {
        const auto& audio = get_audio(audios, audio_id, base_audio_id);
        if (audio.num_audio_tokens == 0) {
            continue;
        }
        OPENVINO_ASSERT(audio.audio_features && audio.audio_features.get_shape() ==
                                                    ov::Shape({audio.num_audio_tokens, m_vlm_config.hidden_size}),
                        "Qwen3-ASR audio features must match [num_audio_tokens, hidden_size]");
        while (token_offset < input_ids.get_size() && tokens[token_offset] != m_vlm_config.audio_token_id) {
            ++token_offset;
        }
        const size_t run_start = token_offset;
        while (token_offset < input_ids.get_size() && tokens[token_offset] == m_vlm_config.audio_token_id) {
            ++token_offset;
        }
        OPENVINO_ASSERT(token_offset - run_start == audio.num_audio_tokens,
                        "Qwen3-ASR audio placeholder run length does not match the supplied audio");
        std::memcpy(inputs_embeds.data<float>() + run_start * m_vlm_config.hidden_size,
                    audio.audio_features.data<const float>(),
                    audio.audio_features.get_byte_size());
    }
    OPENVINO_ASSERT(std::find(tokens + token_offset, tokens + input_ids.get_size(), m_vlm_config.audio_token_id) ==
                        tokens + input_ids.get_size(),
                    "Qwen3-ASR prompt has audio placeholders without matching audio inputs");
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderQwen3ASR::get_position_ids(size_t inputs_embeds_size,
                                                                                       size_t history_size) {
    ov::Tensor position_ids(ov::element::i64, {3, 1, inputs_embeds_size});
    int64_t* positions = position_ids.data<int64_t>();
    for (size_t channel = 0; channel < 3; ++channel) {
        std::iota(positions + channel * inputs_embeds_size,
                  positions + (channel + 1) * inputs_embeds_size,
                  static_cast<int64_t>(history_size));
    }
    return {position_ids, 0};
}

void InputsEmbedderQwen3ASR::start_chat(const std::string& system_message) {
    OPENVINO_THROW("Qwen3-ASR does not support chat mode");
}

}  // namespace ov::genai
