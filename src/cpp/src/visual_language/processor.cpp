// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/processor.hpp"

#include <chrono>
#include <mutex>

#include "utils.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/processor_bridge.hpp"
#include "visual_language/vision_registry.hpp"
#include "visual_language/vlm_chat_context.hpp"
#include "visual_language/vlm_utils.hpp"

namespace ov::genai {

namespace {

// Saves the embedder's cache state on construction and restores it on destruction, so a stateless
// process() call does not change state within shared InputsEmbedder.
// Temporary solution until cache state is owned by pipeline.
class ScopedCacheState {
public:
    explicit ScopedCacheState(utils::CacheState& cache_state)
        : m_cache_state(cache_state), m_saved(cache_state) {
        m_cache_state.reset_state();
    }
    ~ScopedCacheState() {
        m_cache_state = m_saved;
    }
    ScopedCacheState(const ScopedCacheState&) = delete;
    ScopedCacheState& operator=(const ScopedCacheState&) = delete;

private:
    utils::CacheState& m_cache_state;
    utils::CacheState m_saved;
};

}  // namespace

class VLMProcessor::VLMProcessorImpl {
public:
    explicit VLMProcessorImpl(std::shared_ptr<InputsEmbedder> inputs_embedder)
        : m_inputs_embedder(std::move(inputs_embedder)),
          m_vision_registry(std::make_shared<VisionRegistry>()) {}

    ProcessedInputs process(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const std::vector<VideoMetadata>& videos_metadata
    ) {
        VLMPerfMetrics metrics;
        const auto prepare_start = std::chrono::steady_clock::now();

        std::lock_guard<std::mutex> lock(m_process_mutex);
        // snapshot + restore cache state for shared inputs embedder
        ScopedCacheState cache_guard(m_inputs_embedder->get_cache_state());
        // prompt expected to be already templated
        m_inputs_embedder->set_apply_chat_template_status(false);

        const auto vision_start = std::chrono::steady_clock::now();
        std::vector<EncodedImage> encoded_images = m_inputs_embedder->encode_images(images);
        std::vector<EncodedVideo> encoded_videos = m_inputs_embedder->encode_videos(videos, videos_metadata);
        PerfMetrics::emplace_duration(metrics.vlm_raw_metrics.vision_encoding_durations, vision_start);

        vlm_utils::update_image_slice_counts(metrics, encoded_images);

        std::vector<ov::genai::EncodedAudio> encoded_audios;

        // TODO Support audios encodeing
        auto [unified_prompt, image_sequence, video_sequence, audio_sequence] =
            m_inputs_embedder->normalize_prompt(prompt, 0, 0, 0, encoded_images, encoded_videos, encoded_audios);

        const bool recalculate_merged_embeddings = !encoded_images.empty() || !encoded_videos.empty();

        ProcessedInputs result;
        result.inputs_embeds = m_inputs_embedder->get_inputs_embeds(
            unified_prompt,
            encoded_images,
            encoded_videos,
            metrics,
            recalculate_merged_embeddings,
            image_sequence,
            video_sequence,
            {}
        );

        const size_t sequence_length = result.inputs_embeds.get_shape().at(1);

        // VLMProcessor is stateless - stateless prefill for position_ids / attention_mask
        std::tie(result.position_ids, result.rope_delta) =
            m_inputs_embedder->get_position_ids(sequence_length, 0);

        result.attention_mask = ov::Tensor(ov::element::i64, {1, sequence_length});
        std::fill_n(result.attention_mask.data<int64_t>(), result.attention_mask.get_size(), int64_t{1});

        result.lm_extra_inputs = vlm_utils::deep_copy_tensors_map(m_inputs_embedder->get_lm_extra_inputs());

        PerfMetrics::emplace_duration(metrics.vlm_raw_metrics.prepare_embeddings_durations, prepare_start);
        result.vlm_perf_metrics = VLMPerfMetrics(metrics);

        return result;
    }

    ProcessedInputs process(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const std::vector<VideoMetadata>& videos_metadata
    ) {
        VLMPerfMetrics metrics;
        const auto prepare_start = std::chrono::steady_clock::now();

        std::lock_guard<std::mutex> lock(m_process_mutex);
        // snapshot + restore cache state for shared inputs embedder
        ScopedCacheState cache_guard(m_inputs_embedder->get_cache_state());

        // Chat history internal state is tracked per-history via VLMChatContext
        VLMChatContext chat_context(history, m_vision_registry, *m_inputs_embedder);
        VLMChatContext::ProcessedChatData chat_data = chat_context.process(images, videos, videos_metadata);
        metrics.vlm_raw_metrics.vision_encoding_durations.emplace_back(chat_data.vision_encoding_duration);
        vlm_utils::update_image_slice_counts(metrics, chat_data.encoded_images);

        constexpr bool add_generation_prompt = true;
        const std::string templated_prompt = m_inputs_embedder->get_tokenizer().apply_chat_template(
            chat_data.normalized_history, add_generation_prompt);

        m_inputs_embedder->set_apply_chat_template_status(false);

        const bool recalculate_merged_embeddings =
            !chat_data.encoded_images.empty() || !chat_data.encoded_videos.empty();

        ProcessedInputs result;
        result.inputs_embeds = m_inputs_embedder->get_inputs_embeds(
            templated_prompt,
            chat_data.encoded_images,
            chat_data.encoded_videos,
            metrics,
            recalculate_merged_embeddings,
            chat_data.image_sequence,
            chat_data.video_sequence,
            chat_data.vision_counts
        );

        const size_t sequence_length = result.inputs_embeds.get_shape().at(1);

        // VLMProcessor is stateless - stateless prefill for position_ids / attention_mask
        std::tie(result.position_ids, result.rope_delta) =
            m_inputs_embedder->get_position_ids(sequence_length, 0);

        result.attention_mask = ov::Tensor(ov::element::i64, {1, sequence_length});
        std::fill_n(result.attention_mask.data<int64_t>(), result.attention_mask.get_size(), int64_t{1});

        result.lm_extra_inputs = vlm_utils::deep_copy_tensors_map(m_inputs_embedder->get_lm_extra_inputs());

        PerfMetrics::emplace_duration(metrics.vlm_raw_metrics.prepare_embeddings_durations, prepare_start);
        result.vlm_perf_metrics = VLMPerfMetrics(metrics);

        return result;
    }

    Tokenizer get_tokenizer() const {
        return m_inputs_embedder->get_tokenizer();
    }

    std::shared_ptr<InputsEmbedder> get_inputs_embedder() const {
        return m_inputs_embedder;
    }

private:
    std::shared_ptr<InputsEmbedder> m_inputs_embedder;
    std::shared_ptr<VisionRegistry> m_vision_registry;

    // Serializes process() calls that mutate shared InputsEmbedder
    std::mutex m_process_mutex;
};

VLMProcessor::VLMProcessor(
    const std::filesystem::path& models_path,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& properties
) : m_pimpl(std::make_unique<VLMProcessorImpl>(
        std::make_shared<InputsEmbedder>(models_path, tokenizer, device, properties))) {}

VLMProcessor::VLMProcessor(
    const std::filesystem::path& models_path,
    const std::string& device,
    const ov::AnyMap& properties
) : m_pimpl(std::make_unique<VLMProcessorImpl>(
        std::make_shared<InputsEmbedder>(models_path, device, properties))) {}

VLMProcessor::VLMProcessor(
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap& properties
) : m_pimpl(std::make_unique<VLMProcessorImpl>(
        std::make_shared<InputsEmbedder>(models_map, tokenizer, config_dir_path, device, properties))) {}

VLMProcessor::~VLMProcessor() = default;

ProcessedInputs VLMProcessor::process(
    const std::string& prompt,
    const std::vector<ov::Tensor>& images,
    const std::vector<ov::Tensor>& videos,
    const std::vector<VideoMetadata>& videos_metadata
) {
    return m_pimpl->process(prompt, images, videos, videos_metadata);
}

ProcessedInputs VLMProcessor::process(
    const ChatHistory& history,
    const std::vector<ov::Tensor>& images,
    const std::vector<ov::Tensor>& videos,
    const std::vector<VideoMetadata>& videos_metadata
) {
    return m_pimpl->process(history, images, videos, videos_metadata);
}

Tokenizer VLMProcessor::get_tokenizer() const {
    return m_pimpl->get_tokenizer();
}

std::shared_ptr<InputsEmbedder> get_shared_inputs_embedder(const VLMProcessor& processor) {
    return processor.m_pimpl->get_inputs_embedder();
}

} // namespace ov::genai
