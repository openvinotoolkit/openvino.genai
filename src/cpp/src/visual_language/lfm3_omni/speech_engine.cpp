// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/lfm3_omni/speech_engine.hpp"

#include <algorithm>
#include <chrono>

#include "sampling/logit_processor.hpp"
#include "sampling/sampler.hpp"
#include "sampling/sampling_distribution.hpp"

namespace ov::genai {

LFM3OmniSpeechEngine::LFM3OmniSpeechEngine(const std::filesystem::path& models_dir,
                                           ov::InferRequest& language,
                                           EmbeddingsModel::Ptr embedding,
                                           const std::string& device,
                                           const ov::AnyMap& properties)
    : LFM3OmniSpeechEngine(LFM3SpeechModels(models_dir, device, properties), language, std::move(embedding)) {}

LFM3OmniSpeechEngine::LFM3OmniSpeechEngine(const ModelsMap& models_map,
                                           const std::filesystem::path& config_dir_path,
                                           ov::InferRequest& language,
                                           EmbeddingsModel::Ptr embedding,
                                           const std::string& device,
                                           const ov::AnyMap& properties)
    : LFM3OmniSpeechEngine(LFM3SpeechModels(models_map, config_dir_path, device, properties),
                           language,
                           std::move(embedding)) {}

LFM3OmniSpeechEngine::LFM3OmniSpeechEngine(LFM3SpeechModels models,
                                           ov::InferRequest& language,
                                           EmbeddingsModel::Ptr embedding)
    : m_models(std::move(models)),
      m_language(language),
      m_embedding(std::move(embedding)) {
    m_beam_idx.data<int32_t>()[0] = 0;
}

LFM3OmniSpeechEngine::StepOutput LFM3OmniSpeechEngine::language_step(const ov::Tensor& inputs_embeds,
                                                                      size_t attention_length) {
    // set_shape keeps the allocation while the mask fits, so the decode loop does not allocate per step.
    m_attention_mask.set_shape({1, attention_length});
    std::fill_n(m_attention_mask.data<int64_t>(), attention_length, 1);
    m_language.set_tensor("inputs_embeds", inputs_embeds);
    m_language.set_tensor("attention_mask", m_attention_mask);
    m_language.set_tensor("beam_idx", m_beam_idx);
    m_language.infer();

    const ov::Tensor logits = m_language.get_tensor("logits");
    const ov::Tensor hidden_states = m_language.get_tensor("hidden_states");
    OPENVINO_ASSERT(logits.get_element_type() == ov::element::f32 && hidden_states.get_element_type() == ov::element::f32,
                    "LFM3-Omni language model must output f32 logits and hidden_states");
    // Slice-before-matmul cuts logits to the last position, while hidden_states keeps every prompt position.
    const size_t logits_len = logits.get_shape().at(1);
    const size_t hidden_len = hidden_states.get_shape().at(1);
    const size_t vocab_size = logits.get_shape().at(2);
    const size_t hidden_size = hidden_states.get_shape().at(2);
    const float* last_logits = logits.data<const float>() + (logits_len - 1) * vocab_size;
    const float* last_hidden = hidden_states.data<const float>() + (hidden_len - 1) * hidden_size;

    StepOutput output;
    output.logits.assign(last_logits, last_logits + vocab_size);
    output.hidden_state = ov::Tensor(ov::element::f32, {1, hidden_size});
    std::copy_n(last_hidden, hidden_size, output.hidden_state.data<float>());
    return output;
}

LFM3SpeechResult LFM3OmniSpeechEngine::generate(const ov::Tensor& inputs_embeds,
                                                const ov::Tensor& attention_mask,
                                                int64_t last_prompt_token,
                                                const LFM3SpeechRequest& request,
                                                const std::shared_ptr<StreamerBase>& streamer,
                                                const OmniSpeechStreamerVariant& speech_streamer) {
    LFM3SpeechState state(m_models.config(), request, speech_streamer);
    // The continuous batching sampler with the rules it gets on PA, so both backends draw the same text tokens.
    auto text_rules = lfm3_text_sampling_overrides(m_models.config());
    text_rules.forced_token = [&state] {
        return state.forced_token();
    };
    LogitProcessor text_processor(request.text_config, {}, nullptr, text_rules);
    std::mt19937 text_rng(request.text_config.rng_seed);
    size_t generated_slots = 0;

    ov::Tensor prompt_embeds = inputs_embeds;
    const int64_t marker = state.take_prompt_marker(last_prompt_token);
    if (marker >= 0) {
        prompt_embeds = append_lfm3_prompt_row(inputs_embeds, embed_lfm3_token(*m_embedding, marker));
    }
    size_t attention_length = attention_mask.get_shape().at(1) + (marker >= 0 ? 1 : 0);
    state.start(attention_length);

    RawPerfMetrics raw_perf;
    raw_perf.m_inference_durations = {MicroSeconds(0.0f)};
    auto timed_step = [&](const ov::Tensor& embeds, size_t length) {
        const auto start = std::chrono::steady_clock::now();
        StepOutput output = language_step(embeds, length);
        const float duration = PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start);
        raw_perf.m_inference_durations[0] += MicroSeconds(duration);
        raw_perf.m_token_infer_durations.emplace_back(duration);
        return output;
    };
    StepOutput step = timed_step(prompt_embeds, attention_length);

    // As on PA: a failed call still ends a healthy text streamer, and an end() failure is dropped so the original
    // error reaches the caller. A text streamer that threw itself is not called again.
    struct TextStreamerGuard {
        const std::shared_ptr<StreamerBase>& streamer;
        bool done = false;
        ~TextStreamerGuard() {
            if (streamer && !done) {
                try {
                    streamer->end();
                } catch (...) {
                }
            }
        }
    } text_streamer_guard{streamer};
    while (!state.finished()) {
        ov::Tensor next_embeds;
        text_processor.update_generated_len(generated_slots++);
        // A frame slot draws its forced <|audio_out|> too, as on PA, so the random engine stays in step.
        Logits logits(step.logits.data(), step.logits.size());
        text_processor.apply(logits);
        const Token token = request.text_config.is_greedy_decoding()
                                ? greedy_sample(logits, request.text_config.logprobs)
                                : SamplingDistribution(logits).sample(text_rng);
        text_processor.register_new_generated_token(token.m_index);
        if (state.next_is_frame()) {
            OPENVINO_ASSERT(token.m_index == m_models.config().audio_out_token_id,
                            "LFM3-Omni frame slot sampled token ",
                            token.m_index,
                            " instead of the forced <|audio_out|>");
            next_embeds =
                m_models.embed_frame(state.accept_frame(m_models.sample_frame(step.hidden_state, state), m_models));
        } else {
            StreamingStatus status = StreamingStatus::RUNNING;
            if (streamer) {
                text_streamer_guard.done = true;
                status = streamer->write(token.m_index);
                text_streamer_guard.done = false;
            }
            state.accept_text(token.m_index, token.m_log_prob, status);
            next_embeds = embed_lfm3_token(*m_embedding, token.m_index);
        }
        raw_perf.m_new_token_times.emplace_back(std::chrono::steady_clock::now());
        raw_perf.m_batch_sizes.emplace_back(1);
        if (!state.finished()) {
            // The last sampled slot needs no hidden state, so its decode step is skipped.
            step = timed_step(next_embeds, ++attention_length);
        }
    }
    text_streamer_guard.done = true;
    if (streamer) {
        streamer->end();
    }
    LFM3SpeechResult result = state.finish(m_models);
    result.raw_perf_metrics = std::move(raw_perf);
    return result;
}

}  // namespace ov::genai
