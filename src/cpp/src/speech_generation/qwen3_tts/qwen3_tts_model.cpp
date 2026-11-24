// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speech_generation/qwen3_tts/qwen3_tts_model.hpp"

#include <chrono>

#include "openvino/genai/perf_metrics.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_audio_streaming.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_utils.hpp"
#include "utils.hpp"

using namespace ov::genai::qwen3_tts;

namespace ov {
namespace genai {

Qwen3TTSImpl::Qwen3TTSImpl(const std::filesystem::path& models_path,
                           const std::string& device,
                           const ov::AnyMap& properties,
                           const Tokenizer& tokenizer)
    : m_engine(models_path, device, properties, tokenizer) {
    // The talker stays on the requested device. On NPU it is routed through the
    // NPUW LLM compile path (stateful decoder -> static); every other device
    // keeps the generic path.
    auto [talker_device, talker_properties] = resolve_component_target(device, properties, roles::TALKER);
    if (talker_device.find("NPU") != std::string::npos) {
        // Apply TTS-specific static-shape sizing unless the caller already set it.
        constexpr int64_t DEFAULT_MAX_PROMPT_LEN = 1024;
        constexpr int64_t DEFAULT_MIN_RESPONSE_LEN = 128;
        ov::AnyMap npu_talker_properties = talker_properties;
        npu_talker_properties.emplace("MAX_PROMPT_LEN", DEFAULT_MAX_PROMPT_LEN);
        npu_talker_properties.emplace("MIN_RESPONSE_LEN", DEFAULT_MIN_RESPONSE_LEN);

        const auto talker_path = models_path / TALKER_NAME;
        auto talker_model = ov::genai::utils::singleton_core().read_model(talker_path);
        const auto kv_pos = ov::genai::utils::get_kv_axes_pos(talker_model);
        ov::CompiledModel compiled;
        ov::genai::utils::KVDesc kv_desc;
        std::tie(compiled, kv_desc) = ov::genai::utils::compile_decoder_for_npu(talker_model, npu_talker_properties, kv_pos);
        ov::genai::utils::print_compiled_model_properties(compiled, roles::TALKER);
        m_talker = compiled.create_infer_request();
    } else {
        const auto talker_path = models_path / TALKER_NAME;
        m_talker = compile_request(talker_path, roles::TALKER, talker_device, talker_properties);
    }

    m_engine.perf_device()[TALKER_PREFILL_PERF_NAME] = talker_device;
    m_engine.perf_device()[TALKER_GENERATE_PERF_NAME] = talker_device;
}

ov::Tensor Qwen3TTSImpl::infer_talker(const ov::Tensor& inputs_embeds,
                                      const ov::Tensor& attention_mask,
                                      const ov::Tensor& position_ids,
                                      bool reset_state) {
    if (reset_state) {
        m_talker.reset_state();
    }

    m_talker.set_tensor("inputs_embeds", inputs_embeds);
    m_talker.set_tensor("attention_mask", attention_mask);
    m_talker.set_tensor("position_ids", position_ids);

    ov::Tensor beam_idx(ov::element::i32, ov::Shape{1});
    beam_idx.data<int32_t>()[0] = 0;
    m_talker.set_tensor("beam_idx", beam_idx);

    if (reset_state) {
        const auto shape = inputs_embeds.get_shape();
        if (shape.size() >= 2) {
            m_talker_prefill_tokens += static_cast<int64_t>(shape[1]);
        }
        run_and_time([&]{ m_talker.infer(); }, TALKER_PREFILL_PERF_NAME, m_engine.perf_ms(), m_engine.perf_calls());
    } else {
        run_and_time([&]{ m_talker.infer(); }, TALKER_GENERATE_PERF_NAME, m_engine.perf_ms(), m_engine.perf_calls());
    }
    // Return the request's own logits output tensor (no clone). Valid until the
    // next m_talker.infer(). The decode loop samples from it before issuing the
    // next talker infer, so this is safe. Do NOT reorder a talker infer ahead of
    // consuming a previously returned logits tensor.
    return m_talker.get_tensor("logits");
}

Qwen3TTSImpl::TalkerRun::TalkerRun(Qwen3TTSImpl& impl,
                                   const ov::Tensor& talker_prefill,
                                   const ov::Tensor& tts_pad,
                                   const ov::Tensor& trailing_text_hidden,
                                   const SpeechGenerationConfig& generation_config,
                                   const std::vector<bool>& suppress_tokens,
                                   const ov::Tensor& ref_codec_ids,
                                   bool stream_audio,
                                   bool incremental_text)
    : m_impl(impl),
      m_engine(impl.m_engine),
      m_config(generation_config),
      m_suppress_tokens(suppress_tokens),
      m_text(generation_config.non_streaming_mode, tts_pad, trailing_text_hidden),
      m_frames(impl.m_engine, stream_audio, generation_config, ref_codec_ids),
      m_talker_rng(generation_config.rng_seed),
      m_subtalker_rng(generation_config.rng_seed != 0 ? static_cast<uint32_t>(generation_config.rng_seed)
                                                      : std::random_device{}()),
      m_absolute_pos(talker_prefill.get_shape()[1]) {
    auto prefill_mask = make_causal_attention_mask_4d(talker_prefill.get_shape()[1]);
    auto prefill_pos = make_position_ids_prefill(talker_prefill.get_shape()[1]);
    m_logits = impl.infer_talker(talker_prefill, prefill_mask, prefill_pos, true);
    // Hold the talker's own last_hidden_state output (no clone). We only read its last
    // row into past_hidden in step(), which happens before the next m_talker.infer().
    m_hidden_states = impl.m_talker.get_tensor("last_hidden_state");
    OPENVINO_ASSERT(m_hidden_states.get_shape().size() == 3, "Talker last_hidden_state must have rank 3");
    m_hidden = m_hidden_states.get_shape()[2];
    OPENVINO_ASSERT(tts_pad.get_shape().size() == 3 && tts_pad.get_shape()[2] == m_hidden,
                    "tts_pad hidden size (", tts_pad.get_shape()[2], ") must match talker hidden size (", m_hidden, ")");

    if (incremental_text) {
        OPENVINO_ASSERT(!generation_config.non_streaming_mode,
                        "Incremental text needs the streaming prompt layout (non_streaming_mode = false)");
        m_text = qwen3_tts::TrailingText::incremental(tts_pad);
        m_gate_eos = true;
        m_suppress_tokens_and_eos = m_suppress_tokens;
        m_suppress_tokens_and_eos.at(static_cast<size_t>(m_engine.ids().codec_eos_token_id)) = true;
    }

    m_max_steps = generation_config.get_max_new_tokens();
    if (m_max_steps == SIZE_MAX) {
        m_max_steps = 2048;
    }
    // Reference HF generate() computes a token's residual codes on the NEXT forward call (fed
    // that token as input), and its loop stops as soon as max_new_tokens is reached -- before
    // that extra call happens -- so the last sampled token's codes are silently dropped whenever
    // generation is capped by length rather than natural EOS. Mirror that here so max_new_tokens
    // yields the same number of audio-code frames as optimum-intel / native HF.
    if (m_max_steps > 0) {
        --m_max_steps;
    }
    m_generated_main.reserve(std::min(m_max_steps, size_t(8192)));
    m_frames.reserve(std::min(m_max_steps, size_t(8192)));
}

Qwen3TTSImpl::TalkerRun::Status Qwen3TTSImpl::TalkerRun::step(std::vector<float>& chunk) {
    if (m_done || m_step >= m_max_steps) {
        m_done = true;
        return Status::DONE;
    }
    if (m_pending) {
        if (!m_text.available(m_step - 1)) {
            return Status::NEED_TEXT;
        }
        advance();
    }

    // The logits come from a talker input that contained text rows 0 .. m_step - 1, so the row that ends the text
    // (the last one) has been seen once m_step >= rows. Before that the talker must not end the utterance.
    const bool eos_allowed = !m_gate_eos || (m_text.finished() && m_step >= m_text.rows());
    int64_t token = m_engine.sample_token_from_logits(m_logits, m_config, m_generated_main, m_suppress_tokens,
                                                      m_talker_rng);
    if (token == m_engine.ids().codec_eos_token_id && !eos_allowed) {
        // The model wants to end the utterance, but the text has not been spoken yet. The first draw is made from the
        // distribution generate() uses, so the audio matches it whenever it would not have ended here; only now is
        // EOS removed and the token drawn again.
        token = m_engine.sample_token_from_logits(m_logits, m_config, m_generated_main, m_suppress_tokens_and_eos,
                                                  m_talker_rng);
    }
    if (token == m_engine.ids().codec_eos_token_id) {
        m_done = true;
        m_eos = true;
        return Status::DONE;
    }
    m_generated_main.push_back(token);

    const size_t hs_len = m_hidden_states.get_shape()[1];
    ov::Tensor past_hidden(m_hidden_states, ov::Coordinate{0, hs_len - 1, 0}, ov::Coordinate{1, hs_len, m_hidden});

    m_groups = m_engine.generate_codec_groups(past_hidden, token, m_config, m_subtalker_rng);
    chunk = m_frames.push_frame(m_groups);
    m_pending = true;
    ++m_step;
    return Status::FRAME;
}

void Qwen3TTSImpl::TalkerRun::advance() {
    auto token_embed = m_engine.build_talker_step_input(m_groups, m_text, m_step - 1);

    auto attn = make_decode_attention_mask_4d(m_absolute_pos + 1);
    auto pos = make_position_ids_decode(m_absolute_pos);
    m_logits = m_impl.infer_talker(token_embed, attn, pos, false);
    m_hidden_states = m_impl.m_talker.get_tensor("last_hidden_state");
    ++m_absolute_pos;
    m_pending = false;
}

Text2SpeechDecodedResults Qwen3TTSImpl::decode_from_prefill(const ov::Tensor& talker_prefill,
                                                            const ov::Tensor& tts_pad,
                                                            const ov::Tensor& trailing_text_hidden,
                                                            const SpeechGenerationConfig& generation_config,
                                                            const std::vector<bool>& suppress_tokens,
                                                            const OmniSpeechStreamerVariant& streamer,
                                                            const ov::Tensor& ref_codec_ids,
                                                            std::optional<MicroSeconds> tokenization_duration) {
    Text2SpeechDecodedResults result;
    result.output_sample_rate = m_engine.output_sample_rate();

    // Audio output streaming: decode codec frames in small chunks while the talker keeps running.
    const bool streaming = is_speech_streamer_active(streamer);
    TalkerRun run(*this, talker_prefill, tts_pad, trailing_text_hidden, generation_config, suppress_tokens,
                  ref_codec_ids, streaming);

    std::vector<float> streamed_waveform;
    bool stopped_by_streamer = false;
    auto emit_chunk = [&](const std::vector<float>& samples) {
        if (samples.empty()) {
            return StreamingStatus::RUNNING;
        }
        streamed_waveform.insert(streamed_waveform.end(), samples.begin(), samples.end());
        return write_to_speech_streamer(streamer, make_chunk_tensor(samples));
    };

    std::vector<float> chunk;
    for (auto status = run.step(chunk); status != TalkerRun::Status::DONE; status = run.step(chunk)) {
        OPENVINO_ASSERT(status == TalkerRun::Status::FRAME, "fixed text never waits for more");
        if (emit_chunk(chunk) != StreamingStatus::RUNNING) {
            stopped_by_streamer = true;
            break;
        }
    }

    if (streaming && !stopped_by_streamer) {
        emit_chunk(run.finish());  // flush the tail
    }

    auto waveform = streaming ? std::move(streamed_waveform) : run.finish();

    ov::Tensor wav_tensor(ov::element::f32, ov::Shape{waveform.size()});
    if (!waveform.empty()) {
        std::copy(waveform.begin(), waveform.end(), wav_tensor.data<float>());
    }

    result.perf_metrics.num_generated_samples += wav_tensor.get_size();
    result.speeches.push_back(std::move(wav_tensor));
    if (tokenization_duration) {
        result.perf_metrics.raw_metrics.tokenization_durations.emplace_back(*tokenization_duration);
    }
    result.perf_metrics.evaluate_statistics();
    m_perf_metrics = result.perf_metrics;
    return result;
}

Qwen3TTSImpl::VoiceSetup Qwen3TTSImpl::resolve_voice(const ov::Tensor& speaker_embedding,
                                                     const SpeechGenerationConfig& generation_config) {
    VoiceSetup voice;
    voice.language = m_engine.normalize_text_language(generation_config.language);
    voice.speaker = m_engine.normalize_speaker(generation_config.speaker);
    voice.base_model = m_engine.is_base_model();
    const std::string& speaker = voice.speaker;
    const bool base_model = voice.base_model;

    if (m_engine.tts_model_type() == "voice_design") {
        OPENVINO_ASSERT(generation_config.speaker.empty(),
                        "Qwen3 VoiceDesign does not support 'speaker'. Remove speaker and use 'instruct'.");
        OPENVINO_ASSERT(!speaker_embedding,
                        "Qwen3 VoiceDesign does not accept external speaker_embedding. Use language/instruct only.");
        OPENVINO_ASSERT(!generation_config.instruct.empty(), "Qwen3 VoiceDesign requires a non-empty 'instruct' property.");
    } else if (m_engine.tts_model_type() == "custom_voice") {
        OPENVINO_ASSERT( !speaker_embedding,
            "Qwen3 CustomVoice does not accept external speaker_embedding. Use 'speaker' property instead.");
        OPENVINO_ASSERT(!speaker.empty(), "Qwen3 CustomVoice requires a non-empty 'speaker' property.");
    } else if (base_model) {
        OPENVINO_ASSERT(speaker_embedding || generation_config.ref_audio,
                        "Qwen3 Base requires a voice source: provide either 'speaker_embedding' or 'ref_audio'.");
    }

    if (!base_model &&
        (!generation_config.ref_text.empty() || static_cast<bool>(generation_config.ref_audio) ||
         static_cast<bool>(generation_config.ref_codec_ids))) {
        OPENVINO_THROW("ref_text/ref_audio/ref_codec_ids are supported only for Qwen3 Base models");
    }

    // Derive x-vector-only vs ICL mode from the inputs provided.
    // ICL mode requires ref_text (transcript) or pre-computed ref_codec_ids.
    // ref_audio alone implies x-vector-only mode (embedding extraction only, no ICL prompt).
    const bool x_vector_only_mode = base_model &&
        generation_config.ref_text.empty() &&
        !static_cast<bool>(generation_config.ref_codec_ids);

    voice.speaker_embedding = speaker_embedding;
    if (base_model && !voice.speaker_embedding && generation_config.ref_audio) {
        voice.speaker_embedding = m_engine.extract_qwen3_speaker_embedding_from_audio(generation_config.ref_audio);
    }

    voice.icl =
        base_model &&
        !x_vector_only_mode &&
        (!generation_config.ref_text.empty() ||
         static_cast<bool>(generation_config.ref_audio) ||
         static_cast<bool>(generation_config.ref_codec_ids));

    if (voice.icl) {
        OPENVINO_ASSERT(!generation_config.ref_text.empty(),
                        "Qwen3 Base ICL voice cloning requires non-empty 'ref_text'.");

        OPENVINO_ASSERT(voice.speaker_embedding,
                        "Qwen3 Base ICL voice cloning requires a speaker source: pass 'speaker_embedding' "
                        "or provide 'ref_audio'.");

        ov::Tensor resolved_ref_code = generation_config.ref_codec_ids;
        if (!resolved_ref_code && generation_config.ref_audio) {
            resolved_ref_code = m_engine.extract_qwen3_ref_code_from_audio(generation_config.ref_audio);
        }

        OPENVINO_ASSERT(resolved_ref_code,
                        "Qwen3 Base ICL voice cloning requires reference codec ids: pass 'ref_codec_ids' "
                        "or provide 'ref_audio' so ref codes can be extracted.");

        voice.ref_text = generation_config.ref_text;
        voice.ref_codec_ids = resolved_ref_code;
    }
    return voice;
}

Text2SpeechDecodedResults Qwen3TTSImpl::generate(const std::vector<std::string>& texts,
                                                 const ov::Tensor& speaker_embedding,
                                                 const SpeechGenerationConfig& generation_config,
                                                 const OmniSpeechStreamerVariant& streamer) {
    // Constructed first so end() is called exactly once on every exit path, including validation errors.
    SpeechStreamerEndGuard streamer_guard(streamer);
    OPENVINO_ASSERT(m_live_streaming_sessions.load() == 0,
                    "generate() cannot run while a Text2SpeechStreamingSession is alive: they share the talker. "
                    "Destroy the session first.");
    OPENVINO_ASSERT(!is_speech_streamer_active(streamer) || texts.size() == 1,
                    "Audio output streaming supports exactly one input text per generate() call.");
    Text2SpeechDecodedResults result;
    result.output_sample_rate = m_engine.output_sample_rate();

    auto generation_start = std::chrono::steady_clock::now();

    std::vector<bool> suppress_tokens = m_engine.compute_suppress_tokens();

    const VoiceSetup voice = resolve_voice(speaker_embedding, generation_config);
    const std::string& language = voice.language;
    const std::string& speaker = voice.speaker;
    const bool base_model = voice.base_model;
    const ov::Tensor& effective_speaker_embedding = voice.speaker_embedding;

    // Surface the resolved speaker embedding so callers can persist and reuse it (x-vector mode).
    if (base_model && effective_speaker_embedding) {
        result.speaker_embedding = effective_speaker_embedding;
    }

    // ICL voice-clone path: triggered when ref_text or ref_codec_ids is provided.
    if (voice.icl) {
        Qwen3VoiceClonePrompt prompt;
        prompt.ref_spk_embedding = effective_speaker_embedding;
        prompt.ref_text = voice.ref_text;
        prompt.ref_code = voice.ref_codec_ids;

        // Surface the resolved clone artifacts so callers can persist and reuse them (ICL mode).
        result.speaker_embedding = effective_speaker_embedding;
        result.ref_codec_ids = voice.ref_codec_ids;

        for (const auto& text : texts) {
            auto decoded = generate_voice_clone(text, prompt, generation_config, streamer);
            result.speeches.insert(result.speeches.end(), decoded.speeches.begin(), decoded.speeches.end());
            result.perf_metrics.num_generated_samples += decoded.perf_metrics.num_generated_samples;
            result.perf_metrics.raw_metrics.tokenization_durations.insert(
                result.perf_metrics.raw_metrics.tokenization_durations.end(),
                decoded.perf_metrics.raw_metrics.tokenization_durations.begin(),
                decoded.perf_metrics.raw_metrics.tokenization_durations.end());
        }

        auto generation_end = std::chrono::steady_clock::now();
        result.perf_metrics.raw_metrics.generate_durations.emplace_back(
            PerfMetrics::get_microsec(generation_end - generation_start));
        result.perf_metrics.evaluate_statistics();
        m_perf_metrics = result.perf_metrics;
        m_engine.perf_print_and_reset(m_talker_prefill_tokens);
        m_talker_prefill_tokens = 0;
        return result;
    }

    for (const auto& text : texts) {
        auto tokenization_start = std::chrono::steady_clock::now();
        const std::string assistant_text = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
        auto input = m_engine.tokenizer().encode(assistant_text);
        auto input_ids_tensor = input.input_ids;
        auto tokenization_end = std::chrono::steady_clock::now();
        result.perf_metrics.raw_metrics.tokenization_durations.emplace_back(
            PerfMetrics::get_microsec(tokenization_end - tokenization_start));

        auto prefill = m_engine.build_prefill(input_ids_tensor, effective_speaker_embedding, generation_config,
                                              language, speaker, base_model);

        auto decoded = decode_from_prefill(prefill.talker_prefill, prefill.tts_pad, prefill.trailing_text_hidden,
                                          generation_config, suppress_tokens, streamer);
        result.speeches.insert(result.speeches.end(), decoded.speeches.begin(), decoded.speeches.end());
        result.perf_metrics.num_generated_samples += decoded.perf_metrics.num_generated_samples;
    }

    auto generation_end = std::chrono::steady_clock::now();
    result.perf_metrics.raw_metrics.generate_durations.emplace_back(
        PerfMetrics::get_microsec(generation_end - generation_start));
    result.perf_metrics.evaluate_statistics();
    m_perf_metrics = result.perf_metrics;
    m_engine.perf_print_and_reset(m_talker_prefill_tokens);
    m_talker_prefill_tokens = 0;
    return result;
}

Text2SpeechDecodedResults Qwen3TTSImpl::generate_voice_clone(const std::string& text,
                                                             const Qwen3VoiceClonePrompt& prompt,
                                                             const SpeechGenerationConfig& generation_config,
                                                             const OmniSpeechStreamerVariant& streamer) {
    std::vector<bool> suppress_tokens = m_engine.compute_suppress_tokens();
    const auto voice_clone_prefill = m_engine.build_voice_clone_prefill(text,
                                                                        prompt.ref_text,
                                                                        prompt.ref_spk_embedding,
                                                                        prompt.ref_code,
                                                                        generation_config);
    return decode_from_prefill(voice_clone_prefill.prefill.talker_prefill,
                               voice_clone_prefill.prefill.tts_pad,
                               voice_clone_prefill.prefill.trailing_text_hidden,
                               generation_config,
                               suppress_tokens,
                               streamer,
                               voice_clone_prefill.ref_codec_ids,
                               voice_clone_prefill.tokenization_duration);
}

ov::Shape Qwen3TTSImpl::get_speaker_embedding_shape() const {
    if (m_engine.is_base_model()) {
        return ov::Shape{1, 1, m_engine.speaker_embedding_dim()};
    }
    // Qwen3 CustomVoice does not consume external speaker embeddings in the GenAI API.
    return ov::Shape{};
}

}  // namespace genai
}  // namespace ov
