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

    auto prefill_mask = make_causal_attention_mask_4d(talker_prefill.get_shape()[1]);
    auto prefill_pos = make_position_ids_prefill(talker_prefill.get_shape()[1]);
    auto logits = infer_talker(talker_prefill, prefill_mask, prefill_pos, true);
    // Hold the talker's own last_hidden_state output (no clone). We only read its last
    // row into past_hidden below, which happens before the next m_talker.infer().
    const char* hidden_out_name = "last_hidden_state";
    auto hidden_states = m_talker.get_tensor(hidden_out_name);
    OPENVINO_ASSERT(hidden_states.get_shape().size() == 3, "Talker last_hidden_state must have rank 3");
    const size_t hidden = hidden_states.get_shape()[2];
    OPENVINO_ASSERT(tts_pad.get_shape().size() == 3 && tts_pad.get_shape()[2] == hidden,
                    "tts_pad hidden size (", tts_pad.get_shape()[2], ") must match talker hidden size (", hidden, ")");

    std::vector<int64_t> generated_main;
    std::vector<int64_t> all_codes;
    size_t max_steps = generation_config.get_max_new_tokens();
    if (max_steps == SIZE_MAX) {
        max_steps = 2048;
    }
    // Reference HF generate() computes a token's residual codes on the NEXT forward call (fed
    // that token as input), and its loop stops as soon as max_new_tokens is reached -- before
    // that extra call happens -- so the last sampled token's codes are silently dropped whenever
    // generation is capped by length rather than natural EOS. Mirror that here so max_new_tokens
    // yields the same number of audio-code frames as optimum-intel / native HF.
    if (max_steps > 0) {
        --max_steps;
    }
    generated_main.reserve(std::min(max_steps, size_t(8192)));
    all_codes.reserve(std::min(max_steps, size_t(8192)) * m_engine.ids().num_code_groups);

    const qwen3_tts::TrailingText text(generation_config.non_streaming_mode, tts_pad, trailing_text_hidden);
    std::mt19937 talker_rng(generation_config.rng_seed);
    std::mt19937 subtalker_rng(generation_config.rng_seed != 0 ? static_cast<uint32_t>(generation_config.rng_seed)
                                                             : std::random_device{}());
    size_t absolute_pos = talker_prefill.get_shape()[1];

    // Audio output streaming: decode codec frames in small chunks while the talker keeps running.
    const bool streaming = is_speech_streamer_active(streamer);
    std::optional<AudioChunker> chunker;
    std::vector<float> streamed_waveform;
    bool stopped_by_streamer = false;
    if (streaming) {
        chunker.emplace(m_engine,
                        generation_config.audio_first_chunk_frames,
                        generation_config.audio_chunk_frames,
                        generation_config.audio_chunk_left_context_frames,
                        ref_codec_ids);
    }
    auto emit_chunk = [&](const std::vector<float>& samples) {
        if (samples.empty()) {
            return StreamingStatus::RUNNING;
        }
        streamed_waveform.insert(streamed_waveform.end(), samples.begin(), samples.end());
        return write_to_speech_streamer(streamer, make_chunk_tensor(samples));
    };

    for (size_t step = 0; step < max_steps; ++step) {
        int64_t token = m_engine.sample_token_from_logits(logits, generation_config, generated_main, suppress_tokens, talker_rng);
        if (token == m_engine.ids().codec_eos_token_id) {
            break;
        }
        generated_main.push_back(token);

        const size_t hs_len = hidden_states.get_shape()[1];
        ov::Tensor past_hidden(hidden_states, ov::Coordinate{0, hs_len - 1, 0}, ov::Coordinate{1, hs_len, hidden});

        auto groups = m_engine.generate_codec_groups(past_hidden, token, generation_config, subtalker_rng);
        all_codes.insert(all_codes.end(), groups.begin(), groups.end());
        if (chunker) {
            chunker->push_frame(groups);
            if (chunker->chunk_ready()) {
                const auto status = emit_chunk(chunker->take_pending());
                if (status != StreamingStatus::RUNNING) {
                    stopped_by_streamer = true;
                    break;
                }
            }
        }

        auto token_embed = m_engine.build_talker_step_input(groups, text, step);

        auto attn = make_decode_attention_mask_4d(absolute_pos + 1);
        auto pos = make_position_ids_decode(absolute_pos);
        logits = infer_talker(token_embed, attn, pos, false);
        hidden_states = m_talker.get_tensor(hidden_out_name);
        ++absolute_pos;
    }

    if (chunker && !stopped_by_streamer) {
        emit_chunk(chunker->flush());  // flush the tail
    }

    auto waveform = streaming
        ? std::move(streamed_waveform)
        : (ref_codec_ids ? m_engine.decode_speech_tokenizer_with_reference(ref_codec_ids, all_codes)
                         : m_engine.decode_speech_tokenizer(all_codes));

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

Text2SpeechDecodedResults Qwen3TTSImpl::generate(const std::vector<std::string>& texts,
                                                 const ov::Tensor& speaker_embedding,
                                                 const SpeechGenerationConfig& generation_config,
                                                 const OmniSpeechStreamerVariant& streamer) {
    // Constructed first so end() is called exactly once on every exit path, including validation errors.
    SpeechStreamerEndGuard streamer_guard(streamer);
    OPENVINO_ASSERT(!is_speech_streamer_active(streamer) || texts.size() == 1,
                    "Audio output streaming supports exactly one input text per generate() call.");
    Text2SpeechDecodedResults result;
    result.output_sample_rate = m_engine.output_sample_rate();

    auto generation_start = std::chrono::steady_clock::now();

    std::vector<bool> suppress_tokens = m_engine.compute_suppress_tokens();

    const std::string language = m_engine.normalize_text_language(generation_config.language);
    const std::string speaker = m_engine.normalize_speaker(generation_config.speaker);
    const bool base_model = m_engine.is_base_model();

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

    ov::Tensor effective_speaker_embedding = speaker_embedding;
    if (base_model && !effective_speaker_embedding && generation_config.ref_audio) {
        effective_speaker_embedding = m_engine.extract_qwen3_speaker_embedding_from_audio(generation_config.ref_audio);
    }

    // Surface the resolved speaker embedding so callers can persist and reuse it (x-vector mode).
    if (base_model && effective_speaker_embedding) {
        result.speaker_embedding = effective_speaker_embedding;
    }

    // ICL voice-clone path: triggered when ref_text or ref_codec_ids is provided.
    const bool has_qwen_voice_clone_props =
        base_model &&
        !x_vector_only_mode &&
        (!generation_config.ref_text.empty() ||
         static_cast<bool>(generation_config.ref_audio) ||
         static_cast<bool>(generation_config.ref_codec_ids));

    if (has_qwen_voice_clone_props) {
        OPENVINO_ASSERT(!generation_config.ref_text.empty(),
                        "Qwen3 Base ICL voice cloning requires non-empty 'ref_text'.");

        OPENVINO_ASSERT(effective_speaker_embedding,
                        "Qwen3 Base ICL voice cloning requires a speaker source: pass 'speaker_embedding' "
                        "or provide 'ref_audio'.");

        ov::Tensor resolved_ref_code = generation_config.ref_codec_ids;
        if (!resolved_ref_code && generation_config.ref_audio) {
            resolved_ref_code = m_engine.extract_qwen3_ref_code_from_audio(generation_config.ref_audio);
        }

        OPENVINO_ASSERT(resolved_ref_code,
                        "Qwen3 Base ICL voice cloning requires reference codec ids: pass 'ref_codec_ids' "
                        "or provide 'ref_audio' so ref codes can be extracted.");

        Qwen3VoiceClonePrompt prompt;
        prompt.ref_spk_embedding = effective_speaker_embedding;
        prompt.ref_text = generation_config.ref_text;
        prompt.ref_code = resolved_ref_code;

        // Surface the resolved clone artifacts so callers can persist and reuse them (ICL mode).
        result.speaker_embedding = effective_speaker_embedding;
        result.ref_codec_ids = resolved_ref_code;

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
