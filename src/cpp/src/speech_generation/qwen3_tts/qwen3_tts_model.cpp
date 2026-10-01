// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speech_generation/qwen3_tts/qwen3_tts_model.hpp"

#include <chrono>

#include "openvino/genai/perf_metrics.hpp"
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
                                                            const ov::Tensor& ref_codec_ids) {
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

    std::mt19937 rng(generation_config.rng_seed != 0 ? static_cast<uint32_t>(generation_config.rng_seed) : std::random_device{}());
    size_t absolute_pos = talker_prefill.get_shape()[1];
    for (size_t step = 0; step < max_steps; ++step) {
        int64_t token = m_engine.sample_token_from_logits(logits, generation_config, generated_main, suppress_tokens, rng);
        if (token == m_engine.ids().codec_eos_token_id) {
            break;
        }
        generated_main.push_back(token);

        const size_t hs_len = hidden_states.get_shape()[1];
        ov::Tensor past_hidden(hidden_states, ov::Coordinate{0, hs_len - 1, 0}, ov::Coordinate{1, hs_len, hidden});

        auto groups = m_engine.generate_codec_groups(past_hidden, token, generation_config, rng);
        all_codes.insert(all_codes.end(), groups.begin(), groups.end());

        auto token_embed = m_engine.infer_embedding(m_engine.talker_embedding(), groups[0]);
        OPENVINO_ASSERT(token_embed.get_shape().size() == 3 && token_embed.get_shape()[2] == hidden,
                        "Talker embedding hidden size (", token_embed.get_shape()[2],
                        ") must match talker hidden size (", hidden, ")");
        float* token_embed_ptr = token_embed.data<float>();
        for (size_t g = 1; g < groups.size(); ++g) {
            auto pred_emb = m_engine.infer_predictor_embedding(groups[g], static_cast<int64_t>(g - 1));
            OPENVINO_ASSERT(pred_emb.get_shape().size() == 3 && pred_emb.get_shape()[1] == 1 &&
                                pred_emb.get_shape()[2] == token_embed.get_shape()[2],
                            "Code predictor residual embedding shape mismatch. Expected [1, 1, ",
                            token_embed.get_shape()[2], "], got ", pred_emb.get_shape());
            const float* pred_ptr = pred_emb.data<const float>();
            for (size_t h_i = 0; h_i < token_embed.get_shape()[2]; ++h_i) {
                token_embed_ptr[h_i] += pred_ptr[h_i];
            }
        }
        if (generation_config.non_streaming_mode) {
            const float* tts_pad_ptr = tts_pad.data<const float>();
            for (size_t h_i = 0; h_i < token_embed.get_shape()[2]; ++h_i) {
                token_embed_ptr[h_i] += tts_pad_ptr[h_i];
            }
        } else if (trailing_text_hidden) {
            OPENVINO_ASSERT(trailing_text_hidden.get_shape().size() == 3,
                            "trailing_text_hidden is expected to have shape [1, T, H]");
            OPENVINO_ASSERT(trailing_text_hidden.get_shape()[0] == 1,
                            "trailing_text_hidden batch size is expected to be 1");
            OPENVINO_ASSERT(trailing_text_hidden.get_shape()[2] == token_embed.get_shape()[2],
                            "trailing_text_hidden hidden size mismatch");
            const size_t trailing_len = trailing_text_hidden.get_shape()[1];
            const float* trailing_ptr = trailing_text_hidden.data<const float>();
            const float* add_ptr = nullptr;
            if (step < trailing_len) {
                add_ptr = trailing_ptr + step * token_embed.get_shape()[2];
            } else {
                add_ptr = tts_pad.data<const float>();
            }
            for (size_t h_i = 0; h_i < token_embed.get_shape()[2]; ++h_i) {
                token_embed_ptr[h_i] += add_ptr[h_i];
            }
        }

        auto attn = make_decode_attention_mask_4d(absolute_pos + 1);
        auto pos = make_position_ids_decode(absolute_pos);
        logits = infer_talker(token_embed, attn, pos, false);
        hidden_states = m_talker.get_tensor(hidden_out_name);
        ++absolute_pos;
    }

    std::vector<int64_t> codes_for_decode;
    size_t ref_frames = 0;
    if (ref_codec_ids) {
        const ov::Shape ref_shape = ref_codec_ids.get_shape();
        OPENVINO_ASSERT(ref_codec_ids.get_element_type() == ov::element::i64,
                        "Qwen3 Base ref_codec_ids tensor must have element type i64");
        OPENVINO_ASSERT(ref_shape.size() == 2 || ref_shape.size() == 3,
                        "Qwen3 Base ref_codec_ids tensor is expected to have shape [T, G] or [1, T, G]");
        const bool rank3 = ref_shape.size() == 3;
        OPENVINO_ASSERT(!rank3 || ref_shape[0] == 1,
                        "Qwen3 Base ref_codec_ids rank-3 tensor must have batch dimension 1");

        ref_frames = rank3 ? ref_shape[1] : ref_shape[0];
        const size_t ref_groups = rank3 ? ref_shape[2] : ref_shape[1];
        OPENVINO_ASSERT(ref_groups == m_engine.ids().num_code_groups,
                        "Qwen3 Base ref_codec_ids group count mismatch. Expected ",
                        m_engine.ids().num_code_groups,
                        ", got ",
                        ref_groups);

        const int64_t* ref_ptr = ref_codec_ids.data<const int64_t>();
        codes_for_decode.reserve(ref_frames * ref_groups + all_codes.size());
        for (size_t t = 0; t < ref_frames; ++t) {
            for (size_t g = 0; g < ref_groups; ++g) {
                codes_for_decode.push_back(ref_ptr[t * ref_groups + g]);
            }
        }
        codes_for_decode.insert(codes_for_decode.end(), all_codes.begin(), all_codes.end());
    } else {
        codes_for_decode = all_codes;
    }

    auto waveform = m_engine.decode_speech_tokenizer(codes_for_decode);
    if (ref_frames > 0) {
        const size_t generated_frames = all_codes.size() / m_engine.ids().num_code_groups;
        const size_t total_frames = ref_frames + generated_frames;
        const size_t cut_samples = static_cast<size_t>(
            (static_cast<double>(ref_frames) / static_cast<double>(std::max<size_t>(total_frames, 1))) *
            static_cast<double>(waveform.size()));
        if (cut_samples >= waveform.size()) {
            waveform.clear();
        } else {
            waveform.erase(waveform.begin(), waveform.begin() + static_cast<std::ptrdiff_t>(cut_samples));
        }
    }

    ov::Tensor wav_tensor(ov::element::f32, ov::Shape{waveform.size()});
    if (!waveform.empty()) {
        std::copy(waveform.begin(), waveform.end(), wav_tensor.data<float>());
    }

    result.perf_metrics.num_generated_samples += wav_tensor.get_size();
    result.speeches.push_back(std::move(wav_tensor));
    result.perf_metrics.evaluate_statistics();
    m_perf_metrics = result.perf_metrics;
    return result;
}

Text2SpeechDecodedResults Qwen3TTSImpl::generate(const std::vector<std::string>& texts,
                                                 const ov::Tensor& speaker_embedding,
                                                 const SpeechGenerationConfig& generation_config) {
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
            auto decoded = generate_voice_clone(text, prompt, generation_config);
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
                                          generation_config, suppress_tokens);
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
                                                             const SpeechGenerationConfig& generation_config) {
    OPENVINO_ASSERT(m_engine.is_base_model(), "Qwen3 voice-clone prompt generation is supported only for Base models");
    OPENVINO_ASSERT(!prompt.ref_text.empty(), "Qwen3 voice-clone ICL mode requires a non-empty reference transcript");
    OPENVINO_ASSERT(!prompt.ref_code.get_shape().empty(), "Qwen3 voice-clone ICL mode requires reference codec ids");
    OPENVINO_ASSERT(prompt.ref_code.get_element_type() == ov::element::i64,
                    "Qwen3 voice-clone ref_code tensor must have element type i64");
    OPENVINO_ASSERT(prompt.ref_code.get_shape().size() == 2 || prompt.ref_code.get_shape().size() == 3,
                    "Qwen3 voice-clone prompt expects ref_code shaped [T, G] or [1, T, G]");

    const std::string language = m_engine.normalize_text_language(generation_config.language);
    const std::string speaker = m_engine.normalize_speaker(generation_config.speaker);

    std::vector<bool> suppress_tokens = m_engine.compute_suppress_tokens();

    auto result = Text2SpeechDecodedResults{};
    result.output_sample_rate = m_engine.output_sample_rate();

    // Tokenize the reference and target turns SEPARATELY, then concatenate token ids,
    // mirroring upstream qwen_tts (ref_id[:, 3:-2] ++ text_id[:, 3:-5]). Gluing the raw
    // strings before tokenization could merge tokens across the ref/target seam (BPE
    // boundary effects) and diverge from the reference.
    const std::string ref_turn = "<|im_start|>assistant\n" + prompt.ref_text + "<|im_end|>\n";
    const std::string target_turn = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";

    auto ref_input_ids = m_engine.tokenizer().encode(ref_turn).input_ids;
    auto target_input_ids = m_engine.tokenizer().encode(target_turn).input_ids;
    OPENVINO_ASSERT(ref_input_ids.get_element_type() == ov::element::i64 &&
                        target_input_ids.get_element_type() == ov::element::i64,
                    "Qwen3-TTS tokenizer must produce i64 input_ids tensor");
    OPENVINO_ASSERT(ref_input_ids.get_shape().size() == 2 && ref_input_ids.get_shape()[0] == 1 &&
                        target_input_ids.get_shape().size() == 2 && target_input_ids.get_shape()[0] == 1,
                    "Expected input_ids shape [1, T]");

    // Each turn is stripped of its 3-token role prefix (<|im_start|>assistant\n) and its
    // trailing structural tokens: the reference turn drops <|im_end|>\n (2 tokens); the
    // target turn drops <|im_end|>\n<|im_start|>assistant\n (5 tokens).
    const size_t ref_ids_len = ref_input_ids.get_shape()[1];
    const size_t target_ids_len = target_input_ids.get_shape()[1];
    OPENVINO_ASSERT(ref_ids_len > 5, "Qwen3 voice-clone reference transcript tokenized too short");
    OPENVINO_ASSERT(target_ids_len > 8, "Qwen3 voice-clone target text tokenized too short");
    const int64_t* ref_ids = ref_input_ids.data<const int64_t>();
    const int64_t* target_ids = target_input_ids.data<const int64_t>();

    // Role prefix taken from the target turn (matches upstream input_id[:, :3]).
    const std::vector<int64_t> role_ids(target_ids, target_ids + 3);
    // Content = ref_content ++ target_content (token ids), tokenized independently above.
    std::vector<int64_t> icl_content_ids;
    icl_content_ids.reserve((ref_ids_len - 5) + (target_ids_len - 8));
    icl_content_ids.insert(icl_content_ids.end(), ref_ids + 3, ref_ids + (ref_ids_len - 2));
    icl_content_ids.insert(icl_content_ids.end(), target_ids + 3, target_ids + (target_ids_len - 5));

    auto speaker_and_codec_prefill = m_engine.make_speaker_and_codec_prefill_ids(language, speaker);
    auto codec_prefill_embed0 = m_engine.infer_embedding_seq(m_engine.talker_embedding(), speaker_and_codec_prefill);
    // Split codec_pad and codec_bos: they occupy different structural roles in the prefill.
    auto codec_pad_embed1 = m_engine.infer_embedding_seq(m_engine.talker_embedding(), {m_engine.ids().codec_pad_id});
    auto codec_bos_embed = m_engine.infer_embedding_seq(m_engine.talker_embedding(), {m_engine.ids().codec_bos_id});
    auto special_embeds = m_engine.infer_special_embeds();
    const size_t hidden = special_embeds.hidden;
    const ov::Tensor& tts_bos = special_embeds.bos;
    const ov::Tensor& tts_eos = special_embeds.eos;
    const ov::Tensor& tts_pad = special_embeds.pad;

    ov::Tensor speaker_embed = m_engine.normalize_external_speaker_embedding(prompt.ref_spk_embedding, codec_prefill_embed0.get_shape()[2]);
    // codec_input_no_bos = [codec_prefill0, speaker, codec_pad] — codec_bos is placed separately at start of codec_side
    ov::Tensor codec_input_no_bos = concat_embed({codec_prefill_embed0, speaker_embed, codec_pad_embed1});
    OPENVINO_ASSERT(codec_input_no_bos.get_shape()[2] == hidden,
                    "Qwen3 talker embedding hidden size (", codec_input_no_bos.get_shape()[2],
                    ") must match talker text embedding hidden size (", hidden, ")");

    ov::Shape ref_code_shape = prompt.ref_code.get_shape();
    const bool ref_code_rank3 = ref_code_shape.size() == 3;
    OPENVINO_ASSERT(!ref_code_rank3 || ref_code_shape[0] == 1,
                    "Qwen3 voice-clone ref_code rank-3 tensor must have batch dimension 1");
    const size_t ref_len = ref_code_rank3 ? ref_code_shape[1] : ref_code_shape[0];
    const size_t num_code_groups = ref_code_rank3 ? ref_code_shape[2] : ref_code_shape[1];
    OPENVINO_ASSERT(ref_len > 0, "Qwen3 voice-clone ref_code must contain at least one frame");
    OPENVINO_ASSERT(num_code_groups == m_engine.ids().num_code_groups,
                    "Qwen3 voice-clone ref_code group count mismatch. Expected ",
                    m_engine.ids().num_code_groups,
                    ", got ",
                    num_code_groups);

    ov::Tensor ref_code_embed(ov::element::f32, ov::Shape{1, ref_len, hidden});
    std::fill_n(ref_code_embed.data<float>(), ref_code_embed.get_size(), 0.0f);
    for (size_t g = 0; g < num_code_groups; ++g) {
        std::vector<int64_t> group_ids(ref_len);
        const int64_t* codes = prompt.ref_code.data<const int64_t>();
        for (size_t t = 0; t < ref_len; ++t) {
            // Batch dimension is asserted to be 1 above, so the flat index is the same for rank 2 and rank 3.
            group_ids[t] = codes[t * num_code_groups + g];
        }

        ov::Tensor group_embed = (g == 0) ? m_engine.infer_embedding_seq(m_engine.talker_embedding(), group_ids)
                                          : m_engine.infer_predictor_embedding_seq(group_ids, static_cast<int64_t>(g - 1));
        OPENVINO_ASSERT(group_embed.get_shape() == ov::Shape({1, ref_len, hidden}),
                        "Qwen3 voice-clone ref_code group embedding shape mismatch. Expected [1,", ref_len, ",", hidden,
                        "], got ", group_embed.get_shape());
        const float* group_ptr = group_embed.data<const float>();
        float* ref_ptr = ref_code_embed.data<float>();
        for (size_t i = 0; i < ref_len * hidden; ++i) {
            ref_ptr[i] += group_ptr[i];
        }
    }

    // Role embed: first 3 tokens = <|im_start|>assistant\n
    auto role_embed = m_engine.infer_embedding_seq(m_engine.talker_text_embedding(), role_ids);

    // text_embed: embed(ref_content ++ target_content), then append tts_eos
    // (matches Python: text_embed = text_projection(embed(cat([ref_id, text_id]))); cat([text_embed, tts_eos])).
    auto content_embed = m_engine.infer_embedding_seq(m_engine.talker_text_embedding(), icl_content_ids);
    ov::Tensor text_embed_with_eos = concat_embed({content_embed, tts_eos});
    const size_t text_eos_len = text_embed_with_eos.get_shape()[1];

    // Part 1: codec_input_part — blend codec_input_no_bos with [tts_pad × (len-2), tts_bos]
    // Matches Python's: [(tts_pad*(N-2) + tts_bos)] + codec_input_embedding[:-1]
    const size_t ci_len = codec_input_no_bos.get_shape()[1];
    ov::Tensor codec_input_part(ov::element::f32, ov::Shape{1, ci_len, hidden});
    for (size_t i = 0; i < ci_len; ++i) {
        const float* src = codec_input_no_bos.data<const float>() + i * hidden;
        float* dst = codec_input_part.data<float>() + i * hidden;
        const float* blend = (i == ci_len - 1) ? tts_bos.data<const float>() : tts_pad.data<const float>();
        for (size_t h = 0; h < hidden; ++h) {
            dst[h] = src[h] + blend[h];
        }
    }

    // Part 2: codec_side source = (codec_bos + ref_codes)
    ov::Tensor codec_bos_plus_ref = concat_embed({codec_bos_embed, ref_code_embed});
    const size_t codec_side_len = codec_bos_plus_ref.get_shape()[1];
    ov::Tensor icl_input_embed;
    ov::Tensor trailing_text_hidden;
    if (generation_config.non_streaming_mode) {
        // HF ICL non-streaming: (text_with_eos + codec_pad) + (codec_bos_plus_ref + tts_pad)
        auto codec_pad_for_text = m_engine.infer_embedding_seq(m_engine.talker_embedding(),
                                                                std::vector<int64_t>(text_eos_len, m_engine.ids().codec_pad_id));
        ov::Tensor text_side(ov::element::f32, ov::Shape{1, text_eos_len, hidden});
        for (size_t i = 0; i < text_side.get_size(); ++i) {
            text_side.data<float>()[i] = text_embed_with_eos.data<const float>()[i] + codec_pad_for_text.data<const float>()[i];
        }

        ov::Tensor codec_side(ov::element::f32, ov::Shape{1, codec_side_len, hidden});
        for (size_t i = 0; i < codec_side_len; ++i) {
            const float* src = codec_bos_plus_ref.data<const float>() + i * hidden;
            float* dst = codec_side.data<float>() + i * hidden;
            const float* pad = tts_pad.data<const float>();
            for (size_t h = 0; h < hidden; ++h) {
                dst[h] = src[h] + pad[h];
            }
        }

        icl_input_embed = concat_embed({text_side, codec_side});
    } else {
        // HF ICL streaming: align text_with_eos and codec_bos_plus_ref by min(text_len, codec_len),
        // and carry remaining text as trailing_text_hidden (or tts_pad when text is shorter).
        if (text_eos_len > codec_side_len) {
            ov::Tensor overlap(ov::element::f32, ov::Shape{1, codec_side_len, hidden});
            for (size_t i = 0; i < codec_side_len; ++i) {
                const float* text_ptr = text_embed_with_eos.data<const float>() + i * hidden;
                const float* codec_ptr = codec_bos_plus_ref.data<const float>() + i * hidden;
                float* out = overlap.data<float>() + i * hidden;
                for (size_t h = 0; h < hidden; ++h) {
                    out[h] = text_ptr[h] + codec_ptr[h];
                }
            }
            icl_input_embed = overlap;

            trailing_text_hidden = ov::Tensor(text_embed_with_eos,
                                              ov::Coordinate{0, codec_side_len, 0},
                                              ov::Coordinate{1, text_eos_len, hidden});
        } else {
            ov::Tensor text_padded(ov::element::f32, ov::Shape{1, codec_side_len, hidden});
            if (text_eos_len > 0) {
                std::copy_n(text_embed_with_eos.data<const float>(),
                            text_embed_with_eos.get_size(),
                            text_padded.data<float>());
            }
            for (size_t i = text_eos_len; i < codec_side_len; ++i) {
                std::copy_n(tts_pad.data<const float>(), hidden, text_padded.data<float>() + i * hidden);
            }

            ov::Tensor overlap(ov::element::f32, ov::Shape{1, codec_side_len, hidden});
            for (size_t i = 0; i < codec_side_len; ++i) {
                const float* text_ptr = text_padded.data<const float>() + i * hidden;
                const float* codec_ptr = codec_bos_plus_ref.data<const float>() + i * hidden;
                float* out = overlap.data<float>() + i * hidden;
                for (size_t h = 0; h < hidden; ++h) {
                    out[h] = text_ptr[h] + codec_ptr[h];
                }
            }
            icl_input_embed = overlap;
            trailing_text_hidden = tts_pad;
        }
    }

    // Full prefill: role | codec_input_part | icl_input_embed
    ov::Tensor talker_prefill = concat_embed({role_embed, codec_input_part, icl_input_embed});

    talker_prefill = m_engine.prepend_instruct_embed(talker_prefill, generation_config.instruct);

    return decode_from_prefill(talker_prefill,
                               tts_pad,
                               trailing_text_hidden,
                               generation_config,
                               suppress_tokens,
                               prompt.ref_code);
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
