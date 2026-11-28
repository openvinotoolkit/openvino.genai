// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speech_generation/qwen3_tts/qwen3_tts_paged_model.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <set>
#include <unordered_map>

#include "omni/speech_streamer_utils.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_stream_decoder.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_utils.hpp"
#include "utils.hpp"

using namespace ov::genai::qwen3_tts;

namespace ov {
namespace genai {

namespace {

SchedulerConfig make_default_tts_scheduler_config() {
    SchedulerConfig cfg;
    cfg.enable_prefix_caching = true;
    cfg.dynamic_split_fuse = true;
    cfg.max_num_batched_tokens = 256;
    cfg.cache_size = 2;
    return cfg;
}

GenerationConfig make_talker_generation_config(const SpeechGenerationConfig& speech_cfg,
                                               int64_t codec_eos_token_id,
                                               size_t max_steps) {
    GenerationConfig gen_cfg;
    gen_cfg.max_new_tokens = max_steps;
    gen_cfg.do_sample = speech_cfg.do_sample;
    gen_cfg.temperature = speech_cfg.temperature;
    gen_cfg.top_k = speech_cfg.top_k;
    gen_cfg.top_p = speech_cfg.top_p;
    gen_cfg.repetition_penalty = speech_cfg.repetition_penalty;
    gen_cfg.rng_seed = speech_cfg.rng_seed;
    gen_cfg.eos_token_id = codec_eos_token_id;
    gen_cfg.stop_token_ids = {codec_eos_token_id};
    gen_cfg.ignore_eos = false;
    return gen_cfg;
}

}  // namespace

Qwen3TTSPagedImpl::Qwen3TTSPagedImpl(const std::filesystem::path& models_path,
                                     const std::string& device,
                                     const ov::AnyMap& properties,
                                     const Tokenizer& tokenizer)
    : m_engine(models_path, device, properties, tokenizer) {
    auto [talker_device, talker_properties] = resolve_component_target(device, properties, roles::TALKER);
    m_talker_device = talker_device;

    OPENVINO_ASSERT(talker_device.find("NPU") == std::string::npos,
                    "Qwen3TTSPagedImpl (PagedAttention) is supported on CPU/GPU; use SDPA backend for NPU talker.");

    // Extract SchedulerConfig if provided by caller, otherwise use TTS defaults (prefix caching + chunked prefill).
    auto [_, sched_cfg] = utils::extract_scheduler_config(properties, make_default_tts_scheduler_config());

    // Default to U8 KV cache on GPU unless caller explicitly specified kv_cache_precision.
    if (talker_device.find("GPU") != std::string::npos &&
        talker_properties.find(ov::hint::kv_cache_precision.name()) == talker_properties.end()) {
        talker_properties.emplace(ov::hint::kv_cache_precision(ov::element::u8));
    }

    const auto talker_path = models_path / TALKER_NAME;
    auto talker_model = ov::genai::utils::singleton_core().read_model(talker_path);

    const std::vector<bool> suppress_mask = m_engine.compute_suppress_tokens();
    std::set<int64_t> suppressed_ids;
    for (size_t i = 0; i < suppress_mask.size(); ++i) {
        if (suppress_mask[i]) {
            suppressed_ids.insert(static_cast<int64_t>(i));
        }
    }

    GenerationConfig default_gen_cfg = make_talker_generation_config(
        SpeechGenerationConfig{}, m_engine.ids().codec_eos_token_id, /*max_steps=*/2048);

    m_cb_pipeline = std::make_unique<EmbeddingContinuousBatchingPipeline>(
        talker_model,
        m_engine.tokenizer(),
        sched_cfg,
        talker_device,
        talker_properties,
        default_gen_cfg,
        /*post_sample_embed_cb=*/PostSampleEmbedCallback{},
        std::move(suppressed_ids),
        /*enable_hidden_state_export=*/true);

    m_engine.perf_device()[TALKER_PREFILL_PERF_NAME] = talker_device;
    m_engine.perf_device()[TALKER_GENERATE_PERF_NAME] = talker_device;
}

Text2SpeechDecodedResults Qwen3TTSPagedImpl::generate(const std::vector<std::string>& texts,
                                                      const ov::Tensor& speaker_embedding,
                                                      const SpeechGenerationConfig& generation_config) {
    return generate(texts, speaker_embedding, generation_config, std::monostate{});
}

Text2SpeechDecodedResults Qwen3TTSPagedImpl::generate(const std::vector<std::string>& texts,
                                                      const ov::Tensor& speaker_embedding,
                                                      const SpeechGenerationConfig& generation_config,
                                                      const AudioStreamerVariant& streamer) {
    Text2SpeechDecodedResults result;
    result.output_sample_rate = m_engine.output_sample_rate();
    result.perf_metrics.output_sample_rate = result.output_sample_rate;

    if (texts.empty()) {
        result.perf_metrics.evaluate_statistics();
        m_perf_metrics = result.perf_metrics;
        return result;
    }

    const auto generation_start = std::chrono::steady_clock::now();

    const std::string language = m_engine.normalize_text_language(generation_config.language);
    const std::string speaker = m_engine.normalize_speaker(generation_config.speaker);
    const bool base_model = m_engine.is_base_model();

    if (m_engine.tts_model_type() == "voice_design") {
        OPENVINO_ASSERT(generation_config.speaker.empty(),
                        "Qwen3 VoiceDesign does not support 'speaker'. Remove speaker and use 'instruct'.");
        OPENVINO_ASSERT(!speaker_embedding,
                        "Qwen3 VoiceDesign does not accept external speaker_embedding. Use language/instruct only.");
        OPENVINO_ASSERT(!generation_config.instruct.empty(),
                        "Qwen3 VoiceDesign requires a non-empty 'instruct' property.");
    } else if (m_engine.tts_model_type() == "custom_voice") {
        OPENVINO_ASSERT(!speaker_embedding,
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

    const bool x_vector_only_mode =
        base_model && generation_config.ref_text.empty() && !static_cast<bool>(generation_config.ref_codec_ids);

    ov::Tensor effective_speaker_embedding = speaker_embedding;
    if (base_model && !effective_speaker_embedding && generation_config.ref_audio) {
        effective_speaker_embedding = m_engine.extract_qwen3_speaker_embedding_from_audio(generation_config.ref_audio);
    }
    if (base_model && effective_speaker_embedding) {
        result.speaker_embedding = effective_speaker_embedding;
    }

    const bool icl_voice_clone =
        base_model && !x_vector_only_mode &&
        (!generation_config.ref_text.empty() || static_cast<bool>(generation_config.ref_audio) ||
         static_cast<bool>(generation_config.ref_codec_ids));

    ov::Tensor resolved_ref_code;
    if (icl_voice_clone) {
        OPENVINO_ASSERT(!generation_config.ref_text.empty(),
                        "Qwen3 Base ICL voice cloning requires non-empty 'ref_text'.");
        OPENVINO_ASSERT(effective_speaker_embedding,
                        "Qwen3 Base ICL voice cloning requires a speaker source: pass 'speaker_embedding' or 'ref_audio'.");
        resolved_ref_code = generation_config.ref_codec_ids;
        if (!resolved_ref_code && generation_config.ref_audio) {
            resolved_ref_code = m_engine.extract_qwen3_ref_code_from_audio(generation_config.ref_audio);
        }
        OPENVINO_ASSERT(resolved_ref_code,
                        "Qwen3 Base ICL voice cloning requires reference codec ids.");
        result.speaker_embedding = effective_speaker_embedding;
        result.ref_codec_ids = resolved_ref_code;
    }

    const bool streaming = is_speech_streamer_active(streamer);
    OPENVINO_ASSERT(!streaming || texts.size() == 1,
                    "Streaming audio output in Qwen3TTSPagedImpl currently supports a single prompt per generate() call");

    size_t max_steps = generation_config.get_max_new_tokens();
    if (max_steps == SIZE_MAX) {
        max_steps = 2048;
    }
    if (max_steps > 0) {
        --max_steps;
    }

    struct RequestContext {
        uint64_t request_id = 0;
        Qwen3TTSCodecEngine::PrefillResult prefill;
        ov::Tensor ref_codec_ids;
        std::vector<int64_t> all_codes;
        size_t generated_frames = 0;
        std::unique_ptr<Qwen3TTSStreamSession> stream_session;
    };

    std::vector<RequestContext> contexts(texts.size());
    std::unordered_map<uint64_t, size_t> req_id_to_ctx;
    std::vector<GenerationHandle> handles;
    handles.reserve(texts.size());

    std::vector<int64_t> prefix_ref_codes;
    if (streaming && resolved_ref_code) {
        const auto ref_shape = resolved_ref_code.get_shape();
        const bool rank3 = ref_shape.size() == 3;
        const size_t ref_frames = rank3 ? ref_shape[1] : ref_shape[0];
        const size_t ref_groups = rank3 ? ref_shape[2] : ref_shape[1];
        const int64_t* ref_ptr = resolved_ref_code.data<const int64_t>();
        prefix_ref_codes.assign(ref_ptr, ref_ptr + ref_frames * ref_groups);
    }

    int64_t total_prefill_tokens = 0;
    for (size_t idx = 0; idx < texts.size(); ++idx) {
        auto& ctx = contexts[idx];
        ctx.request_id = static_cast<uint64_t>(idx);
        req_id_to_ctx[ctx.request_id] = idx;

        if (icl_voice_clone) {
            auto vc = m_engine.build_voice_clone_prefill(
                texts[idx], generation_config.ref_text, effective_speaker_embedding, resolved_ref_code, generation_config);
            ctx.prefill = std::move(vc.prefill);
            ctx.ref_codec_ids = std::move(vc.ref_codec_ids);
        } else {
            const auto tok_start = std::chrono::steady_clock::now();
            const std::string assistant_text = "<|im_start|>assistant\n" + texts[idx] + "<|im_end|>\n<|im_start|>assistant\n";
            auto input_ids_tensor = m_engine.tokenizer().encode(assistant_text).input_ids;
            const auto tok_end = std::chrono::steady_clock::now();
            result.perf_metrics.raw_metrics.tokenization_durations.emplace_back(
                PerfMetrics::get_microsec(tok_end - tok_start));

            ctx.prefill = m_engine.build_prefill(
                input_ids_tensor, effective_speaker_embedding, generation_config, language, speaker, base_model);
        }

        const size_t prompt_len = ctx.prefill.talker_prefill.get_shape()[1];
        total_prefill_tokens += static_cast<int64_t>(prompt_len);

        if (streaming) {
            ctx.stream_session = std::make_unique<Qwen3TTSStreamSession>(
                m_engine.stream_decoder_pool(),
                [this](const std::vector<int64_t>& window, size_t ctx_frames, size_t new_frames) {
                    return m_engine.decode_speech_tokenizer_chunk(window, ctx_frames, new_frames);
                },
                generation_config,
                streamer,
                generation_start,
                prefix_ref_codes);
        } else {
            ctx.all_codes.reserve(std::min(max_steps, size_t(8192)) * m_engine.ids().num_code_groups);
        }

        auto gen_cfg = make_talker_generation_config(
            generation_config, m_engine.ids().codec_eos_token_id, std::max<size_t>(max_steps, 1));
        if (max_steps == 0) {
            continue;
        }
        auto pos_ids = make_position_ids_prefill(prompt_len);
        handles.push_back(m_cb_pipeline->add_request_with_embeddings(
            ctx.request_id, ctx.prefill.talker_prefill, pos_ids, gen_cfg));
    }

    std::mt19937 rng(generation_config.rng_seed != 0 ? static_cast<uint32_t>(generation_config.rng_seed)
                                                     : std::random_device{}());
    const int64_t codec_eos_id = m_engine.ids().codec_eos_token_id;
    const size_t hidden = m_engine.talker_hidden_size();

    auto codegen_start = std::chrono::steady_clock::now();

    m_cb_pipeline->set_post_sample_embed_callback(
        [&](const std::vector<StepSampledSequence>& batch) -> std::vector<StepNextEmbed> {
            std::vector<StepNextEmbed> outputs(batch.size());

            // Collect indices of sequences that sampled a non-EOS codec token and need subcode prediction.
            std::vector<size_t> active_indices;
            active_indices.reserve(batch.size());
            for (size_t i = 0; i < batch.size(); ++i) {
                const auto& item = batch[i];
                if (item.sampled_token_id == codec_eos_id) {
                    continue;
                }
                auto& ctx = contexts[req_id_to_ctx.at(item.request_id)];
                if (ctx.stream_session && ctx.stream_session->should_stop()) {
                    outputs[i].stop_sequence = true;
                    continue;
                }
                active_indices.push_back(i);
            }

            if (active_indices.empty()) {
                return outputs;
            }

            const size_t B = active_indices.size();
            ov::Tensor past_hidden_batch(ov::element::f32, ov::Shape{B, 1, hidden});
            std::vector<int64_t> first_codes(B);
            std::vector<size_t> steps(B);
            std::vector<ov::Tensor> tts_pads(B);
            std::vector<ov::Tensor> trailings(B);
            std::vector<SpeechGenerationConfig> cfgs(B, generation_config);

            float* ph_dst = past_hidden_batch.data<float>();
            for (size_t b = 0; b < B; ++b) {
                const auto& item = batch[active_indices[b]];
                const auto& ctx = contexts[req_id_to_ctx.at(item.request_id)];
                OPENVINO_ASSERT(item.last_hidden_state && item.last_hidden_state.get_size() >= hidden,
                                "Missing exported last_hidden_state in Qwen3TTSPagedImpl step");
                std::memcpy(ph_dst + b * hidden, item.last_hidden_state.data<const float>(), hidden * sizeof(float));
                first_codes[b] = item.sampled_token_id;
                steps[b] = item.generated_index;
                tts_pads[b] = ctx.prefill.tts_pad;
                trailings[b] = ctx.prefill.trailing_text_hidden;
            }

            auto subcode_batch = m_engine.generate_codec_groups_and_next_embed_batch(
                past_hidden_batch, first_codes, steps, tts_pads, trailings, cfgs, rng);

            for (size_t b = 0; b < B; ++b) {
                const size_t out_idx = active_indices[b];
                const auto& item = batch[out_idx];
                auto& ctx = contexts[req_id_to_ctx.at(item.request_id)];

                const auto& frame_codes = subcode_batch.codes[b];
                ++ctx.generated_frames;
                if (ctx.stream_session) {
                    if (!ctx.stream_session->push_frame(frame_codes.data(), frame_codes.size())) {
                        outputs[out_idx].stop_sequence = true;
                        continue;
                    }
                } else {
                    ctx.all_codes.insert(ctx.all_codes.end(), frame_codes.begin(), frame_codes.end());
                }

                if (!item.is_finished) {
                    ov::Tensor row_embed(ov::element::f32, ov::Shape{1, 1, hidden});
                    std::memcpy(row_embed.data<float>(),
                                subcode_batch.next_embeds.data<const float>() + b * hidden,
                                hidden * sizeof(float));
                    outputs[out_idx].next_embed = std::move(row_embed);
                    outputs[out_idx].next_position_ids =
                        make_position_ids_decode(item.prompt_len + item.generated_index);
                }
            }

            return outputs;
        });

    struct CallbackResetGuard {
        EmbeddingContinuousBatchingPipeline& cb;
        ~CallbackResetGuard() {
            cb.set_post_sample_embed_callback({});
        }
    } cb_guard{*m_cb_pipeline};

    while (m_cb_pipeline->has_non_finished_requests()) {
        m_cb_pipeline->step();
    }

    const auto codegen_end = std::chrono::steady_clock::now();
    result.perf_metrics.raw_codegen_durations.emplace_back(
        PerfMetrics::get_microsec(codegen_end - codegen_start));

    for (auto& ctx : contexts) {
        result.perf_metrics.num_generated_frames += ctx.generated_frames;
        std::vector<float> waveform;
        if (ctx.stream_session) {
            auto stream_res = ctx.stream_session->finish();
            waveform = std::move(stream_res.waveform);
            if (stream_res.has_ttfa) {
                result.perf_metrics.raw_ttfa_durations.emplace_back(static_cast<float>(stream_res.ttfa_us));
            }
            result.perf_metrics.raw_audio_decode_durations.emplace_back(
                static_cast<float>(stream_res.total_decode_us));
        } else {
            const auto decode_start = std::chrono::steady_clock::now();
            waveform = ctx.ref_codec_ids
                ? m_engine.decode_speech_tokenizer_with_reference(ctx.ref_codec_ids, ctx.all_codes)
                : m_engine.decode_speech_tokenizer(ctx.all_codes);
            const auto decode_end = std::chrono::steady_clock::now();
            result.perf_metrics.raw_audio_decode_durations.emplace_back(
                PerfMetrics::get_microsec(decode_end - decode_start));
        }

        ov::Tensor wav_tensor(ov::element::f32, ov::Shape{waveform.size()});
        if (!waveform.empty()) {
            std::copy(waveform.begin(), waveform.end(), wav_tensor.data<float>());
        }
        result.perf_metrics.num_generated_samples += wav_tensor.get_size();
        result.speeches.push_back(std::move(wav_tensor));
    }

    const auto generation_end = std::chrono::steady_clock::now();
    result.perf_metrics.raw_metrics.generate_durations.emplace_back(
        PerfMetrics::get_microsec(generation_end - generation_start));
    result.perf_metrics.evaluate_statistics();
    m_perf_metrics = result.perf_metrics;
    m_engine.perf_print_and_reset(total_prefill_tokens);
    return result;
}

ov::Shape Qwen3TTSPagedImpl::get_speaker_embedding_shape() const {
    if (m_engine.is_base_model()) {
        return ov::Shape{1, 1, m_engine.speaker_embedding_dim()};
    }
    return ov::Shape{};
}

}  // namespace genai
}  // namespace ov
