// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speech_generation/qwen3_tts/qwen3_tts_continuous_batching_impl.hpp"

#include <fstream>
#include <iostream>
#include <limits>

#include <nlohmann/json.hpp>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset15.hpp>

#include "speech_generation/qwen3_tts/qwen3_tts_utils.hpp"
#include "utils.hpp"

using namespace ov::genai::qwen3_tts;

namespace {

struct TalkerVocabInfo {
    uint64_t vocab_size = 3072;
    int64_t eos_token_id = -1;
};

// Minimal duplicate of the two fields Qwen3TTSCodecEngine::init_config also reads from
// talker_config; needed here because the talker model must be prepared (and its vocab size known)
// BEFORE m_engine (a class member, constructed after the base ContinuousBatchingImpl) can exist.
TalkerVocabInfo read_talker_vocab_info(const std::filesystem::path& models_path) {
    const auto config_path = models_path / "config.json";
    std::ifstream config_stream(config_path);
    OPENVINO_ASSERT(config_stream.is_open(), "Failed to open ", config_path);
    nlohmann::json config = nlohmann::json::parse(config_stream);
    const nlohmann::json talker = config.at("talker_config");

    TalkerVocabInfo info;
    const auto vocab_it = talker.find("vocab_size");
    if (vocab_it != talker.end()) {
        OPENVINO_ASSERT(vocab_it->is_number_unsigned() || vocab_it->is_number_integer(),
                        "Qwen3-TTS talker_config.vocab_size must be a positive integer");
        if (vocab_it->is_number_unsigned()) {
            info.vocab_size = vocab_it->get<uint64_t>();
        } else {
            const int64_t vocab_size = vocab_it->get<int64_t>();
            OPENVINO_ASSERT(vocab_size > 0,
                            "Qwen3-TTS talker_config.vocab_size must be positive, got ",
                            vocab_size);
            info.vocab_size = static_cast<uint64_t>(vocab_size);
        }
    }
    info.eos_token_id = talker.value("codec_eos_token_id", static_cast<int64_t>(-1));
    OPENVINO_ASSERT(info.vocab_size > 0 && info.vocab_size <= std::numeric_limits<size_t>::max(),
                    "Qwen3-TTS talker_config.vocab_size is out of range: ",
                    info.vocab_size);
    return info;
}

// Bakes the talker's reserved-vocab suppression mask (blocks the top-1024 ids except EOS, see
// Qwen3TTSCodecEngine::compute_suppress_tokens()) directly into the model graph, since the stock
// ContinuousBatching Sampler has no per-vocab-id suppression mechanism (only
// top_k/top_p/temperature/repetition_penalty via GenerationConfig).
void apply_talker_logit_suppression(const std::shared_ptr<ov::Model>& model, const TalkerVocabInfo& info) {
    std::shared_ptr<ov::Node> logits_result;
    for (const auto& result : model->get_results()) {
        if (result->get_output_tensor(0).get_names().count("logits")) {
            logits_result = result;
            break;
        }
    }
    OPENVINO_ASSERT(logits_result, "Qwen3-TTS talker model is expected to have a 'logits' output");

    const ov::PartialShape logits_shape = logits_result->input_value(0).get_partial_shape();
    OPENVINO_ASSERT(logits_shape.rank().is_static() && logits_shape.rank().get_length() == 3 &&
                        logits_shape[2].is_static(),
                    "Qwen3-TTS talker logits must have a statically known vocabulary dimension, got ",
                    logits_shape);
    const size_t logits_vocab_size = static_cast<size_t>(logits_shape[2].get_length());
    OPENVINO_ASSERT(info.vocab_size == logits_vocab_size,
                    "Qwen3-TTS config vocab_size (",
                    info.vocab_size,
                    ") does not match talker logits vocabulary (",
                    logits_vocab_size,
                    ")");
    OPENVINO_ASSERT(info.eos_token_id >= 0 && static_cast<uint64_t>(info.eos_token_id) < info.vocab_size,
                    "Qwen3-TTS codec EOS token id must be in [0, ",
                    info.vocab_size,
                    "), got ",
                    info.eos_token_id);

    const size_t vocab_size = static_cast<size_t>(info.vocab_size);
    std::vector<float> bias(vocab_size, 0.0f);
    const size_t suppress_begin = vocab_size > 1024 ? (vocab_size - 1024) : 0;
    for (size_t i = suppress_begin; i < vocab_size; ++i) {
        if (static_cast<int64_t>(i) != info.eos_token_id) {
            bias[i] = -std::numeric_limits<float>::infinity();
        }
    }
    auto bias_const = ov::opset15::Constant::create(ov::element::f32, ov::Shape{1, 1, vocab_size}, bias);
    auto biased_logits = std::make_shared<ov::opset15::Add>(logits_result->input_value(0), bias_const);
    biased_logits->set_friendly_name(logits_result->get_input_node_shared_ptr(0)->get_friendly_name() + "/qwen3_tts_suppress");
    logits_result->input(0).replace_source_output(biased_logits->output(0));
    model->validate_nodes_and_infer_types();
}

std::shared_ptr<ov::Model> prepare_talker_model(const std::filesystem::path& models_path) {
    auto model = ov::genai::utils::singleton_core().read_model(models_path / TALKER_NAME);
    apply_talker_logit_suppression(model, read_talker_vocab_info(models_path));
    return model;
}

}  // namespace

namespace ov {
namespace genai {

ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::Qwen3TTSContinuousBatchingImpl(
    const std::filesystem::path& models_path,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& properties,
    const Tokenizer& tokenizer)
    : ContinuousBatchingImpl(prepare_talker_model(models_path),
                             tokenizer,
                             scheduler_config,
                             device,
                             utils::get_model_properties(properties, roles::TALKER, device),
                             GenerationConfig()),
      m_engine(models_path, device, properties, tokenizer) {
    OPENVINO_ASSERT(device.find("NPU") == std::string::npos,
                    "Text2SpeechContinuousBatchingPipeline does not support NPU; use Text2SpeechPipeline "
                    "with the default (SDPA) backend on NPU instead.");
    m_model_input_type = ModelInputType::EMBEDDINGS;
    m_model_runner->enable_hidden_state_export(true);
}

void ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::prepare_shared_voice_inputs(
    ov::Tensor& speaker_embedding,
    SpeechGenerationConfig& generation_config) {
    if (!m_engine.is_base_model() || !generation_config.ref_audio) {
        return;
    }

    std::lock_guard<std::mutex> lock{m_request_state_mutex};
    const ov::Tensor ref_audio = generation_config.ref_audio;
    if (!speaker_embedding) {
        speaker_embedding = m_engine.extract_qwen3_speaker_embedding_from_audio(ref_audio);
    }
    if (!generation_config.ref_text.empty() && !generation_config.ref_codec_ids) {
        generation_config.ref_codec_ids = m_engine.extract_qwen3_ref_code_from_audio(ref_audio);
    }
    generation_config.ref_audio = ov::Tensor();
}

SpeechGenerationHandle ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::add_speech_request(
    uint64_t request_id,
    const std::string& text,
    const ov::Tensor& speaker_embedding,
    const SpeechGenerationConfig& generation_config) {
    const auto request_start_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(generation_config.num_return_sequences == 1,
                    "Qwen3-TTS continuous batching does not support beam search / parallel sampling "
                    "(num_return_sequences must be 1)");
    OPENVINO_ASSERT(generation_config.num_beams == 1,
                    "Qwen3-TTS continuous batching does not support beam search (num_beams must be 1)");

    const std::string language = m_engine.normalize_text_language(generation_config.language);
    const std::string speaker = m_engine.normalize_speaker(generation_config.speaker);
    const bool base_model = m_engine.is_base_model();
    const bool has_voice_clone_props = !generation_config.ref_text.empty() ||
                                       static_cast<bool>(generation_config.ref_codec_ids);

    if (!base_model &&
        (!generation_config.ref_text.empty() || static_cast<bool>(generation_config.ref_audio) ||
         static_cast<bool>(generation_config.ref_codec_ids))) {
        OPENVINO_THROW("ref_text/ref_audio/ref_codec_ids are supported only for Qwen3 Base models");
    }

    if (m_engine.tts_model_type() == "voice_design") {
        OPENVINO_ASSERT(generation_config.speaker.empty(),
                        "Qwen3 VoiceDesign does not support 'speaker'. Remove speaker and use 'instruct'.");
        OPENVINO_ASSERT(!speaker_embedding,
                        "Qwen3 VoiceDesign does not accept external speaker_embedding. Use language/instruct only.");
        OPENVINO_ASSERT(!generation_config.instruct.empty(), "Qwen3 VoiceDesign requires a non-empty 'instruct' property.");
    } else if (m_engine.tts_model_type() == "custom_voice") {
        OPENVINO_ASSERT(!speaker_embedding,
                        "Qwen3 CustomVoice does not accept external speaker_embedding. Use 'speaker' property instead.");
        OPENVINO_ASSERT(!speaker.empty(), "Qwen3 CustomVoice requires a non-empty 'speaker' property.");
    } else if (base_model) {
        OPENVINO_ASSERT(speaker_embedding || generation_config.ref_audio,
                        "Qwen3 Base requires a voice source: provide either 'speaker_embedding' or 'ref_audio'.");
    }

    std::unique_lock<std::mutex> state_lock{m_request_state_mutex};
    ov::Tensor effective_speaker_embedding = speaker_embedding;
    if (base_model && !effective_speaker_embedding && generation_config.ref_audio) {
        effective_speaker_embedding = m_engine.extract_qwen3_speaker_embedding_from_audio(generation_config.ref_audio);
    }

    Qwen3TTSCodecEngine::PrefillResult prefill;
    ov::Tensor effective_ref_codec_ids;
    if (has_voice_clone_props) {
        OPENVINO_ASSERT(!generation_config.ref_text.empty(),
                        "Qwen3 Base ICL voice cloning requires non-empty 'ref_text'.");
        OPENVINO_ASSERT(effective_speaker_embedding,
                        "Qwen3 Base ICL voice cloning requires a speaker source: pass 'speaker_embedding' "
                        "or provide 'ref_audio'.");

        effective_ref_codec_ids = generation_config.ref_codec_ids;
        if (!effective_ref_codec_ids && generation_config.ref_audio) {
            effective_ref_codec_ids = m_engine.extract_qwen3_ref_code_from_audio(generation_config.ref_audio);
        }
        OPENVINO_ASSERT(effective_ref_codec_ids,
                        "Qwen3 Base ICL voice cloning requires reference codec ids: pass 'ref_codec_ids' "
                        "or provide 'ref_audio' so ref codes can be extracted.");

        auto voice_clone_prefill = m_engine.build_voice_clone_prefill(text,
                                                                      generation_config.ref_text,
                                                                      effective_speaker_embedding,
                                                                      effective_ref_codec_ids,
                                                                      generation_config);
        prefill = std::move(voice_clone_prefill.prefill);
        effective_ref_codec_ids = std::move(voice_clone_prefill.ref_codec_ids);
    } else {
        const std::string assistant_text = "<|im_start|>assistant\n" + text +
                                           "<|im_end|>\n<|im_start|>assistant\n";
        const ov::Tensor input_ids_tensor = m_engine.tokenizer().encode(assistant_text).input_ids;
        prefill = m_engine.build_prefill(input_ids_tensor,
                                         effective_speaker_embedding,
                                         generation_config,
                                         language,
                                         speaker,
                                         base_model);
    }
    const size_t prompt_len = prefill.talker_prefill.get_shape()[1];

    // Mirror decode_from_prefill()'s max_steps adjustment: the reference implementation computes a
    // token's residual codes while forwarding the NEXT token, so a length-capped generation never
    // gets to compute (or render) its last sampled token's codes. Reducing max_new_tokens by one
    // makes the stock Sampler's own length-based stop land on the same effective frame count.
    size_t max_new_tokens = generation_config.get_max_new_tokens();
    if (max_new_tokens == SIZE_MAX) {
        max_new_tokens = 2048;
    }
    if (max_new_tokens > 0) {
        --max_new_tokens;
    }

    if (max_new_tokens == 0) {
        SpeechGenerationStream::Ptr stream = SpeechGenerationStream::create();
        SpeechGenerationOutput output;
        output.waveform = ov::Tensor(ov::element::f32, ov::Shape{0});
        output.output_sample_rate = m_engine.output_sample_rate();
        output.finish_reason = GenerationFinishReason::LENGTH;
        output.speaker_embedding = effective_speaker_embedding;
        output.ref_codec_ids = effective_ref_codec_ids;

        SpeechGenerationPerfMetrics perf_metrics;
        perf_metrics.raw_metrics.generate_durations.emplace_back(
            PerfMetrics::get_microsec(std::chrono::steady_clock::now() - request_start_time));
        perf_metrics.evaluate_statistics();
        stream->set_perf_metrics(std::move(perf_metrics));
        stream->push_and_close(std::move(output), GenerationStatus::FINISHED);
        state_lock.unlock();
        return std::make_shared<SpeechGenerationHandleImpl>(stream);
    }

    GenerationConfig sampling_params = generation_config;
    sampling_params.set_eos_token_id(static_cast<size_t>(m_engine.ids().codec_eos_token_id));
    sampling_params.max_new_tokens = max_new_tokens;
    sampling_params.min_new_tokens = std::min(sampling_params.min_new_tokens, max_new_tokens);
    sampling_params.max_length = SIZE_MAX;
    sampling_params.validate();

    const ov::Tensor position_ids = make_position_ids_prefill(prompt_len);
    auto sequence_group = std::make_shared<SequenceGroup>(request_id,
                                                          prefill.talker_prefill,
                                                          sampling_params,
                                                          std::nullopt,
                                                          position_ids,
                                                          std::nullopt,
                                                          std::nullopt);

    RequestState state;
    state.generation_config = generation_config;
    state.start_time = request_start_time;
    state.generation_config.ref_audio = ov::Tensor();
    state.generation_config.ref_codec_ids = ov::Tensor();
    state.generation_config.ref_text.clear();
    state.speaker_embedding = effective_speaker_embedding;
    state.rng.seed(generation_config.rng_seed != 0 ? static_cast<uint32_t>(generation_config.rng_seed)
                                                   : std::random_device{}());
    state.tts_pad = prefill.tts_pad;
    state.trailing_text_hidden = prefill.trailing_text_hidden;
    state.ref_codec_ids = effective_ref_codec_ids;
    state.stream = SpeechGenerationStream::create();
    SpeechGenerationStream::Ptr stream = state.stream;

    m_request_state.emplace(request_id, std::move(state));
    const auto group_stream = sequence_group->get_generation_stream();
    stream->set_terminal_callback([group_stream](GenerationStatus status, GenerationFinishReason finish_reason) {
        if (status == GenerationStatus::STOP) {
            group_stream->stop(finish_reason);
        } else if (status == GenerationStatus::CANCEL) {
            group_stream->cancel();
        }
    });
    state_lock.unlock();
    try {
        std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
        _throw_if_failed();
        initialize_prefix_cache(sequence_group);
        sequence_group->set_num_prefix_cache_hit_tokens(sequence_group->get_num_processed_tokens());
        m_awaiting_requests.push_back(sequence_group);
    } catch (...) {
        on_request_failed(sequence_group, std::current_exception());
        throw;
    }

    return std::make_shared<SpeechGenerationHandleImpl>(stream);
}

void ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::generate_candidates_for_prompt_lookup() {
    std::lock_guard<std::mutex> lock{m_request_state_mutex};
    struct PendingFrame {
        SequenceGroup::Ptr sequence_group;
        Sequence::Ptr sequence;
        RequestState* state;
        ov::Tensor last_hidden;
        int64_t first_codec_token;
    };

    size_t subtalker_frames_this_forward = 0;
    if (qwen_perf_enabled() && m_pipeline_metrics.scheduled_requests > 0) {
        constexpr const char* talker_cb_forward = "talker_cb_forward";
        m_engine.perf_ms()[talker_cb_forward] += m_pipeline_metrics.inference_duration / 1000.0;
        ++m_engine.perf_calls()[talker_cb_forward];
        m_engine.perf_device()[talker_cb_forward] = m_device;
        m_talker_scheduled_request_groups += m_pipeline_metrics.scheduled_requests;
    }

    std::vector<PendingFrame> pending_frames;
    std::vector<SequenceGroup::Ptr> completed_groups;
    for (auto& sequence_group : m_requests) {
        auto state_it = m_request_state.find(sequence_group->get_request_id());
        if (state_it == m_request_state.end()) {
            continue;
        }
        // Process newly sampled tokens for every sequence, not only RUNNING ones.
        // A sequence can become FINISHED in the same step that sampled its last
        // token due to max_new_tokens; that token still needs code-group
        // expansion and embedding injection to preserve SDPA-equivalent output
        // length/content.
        for (auto& sequence : sequence_group->get_sequences()) {
            if (sequence->get_generated_len() <= sequence->get_generated_ids_embeds().size()) {
                continue;  // no new token sampled this step (mid multi-step prefill, or already synced)
            }
            const int64_t token = sequence->get_generated_ids().back();
            if (token == m_engine.ids().codec_eos_token_id) {
                process_new_frame(sequence_group, sequence, state_it->second);
            } else {
                if (qwen_perf_enabled()) {
                    ++subtalker_frames_this_forward;
                }
                pending_frames.push_back({sequence_group,
                                          sequence,
                                          &state_it->second,
                                          get_last_hidden_state(sequence),
                                          token});
            }
        }

        if (sequence_group->has_finished()) {
            completed_groups.push_back(sequence_group);
        }
    }

    std::vector<Qwen3TTSCodecEngine::CodecGroupBatchItem> batch_items;
    batch_items.reserve(pending_frames.size());
    for (const PendingFrame& frame : pending_frames) {
        batch_items.push_back({frame.last_hidden,
                               frame.first_codec_token,
                               &frame.state->generation_config,
                               &frame.state->rng});
    }
    if (!batch_items.empty()) {
        const auto generated_groups = m_engine.generate_codec_groups_batch(batch_items);
        OPENVINO_ASSERT(generated_groups.size() == pending_frames.size(),
                        "Code-group batch output count mismatch. Expected ",
                        pending_frames.size(),
                        ", got ",
                        generated_groups.size());
        for (size_t frame_idx = 0; frame_idx < pending_frames.size(); ++frame_idx) {
            PendingFrame& frame = pending_frames[frame_idx];
            process_new_frame(frame.sequence_group,
                              frame.sequence,
                              *frame.state,
                              &generated_groups[frame_idx]);
        }
    }

    for (const SequenceGroup::Ptr& sequence_group : completed_groups) {
        const auto state_it = m_request_state.find(sequence_group->get_request_id());
        if (state_it != m_request_state.end()) {
            finalize_request(sequence_group, state_it->second);
            m_request_state.erase(state_it);
        }
    }

    if (qwen_perf_enabled() && m_pipeline_metrics.scheduled_requests > 0) {
        ++m_subtalker_frames_per_step[subtalker_frames_this_forward];
    }
}

void ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::on_request_terminal(
    const SequenceGroup::Ptr& sequence_group,
    GenerationStatus status) {
    std::lock_guard<std::mutex> lock{m_request_state_mutex};
    auto state_it = m_request_state.find(sequence_group->get_request_id());
    if (state_it == m_request_state.end()) {
        return;
    }

    RequestState& state = state_it->second;
    if (status == GenerationStatus::IGNORED) {
        SpeechGenerationPerfMetrics perf_metrics;
        perf_metrics.raw_metrics.generate_durations.emplace_back(
            PerfMetrics::get_microsec(std::chrono::steady_clock::now() - state.start_time));
        perf_metrics.evaluate_statistics();
        state.stream->set_perf_metrics(std::move(perf_metrics));
        state.stream->close(GenerationStatus::IGNORED);
    } else if (status == GenerationStatus::STOP) {
        state.stream->stop(sequence_group->get_generation_stream()->get_finish_reason());
    } else if (status == GenerationStatus::CANCEL) {
        state.stream->cancel();
    } else if (status == GenerationStatus::FINISHED) {
        finalize_request(sequence_group, state);
    }
    m_request_state.erase(state_it);
}

void ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::on_request_failed(
    const SequenceGroup::Ptr& sequence_group,
    std::exception_ptr error) noexcept {
    std::lock_guard<std::mutex> lock{m_request_state_mutex};
    auto state_it = m_request_state.find(sequence_group->get_request_id());
    if (state_it == m_request_state.end()) {
        return;
    }
    state_it->second.stream->fail(std::move(error));
    m_request_state.erase(state_it);
}

ov::Tensor ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::get_last_hidden_state(
    const Sequence::Ptr& sequence) const {
    const size_t hidden = m_engine.talker_hidden_size();
    const ov::Tensor sequence_hidden = sequence->get_hidden_state();
    const ov::Shape hidden_shape = sequence_hidden.get_shape();
    OPENVINO_ASSERT(hidden_shape.size() == 3,
                    "Unexpected talker hidden-state rank for Qwen3-TTS CB: ",
                    hidden_shape.size());
    OPENVINO_ASSERT(hidden_shape[2] == hidden,
                    "Unexpected talker hidden-state width for Qwen3-TTS CB. Expected ",
                    hidden,
                    ", got ",
                    hidden_shape[2]);

    ov::Tensor last_hidden(ov::element::f32, ov::Shape{1, 1, hidden});
    if (hidden_shape[0] == 1) {
        OPENVINO_ASSERT(hidden_shape[1] > 0,
                        "Talker hidden-state sequence length must be > 0");
        const size_t last_idx = hidden_shape[1] - 1;
        const float* src = sequence_hidden.data<const float>() + last_idx * hidden;
        std::copy_n(src, hidden, last_hidden.data<float>());
    } else {
        OPENVINO_ASSERT(hidden_shape[1] == 1 && hidden_shape[0] > 0,
                        "Unexpected talker hidden-state shape for Qwen3-TTS CB: ",
                        hidden_shape,
                        ". Expected [1,T,H] or [T,1,H]");
        const size_t last_idx = hidden_shape[0] - 1;
        const float* src = sequence_hidden.data<const float>() + last_idx * hidden;
        std::copy_n(src, hidden, last_hidden.data<float>());
    }
    return last_hidden;
}

void ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::process_new_frame(
    const SequenceGroup::Ptr& sequence_group,
    const Sequence::Ptr& sequence,
    RequestState& state,
    const std::vector<int64_t>* generated_groups) {
    const int64_t token = sequence->get_generated_ids().back();
    const size_t hidden = m_engine.talker_hidden_size();
    const size_t position_id = sequence->get_generated_ids_embeds().size() + sequence_group->get_prompt_len();
    const bool is_eos = (token == m_engine.ids().codec_eos_token_id);

    if (is_eos) {
        // No frame to render for the eos token itself (matches decode_from_prefill(), which breaks
        // out of its loop before calling generate_codec_groups() for it). Still append a placeholder
        // embedding/position purely to keep m_generated_ids/m_generated_ids_embeds in lock-step; this
        // sequence will not be scheduled again.
        ov::Tensor placeholder(ov::element::f32, ov::Shape{1, 1, hidden});
        std::fill_n(placeholder.data<float>(), hidden, 0.0f);
        sequence->append_generated_ids_embeds(placeholder);
        sequence->append_position_ids(make_position_ids_decode(position_id));
        return;
    }

    if (qwen_perf_enabled()) {
        ++m_subtalker_codec_frames;
    }

    std::vector<int64_t> serial_groups;
    if (!generated_groups) {
        // Match SDPA semantics: the predictor uses only the newest talker hidden row.
        const ov::Tensor last_hidden = get_last_hidden_state(sequence);
        serial_groups = m_engine.generate_codec_groups(last_hidden, token, state.generation_config, state.rng);
        generated_groups = &serial_groups;
    }

    const std::vector<int64_t>& groups = *generated_groups;
    state.all_codes.insert(state.all_codes.end(), groups.begin(), groups.end());

    auto token_embed = m_engine.infer_embedding(m_engine.talker_embedding(), groups[0]);
    float* token_embed_ptr = token_embed.data<float>();
    for (size_t g = 1; g < groups.size(); ++g) {
        auto pred_emb = m_engine.infer_predictor_embedding(groups[g], static_cast<int64_t>(g - 1));
        const float* pred_ptr = pred_emb.data<const float>();
        for (size_t h = 0; h < hidden; ++h) {
            token_embed_ptr[h] += pred_ptr[h];
        }
    }

    const size_t step_index = position_id - sequence_group->get_prompt_len();
    if (state.generation_config.non_streaming_mode) {
        const float* tts_pad_ptr = state.tts_pad.data<const float>();
        for (size_t h = 0; h < hidden; ++h) {
            token_embed_ptr[h] += tts_pad_ptr[h];
        }
    } else if (state.trailing_text_hidden) {
        const size_t trailing_len = state.trailing_text_hidden.get_shape()[1];
        const float* add_ptr = (step_index < trailing_len)
            ? state.trailing_text_hidden.data<const float>() + step_index * hidden
            : state.tts_pad.data<const float>();
        for (size_t h = 0; h < hidden; ++h) {
            token_embed_ptr[h] += add_ptr[h];
        }
    }

    sequence->append_generated_ids_embeds(token_embed);
    sequence->append_position_ids(make_position_ids_decode(position_id));
}

void ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::finalize_request(
    const SequenceGroup::Ptr& sequence_group,
    RequestState& state) {
    auto waveform = state.ref_codec_ids
        ? m_engine.decode_speech_tokenizer_with_reference(state.ref_codec_ids, state.all_codes)
        : m_engine.decode_speech_tokenizer(state.all_codes);

    ov::Tensor wav_tensor(ov::element::f32, ov::Shape{waveform.size()});
    if (!waveform.empty()) {
        std::copy(waveform.begin(), waveform.end(), wav_tensor.data<float>());
    }

    SpeechGenerationOutput output;
    output.waveform = wav_tensor;
    output.output_sample_rate = m_engine.output_sample_rate();
    output.finish_reason = sequence_group->get_sequences().front()->get_finish_reason();
    output.speaker_embedding = state.speaker_embedding;
    output.ref_codec_ids = state.ref_codec_ids;

    SpeechGenerationPerfMetrics perf_metrics;
    perf_metrics.num_generated_samples = wav_tensor.get_size();
    perf_metrics.num_prefix_cache_hit_tokens = sequence_group->get_num_prefix_cache_hit_tokens();
    perf_metrics.raw_metrics.generate_durations.emplace_back(
        PerfMetrics::get_microsec(std::chrono::steady_clock::now() - state.start_time));
    perf_metrics.evaluate_statistics();
    state.stream->set_perf_metrics(std::move(perf_metrics));
    state.stream->push_and_close(std::move(output), GenerationStatus::FINISHED);
}

void ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::report_perf_if_drained() {
    if (!qwen_perf_enabled()) {
        return;
    }

    std::lock_guard<std::mutex> lock{m_request_state_mutex};
    if (!m_request_state.empty() || m_engine.perf_ms().empty()) {
        return;
    }

    const auto talker_calls = m_engine.perf_calls().find("talker_cb_forward");
    const size_t num_talker_batches = talker_calls == m_engine.perf_calls().end()
        ? 0
        : static_cast<size_t>(talker_calls->second);
    std::cout << "[QWEN3_TTS_PERF] Continuous-batching aggregate: "
              << num_talker_batches << " talker batches, "
              << m_talker_scheduled_request_groups << " scheduled request-groups, "
              << m_subtalker_codec_frames << " subtalker codec frames" << std::endl;
    std::cout << "[QWEN3_TTS_PERF] Subtalker frame opportunities per talker forward (frames:forwards): ";
    if (m_subtalker_frames_per_step.empty()) {
        std::cout << "none";
    } else {
        bool first = true;
        for (const auto& [frames, forwards] : m_subtalker_frames_per_step) {
            if (!first) {
                std::cout << ", ";
            }
            std::cout << frames << ":" << forwards;
            first = false;
        }
    }
    std::cout << std::endl;
    m_engine.perf_print_and_reset(0);
    m_talker_scheduled_request_groups = 0;
    m_subtalker_codec_frames = 0;
    m_subtalker_frames_per_step.clear();
}

ov::Shape ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::get_speaker_embedding_shape() const {
    if (m_engine.is_base_model()) {
        return ov::Shape{1, 1, m_engine.speaker_embedding_dim()};
    }
    return ov::Shape{};
}

uint32_t ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::get_output_sample_rate() const {
    return m_engine.output_sample_rate();
}

}  // namespace genai
}  // namespace ov
