// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speech_generation/qwen3_tts/qwen3_tts_continuous_batching_impl.hpp"

#include <fstream>
#include <limits>

#include <nlohmann/json.hpp>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset15.hpp>

#include "speech_generation/qwen3_tts/qwen3_tts_utils.hpp"
#include "utils.hpp"

using namespace ov::genai::qwen3_tts;

namespace {

struct TalkerVocabInfo {
    size_t vocab_size = 3072;
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
    info.vocab_size = talker.value("vocab_size", static_cast<size_t>(3072));
    info.eos_token_id = talker.value("codec_eos_token_id", static_cast<int64_t>(-1));
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

    std::vector<float> bias(info.vocab_size, 0.0f);
    const size_t suppress_begin = info.vocab_size > 1024 ? (info.vocab_size - 1024) : 0;
    for (size_t i = suppress_begin; i < info.vocab_size; ++i) {
        if (static_cast<int64_t>(i) != info.eos_token_id) {
            bias[i] = -std::numeric_limits<float>::infinity();
        }
    }
    auto bias_const = ov::opset15::Constant::create(ov::element::f32, ov::Shape{1, 1, info.vocab_size}, bias);
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
                             properties,
                             GenerationConfig()),
      m_engine(models_path, device, properties, tokenizer) {
    OPENVINO_ASSERT(device.find("NPU") == std::string::npos,
                    "Text2SpeechContinuousBatchingPipeline does not support NPU; use Text2SpeechPipeline "
                    "with the default (SDPA) backend on NPU instead.");
    m_model_input_type = ModelInputType::EMBEDDINGS;
    m_model_runner->enable_hidden_state_export(true);
}

SpeechGenerationHandle ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::add_speech_request(
    uint64_t request_id,
    const std::string& text,
    const ov::Tensor& speaker_embedding,
    const SpeechGenerationConfig& generation_config) {
    OPENVINO_ASSERT(generation_config.num_return_sequences == 1,
                    "Qwen3-TTS continuous batching does not support beam search / parallel sampling "
                    "(num_return_sequences must be 1)");
    OPENVINO_ASSERT(generation_config.ref_text.empty() && !generation_config.ref_audio && !generation_config.ref_codec_ids,
                    "ref_text/ref_audio/ref_codec_ids voice cloning is not yet supported by "
                    "Text2SpeechContinuousBatchingPipeline; use Text2SpeechPipeline instead");

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
        OPENVINO_ASSERT(!speaker_embedding,
                        "Qwen3 CustomVoice does not accept external speaker_embedding. Use 'speaker' property instead.");
        OPENVINO_ASSERT(!speaker.empty(), "Qwen3 CustomVoice requires a non-empty 'speaker' property.");
    } else if (base_model) {
        OPENVINO_ASSERT(speaker_embedding, "Qwen3 Base requires a 'speaker_embedding' input.");
    }

    const std::string assistant_text = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
    auto input_ids_tensor = m_engine.tokenizer().encode(assistant_text).input_ids;

    auto prefill = m_engine.build_prefill(input_ids_tensor, speaker_embedding, generation_config, language, speaker, base_model);
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

    GenerationConfig sampling_params = generation_config;
    sampling_params.set_eos_token_id(static_cast<size_t>(m_engine.ids().codec_eos_token_id));
    sampling_params.max_new_tokens = max_new_tokens;
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
    state.rng.seed(generation_config.rng_seed != 0 ? static_cast<uint32_t>(generation_config.rng_seed)
                                                   : std::random_device{}());
    state.tts_pad = prefill.tts_pad;
    state.trailing_text_hidden = prefill.trailing_text_hidden;
    state.stream = SpeechGenerationStream::create();
    SpeechGenerationStream::Ptr stream = state.stream;

    {
        std::lock_guard<std::mutex> state_lock{m_request_state_mutex};
        m_request_state.emplace(request_id, std::move(state));
    }
    {
        std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
        m_awaiting_requests.push_back(sequence_group);
    }

    return std::make_shared<SpeechGenerationHandleImpl>(stream);
}

void ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::generate_candidates_for_prompt_lookup() {
    std::lock_guard<std::mutex> lock{m_request_state_mutex};
    for (auto& sequence_group : m_requests) {
        auto state_it = m_request_state.find(sequence_group->get_request_id());
        if (state_it == m_request_state.end()) {
            continue;
        }

        for (auto& sequence : sequence_group->get_running_sequences()) {
            if (sequence->get_generated_len() <= sequence->get_generated_ids_embeds().size()) {
                continue;  // no new token sampled this step (mid multi-step prefill, or already synced)
            }
            process_new_frame(sequence_group, sequence, state_it->second);
        }

        if (sequence_group->has_finished()) {
            finalize_request(sequence_group, state_it->second);
            m_request_state.erase(state_it);
        }
    }
}

void ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::process_new_frame(
    const SequenceGroup::Ptr& sequence_group,
    const Sequence::Ptr& sequence,
    RequestState& state) {
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

    ov::Tensor past_hidden = sequence->get_hidden_state();
    auto groups = m_engine.generate_codec_groups(past_hidden, token, state.generation_config, state.rng);
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
    auto waveform = m_engine.decode_speech_tokenizer(state.all_codes);

    ov::Tensor wav_tensor(ov::element::f32, ov::Shape{waveform.size()});
    if (!waveform.empty()) {
        std::copy(waveform.begin(), waveform.end(), wav_tensor.data<float>());
    }

    SpeechGenerationOutput output;
    output.waveform = wav_tensor;
    output.finish_reason = sequence_group->get_sequences().front()->get_finish_reason();
    state.stream->push(std::move(output));

    SpeechGenerationPerfMetrics perf_metrics;
    perf_metrics.num_generated_samples = wav_tensor.get_size();
    perf_metrics.evaluate_statistics();
    state.stream->set_perf_metrics(std::move(perf_metrics));
    state.stream->set_generation_status(GenerationStatus::FINISHED);
}

ov::Shape ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl::get_speaker_embedding_shape() const {
    if (m_engine.is_base_model()) {
        return ov::Shape{1, 1, m_engine.speaker_embedding_dim()};
    }
    return ov::Shape{};
}

}  // namespace genai
}  // namespace ov
