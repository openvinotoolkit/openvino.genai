// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_lfm3_audio.hpp"
#ifdef ENABLE_GGUF
#    include <algorithm>
#    include <chrono>
#    include <cmath>
#    include <numeric>

#    include "gguf_frontend_extensions.hpp"
#    include "omni/talker_speech_config_utils.hpp"
#    include "openvino/genai/gguf.hpp"
#    include "openvino/genai/omni/pipeline.hpp"
#    include "openvino/opsets/opset15.hpp"
#    include "utils.hpp"
#    include "visual_language/lfm3/classes.hpp"
#    include "visual_language/vision_registry.hpp"
#    include "visual_language/vlm_chat_context.hpp"

namespace ov::genai {
namespace {
ov::Tensor owned(const ov::Tensor& value) {
    ov::Tensor result(value.get_element_type(), value.get_shape());
    value.copy_to(result);
    return result;
}
int64_t sample(std::vector<float> logits,
               bool greedy,
               float temperature,
               size_t top_k,
               float top_p,
               std::mt19937& rng) {
    OPENVINO_ASSERT(std::any_of(logits.begin(),
                                logits.end(),
                                [](float value) {
                                    return std::isfinite(value);
                                }),
                    "LFM3 sampling has no finite logits");
    if (greedy || temperature <= 0.f || top_k == 1)
        return std::max_element(logits.begin(), logits.end()) - logits.begin();
    std::vector<size_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return logits[a] > logits[b];
    });
    if (top_k && top_k < indices.size())
        indices.resize(top_k);
    std::vector<double> probabilities;
    double sum = 0.;
    for (auto id : indices) {
        probabilities.push_back(std::exp(double(logits[id] - logits[indices.front()]) / temperature));
        sum += probabilities.back();
    }
    double cumulative = 0.;
    size_t keep = 0;
    do {
        cumulative += probabilities[keep++] / sum;
    } while (keep < probabilities.size() && cumulative < top_p);
    probabilities.resize(keep);
    return indices[std::discrete_distribution<size_t>(probabilities.begin(), probabilities.end())(rng)];
}
}  // namespace

GGUFLFM3Speech::GGUFLFM3Speech(const std::filesystem::path& language,
                               const std::string& device,
                               ov::AnyMap properties) {
    OPENVINO_ASSERT(device == "CPU", "LFM3 GGUF speech currently requires CPU");
    const auto extensions = utils::extract_extensions_to_core(properties);
    if (auto it = properties.find("ATTENTION_BACKEND"); it != properties.end()) {
        OPENVINO_ASSERT(it->second.as<std::string>() == "SDPA", "LFM3 speech uses SDPA");
        properties.erase(it);
    }
    if (std::filesystem::is_directory(language)) {
        m_models.language = utils::singleton_core().read_model(language / "openvino_language_model.xml");
        m_models.vlm = read_lfm3_saved_models(language);
        m_models.tokenizer = Tokenizer(language, properties);
        m_config = GenerationConfig(language / "generation_config.json");
        m_models.stop_token_ids = m_config.stop_token_ids;
        for (const auto& name : {"depthformer", "feedback", "codec"})
            m_models.audio_generation[name] =
                utils::singleton_core().read_model(language / (std::string("openvino_audio_") + name + "_model.xml"));
    } else {
        auto it = properties.find(mmproj_path.name());
        OPENVINO_ASSERT(it != properties.end(), "LFM3 speech requires mmproj_path");
        const auto projector = it->second.as<std::string>();
        properties.erase(it);
        m_models = read_gguf_multimodal(language, projector, properties, extensions, true);
        m_config = m_models.generation_config();
    }
    OPENVINO_ASSERT(m_models.vlm.config.model_type == VLMModelType::LFM3_AUDIO ||
                        m_models.vlm.config.model_type == VLMModelType::LFM3_OMNI,
                    "LFM3 speech requires Audio or Omni");
    OPENVINO_ASSERT(!m_models.audio_generation.empty(), "LFM3 mmproj has no audio generation heads");
    const auto compile = [&](const std::shared_ptr<ov::Model>& model, const std::string& component) {
        return utils::singleton_core()
            .compile_model(model, device, utils::get_model_properties(properties, component, device))
            .create_infer_request();
    };
    m_language = compile(m_models.language, "language_model");
    m_embedding = compile(m_models.vlm.at("text_embeddings"), "text_embeddings");
    m_embedder = std::make_shared<InputsEmbedder>(m_models.vlm, m_models.tokenizer, device, properties);
    m_depth = compile(m_models.audio_generation.at("depthformer"), "audio_depthformer");
    m_feedback = compile(m_models.audio_generation.at("feedback"), "audio_feedback");
    const auto codec = m_models.audio_generation.at("codec");
    m_codec = compile(codec, "audio_codec");
    const auto window = ov::as_type_ptr<ov::op::v0::Constant>(
        codec->output("audio.istft_window").get_node()->input_value(0).get_node_shared_ptr());
    OPENVINO_ASSERT(window, "LFM3 codec must expose its ISTFT window");
    m_istft = std::make_unique<LFM3ISTFT>(window->cast_vector<float>());
    using namespace ov::opset15;
    auto spectrum = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{-1, 641, 2});
    auto inverse = std::make_shared<IRDFT>(spectrum,
                                           Constant::create(ov::element::i64, {1}, {1}),
                                           Constant::create(ov::element::i64, {1}, {1280}));
    m_ifft =
        compile(std::make_shared<ov::Model>(ov::OutputVector{inverse}, ov::ParameterVector{spectrum}), "audio_istft");
    const auto token = [&](const char* text) {
        const auto ids = m_models.tokenizer.encode(text, add_special_tokens(false)).input_ids;
        OPENVINO_ASSERT(ids.get_size() == 1, "Missing LFM3 token ", text);
        return ids.data<const int64_t>()[0];
    };
    m_audio_start = token("<|audio_start|>");
    m_mixed_start = token("<|mixed_start|>");
    m_text_end = token("<|text_end|>");
}

GenerationConfig GGUFLFM3Speech::get_generation_config() const {
    return m_config;
}
std::string GGUFLFM3Speech::default_system_prompt(bool audio) const {
    return audio ? "Respond with interleaved text and audio." : "Respond with text.";
}
void GGUFLFM3Speech::reset() {
    m_language.reset_state();
    m_codec.reset_state();
    m_codec_position = 0;
    m_audio_history.clear();
    m_istft->reset();
}

ov::Tensor GGUFLFM3Speech::sample_frame(const ov::Tensor& hidden,
                                        const OmniTalkerSpeechConfig& speech,
                                        std::mt19937& rng) {
    const auto width = hidden.get_shape().back();
    ov::Tensor source(ov::element::f32,
                      {1, 1, 1, width},
                      const_cast<float*>(hidden.data<const float>() + hidden.get_size() - width));
    ov::Tensor codes(ov::element::i32, {1, 1, 1, 12}), mask(ov::element::f32, {1, 1, 1, 12});
    std::fill_n(codes.data<int32_t>(), 12, 0);
    std::fill_n(mask.data<float>(), 12, 0.f);
    std::vector<size_t> order(12);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    for (size_t pass = 0; pass < 3; ++pass) {
        m_depth.set_tensor("audio.depth.hidden_states", source);
        m_depth.set_tensor("audio.depth.codes", codes);
        m_depth.set_tensor("audio.depth.mask", mask);
        m_depth.infer();
        const auto logits = m_depth.get_tensor("audio.depth_logits");
        OPENVINO_ASSERT(logits.get_size() == 12 * 2017, "LFM3 depthformer must return twelve 2017-entry codebooks");
        for (size_t pick = pass * 4; pick < (pass + 1) * 4; ++pick) {
            const auto book = order[pick];
            std::vector<float> values(logits.data<const float>() + book * 2017,
                                      logits.data<const float>() + (book + 1) * 2017);
            const auto penalty = speech.talker_repetition_penalty.value_or(1.1f);
            std::vector<bool> seen(2017);
            const auto begin = m_audio_history.size() > 4096 * 12 ? m_audio_history.size() - 4096 * 12 : 0;
            for (size_t i = begin + book; i < m_audio_history.size(); i += 12)
                seen[m_audio_history[i]] = true;
            for (size_t id = 0; id < seen.size(); ++id)
                if (seen[id])
                    values[id] = values[id] < 0 ? values[id] * penalty : values[id] / penalty;
            codes.data<int32_t>()[book] = int32_t(sample(std::move(values),
                                                         speech.cp_top_k == 1,
                                                         speech.cp_temperature.value_or(1.f),
                                                         speech.cp_top_k.value_or(128),
                                                         1.f,
                                                         rng));
        }
        for (size_t pick = pass * 4; pick < (pass + 1) * 4; ++pick)
            mask.data<float>()[order[pick]] = 1.f;
    }
    return codes;
}

ov::Tensor GGUFLFM3Speech::decode(const std::vector<int32_t>& codes) {
    OPENVINO_ASSERT(codes.size() % 12 == 0 && !codes.empty() &&
                        std::all_of(codes.begin(),
                                    codes.end(),
                                    [](int32_t code) {
                                        return code >= 0 && code < 2016;
                                    }),
                    "LFM3 codec requires twelve FSQ codes per frame in [0, 2016)");
    const auto frames = codes.size() / 12, steps = frames * 6;
    ov::Tensor input(ov::element::i32, {1, 1, frames, 12}, const_cast<int32_t*>(codes.data()));
    ov::Tensor positions(ov::element::i32, {1, 1, 1, steps});
    std::iota(positions.data<int32_t>(), positions.data<int32_t>() + steps, int32_t(m_codec_position));
    ov::Tensor mask(ov::element::f32, {1, 1, steps, 29 + steps});
    for (size_t q = 0; q < steps; ++q)
        for (size_t k = 0; k < 29 + steps; ++k)
            mask.data<float>()[q * (29 + steps) + k] =
                k >= q && k <= q + 29 && int64_t(m_codec_position) - 29 + int64_t(k) >= 0
                    ? 0.f
                    : -std::numeric_limits<float>::infinity();
    m_codec.set_tensor("audio.codec.codes", input);
    m_codec.set_tensor("audio.codec.positions", positions);
    m_codec.set_tensor("audio.codec.attention_mask", mask);
    m_codec.infer();
    m_codec_position += steps;
    return m_istft->append(m_codec.get_tensor("audio.spectrogram"), m_ifft);
}

OmniDecodedResults GGUFLFM3Speech::generate(const ChatHistory& history,
                                            const std::vector<ov::Tensor>& images,
                                            const std::vector<ov::Tensor>& videos,
                                            const std::vector<VideoMetadata>& metadata,
                                            const std::vector<ov::Tensor>& audios,
                                            const GenerationConfig& requested,
                                            const OmniTalkerSpeechConfig& speech,
                                            const StreamerVariant& streamer,
                                            const OmniSpeechStreamerVariant& speech_streamer) {
    std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
    OPENVINO_ASSERT(lock.owns_lock(), "Concurrent LFM3 generation is not supported");
    auto config = requested;
    config.set_eos_token_id(m_models.tokenizer.get_eos_token_id());
    config.stop_token_ids.insert(m_models.stop_token_ids.begin(), m_models.stop_token_ids.end());
    config.validate();
    validate_omni_talker_speech_config(speech);
    OPENVINO_ASSERT(config.num_beams == 1 && config.num_return_sequences == 1 && !config.is_prompt_lookup() &&
                        !config.is_assisting_generation(),
                    "LFM3 speech requires a single autoregressive sequence");
    OPENVINO_ASSERT(!config.echo && config.logprobs == 0 && config.min_new_tokens == 0 && config.min_p == 0.f &&
                        config.stop_strings.empty() && config.parsers.empty() && !config.structured_output_config &&
                        !config.adapters && config.pruning_ratio == 0 && config.repetition_penalty == 1.f &&
                        config.presence_penalty == 0.f && config.frequency_penalty == 0.f &&
                        (config.no_repeat_ngram_size == 0 || config.no_repeat_ngram_size == SIZE_MAX),
                    "Unsupported LFM3 speech text generation option");
    OPENVINO_ASSERT(std::holds_alternative<std::string>(speech.speaker) &&
                        std::get<std::string>(speech.speaker).empty() && !speech.talker_temperature &&
                        !speech.talker_top_k,
                    "LFM3 uses a fixed voice and cp sampling options");
    const auto started = std::chrono::steady_clock::now();
    OmniDecodedResults result;
    if (!m_registry)
        m_registry = std::make_shared<VisionRegistry>();
    VLMChatContext chat(history, m_registry, *m_embedder);
    auto text_streamer = utils::create_streamer(streamer, m_models.tokenizer);
    bool cancelled = false, stopped = false, speaking = false, text_done = false;
    enum Mode { TEXT, AUDIO, MIXED };
    Mode mode = TEXT;
    size_t remaining = m_text_cadence, audio_frames = 0;
    std::vector<int64_t> text;
    std::vector<int32_t> pending;
    std::vector<ov::Tensor> pcm;
    const auto emit = [&](ov::Tensor chunk) {
        if (!chunk.get_size())
            return;
        if (cancelled)
            return;
        pcm.push_back(chunk);
        if (stopped)
            return;
        StreamingStatus status = StreamingStatus::RUNNING;
        if (auto callback = std::get_if<std::function<StreamingStatus(const ov::Tensor&)>>(&speech_streamer))
            status = (*callback)(chunk);
        else if (auto object = std::get_if<std::shared_ptr<OmniSpeechStreamerBase>>(&speech_streamer)) {
            OPENVINO_ASSERT(*object, "Speech streamer is null");
            status = (*object)->write(chunk);
        }
        stopped = status != StreamingStatus::RUNNING;
        cancelled = status == StreamingStatus::CANCEL;
    };
    try {
        reset();
        auto data = chat.process(images, videos, metadata, audios);
        auto prompt = m_models.tokenizer.apply_chat_template(data.normalized_history, true);
        if (speech.return_audio)
            prompt += "<|mixed_start|>";
        m_embedder->finish_chat();
        m_embedder->set_apply_chat_template_status(false, true);
        auto embeds = m_embedder->get_inputs_embeds(prompt,
                                                    data.encoded_images,
                                                    data.encoded_videos,
                                                    data.encoded_audios,
                                                    result.perf_metrics,
                                                    true,
                                                    data.image_sequence,
                                                    data.video_sequence,
                                                    data.audio_sequence,
                                                    data.vision_counts);
        mode = speech.return_audio ? MIXED : TEXT;
        result.perf_metrics.num_input_tokens = embeds.get_shape()[1];
        std::mt19937 text_rng{uint32_t(config.rng_seed)}, audio_rng{uint32_t(speech.rng_seed)};
        size_t past = 0;
        for (size_t step = 0; step < config.get_max_new_tokens(result.perf_metrics.num_input_tokens) && !stopped;
             ++step) {
            const auto length = embeds.get_shape()[1];
            ov::Tensor mask(ov::element::i64, {1, past + length}), positions(ov::element::i64, {1, length}),
                beam(ov::element::i32, {1});
            std::fill_n(mask.data<int64_t>(), mask.get_size(), 1);
            std::iota(positions.data<int64_t>(), positions.data<int64_t>() + length, int64_t(past));
            beam.data<int32_t>()[0] = 0;
            m_language.set_tensor("inputs_embeds", embeds);
            m_language.set_tensor("attention_mask", mask);
            m_language.set_tensor("position_ids", positions);
            m_language.set_tensor("beam_idx", beam);
            m_language.infer();
            past += length;
            const bool interleaved = mode == MIXED;
            if (mode == AUDIO || (interleaved && speaking)) {
                if (audio_frames >= speech.max_new_tokens)
                    break;
                auto codes = sample_frame(m_language.get_tensor("audio_hidden_states"), speech, audio_rng);
                const bool eos =
                    std::any_of(codes.data<const int32_t>(), codes.data<const int32_t>() + 12, [](int32_t code) {
                        return code == 2016;
                    });
                if (eos) {
                    std::fill_n(codes.data<int32_t>(), 12, 2016);
                    mode = TEXT;
                    if (!pending.empty()) {
                        emit(decode(pending));
                        pending.clear();
                    }
                    emit(m_istft->flush());
                    m_codec.reset_state();
                    m_codec_position = 0;
                    m_istft->reset();
                } else {
                    pending.insert(pending.end(), codes.data<const int32_t>(), codes.data<const int32_t>() + 12);
                    m_audio_history.insert(m_audio_history.end(),
                                           codes.data<const int32_t>(),
                                           codes.data<const int32_t>() + 12);
                    ++audio_frames;
                    if (pending.size() / 12 >= speech.audio_chunk_frames) {
                        emit(decode(pending));
                        pending.clear();
                    }
                }
                m_feedback.set_tensor("audio.feedback.codes", codes);
                m_feedback.infer();
                auto feedback = owned(m_feedback.get_tensor("audio.feedback"));
                feedback.set_shape({1, 1, feedback.get_shape().back()});
                embeds = feedback;
            } else {
                const auto values = m_language.get_tensor("logits");
                const auto count = values.get_shape().back();
                std::vector<float> logits(values.data<const float>() + values.get_size() - count,
                                          values.data<const float>() + values.get_size());
                if (!speech.return_audio) {
                    logits.at(m_audio_start) = -std::numeric_limits<float>::infinity();
                    logits.at(m_mixed_start) = -std::numeric_limits<float>::infinity();
                }
                const auto id = sample(std::move(logits),
                                       !config.do_sample,
                                       config.temperature,
                                       config.top_k,
                                       config.top_p,
                                       text_rng);
                text.push_back(id);
                if (text_streamer) {
                    const auto status = text_streamer->write(id);
                    stopped = status != StreamingStatus::RUNNING;
                    cancelled = status == StreamingStatus::CANCEL;
                }
                if (id == m_text_end) {
                    if (interleaved) {
                        text_done = true;
                        speaking = true;
                    } else
                        break;
                } else if (config.stop_token_ids.count(id) && !(config.ignore_eos && id == config.eos_token_id))
                    break;
                else if (!interleaved && id == m_audio_start)
                    mode = AUDIO;
                else if (!interleaved && id == m_mixed_start) {
                    mode = MIXED;
                    remaining = m_text_cadence;
                    speaking = false;
                    text_done = false;
                }
                ov::Tensor token(ov::element::i64, {1, 1});
                token.data<int64_t>()[0] = id;
                m_embedding.set_input_tensor(token);
                m_embedding.infer();
                embeds = owned(m_embedding.get_output_tensor());
            }
            if (interleaved && !text_done && --remaining == 0) {
                speaking = !speaking;
                remaining = speaking ? m_audio_cadence : m_text_cadence;
            }
        }
        if (!cancelled) {
            if (!pending.empty()) {
                emit(decode(pending));
                pending.clear();
            }
            emit(m_istft->flush());
        }
        if (text_streamer)
            text_streamer->end();
        if (auto object = std::get_if<std::shared_ptr<OmniSpeechStreamerBase>>(&speech_streamer)) {
            OPENVINO_ASSERT(*object, "Speech streamer is null");
            (*object)->end();
        }
        result.texts = {m_models.tokenizer.decode(text)};
        result.scores = {0.f};
        result.perf_metrics.num_generated_tokens = text.size();
        if (cancelled)
            chat.rollback();
        else if (!pcm.empty()) {
            size_t samples = 0;
            for (const auto& chunk : pcm)
                samples += chunk.get_size();
            ov::Tensor waveform(ov::element::f32, {1, samples});
            auto* target = waveform.data<float>();
            for (const auto& chunk : pcm) {
                std::copy_n(chunk.data<const float>(), chunk.get_size(), target);
                target += chunk.get_size();
            }
            result.speech_result.waveforms = {waveform};
            result.speech_result.perf_metrics.num_generated_samples = samples;
        }
        result.speech_result.perf_metrics.generation_time_ms =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - started).count();
        reset();
        return result;
    } catch (...) {
        chat.rollback();
        m_embedder->finish_chat();
        reset();
        throw;
    }
}
}  // namespace ov::genai
#endif
