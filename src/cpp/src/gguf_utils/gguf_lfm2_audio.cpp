// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_lfm2_audio.hpp"
#ifdef ENABLE_GGUF
#    include <algorithm>
#    include <chrono>
#    include <cmath>
#    include <cstring>
#    include <numeric>
#    include <unordered_set>

#    include "gguf.hpp"
#    include "gguf_frontend_extensions.hpp"
#    include "omni/talker_speech_config_utils.hpp"
#    include "openvino/frontend/gguf/extension/genai.hpp"
#    include "openvino/frontend/gguf/frontend.hpp"
#    include "openvino/genai/omni/pipeline.hpp"
#    include "openvino/opsets/opset15.hpp"
#    include "utils.hpp"
#    include "visual_language/vision_registry.hpp"
#    include "visual_language/vlm_chat_context.hpp"

namespace ov::genai {
namespace {
std::shared_ptr<ov::Model> submodel(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    auto selected = model->output(name);
    std::unordered_set<const ov::Node*> live;
    const auto visit = [&](auto&& self, const ov::Output<ov::Node>& value) -> void {
        if (!live.insert(value.get_node()).second)
            return;
        for (const auto& input : value.get_node()->input_values())
            self(self, input);
    };
    visit(visit, selected);
    ov::ParameterVector parameters;
    for (const auto& p : model->get_parameters())
        if (live.count(p.get()))
            parameters.push_back(p);
    return std::make_shared<ov::Model>(ov::OutputVector{selected}, parameters);
}
ov::Tensor owned(const ov::Tensor& tensor) {
    ov::Tensor result(tensor.get_element_type(), tensor.get_shape());
    tensor.copy_to(result);
    return result;
}
}  // namespace

GGUFLFM2Speech::GGUFLFM2Speech(const std::filesystem::path& language,
                               const std::string& device,
                               ov::AnyMap properties) {
    OPENVINO_ASSERT(device == "CPU", "LFM2 GGUF speech generation currently requires CPU");
    const auto extensions = utils::extract_extensions_to_core(properties);
    const auto path = [&](const std::string& name) {
        auto it = properties.find(name);
        OPENVINO_ASSERT(it != properties.end(), "LFM2 speech requires ", name);
        const auto value = it->second.as<std::string>();
        properties.erase(it);
        return value;
    };
    if (auto it = properties.find("ATTENTION_BACKEND"); it != properties.end()) {
        OPENVINO_ASSERT(it->second.as<std::string>() == "SDPA", "LFM2 speech generation uses the SDPA backend");
        properties.erase(it);
    }
    const auto projector = path(mmproj_path.name());
    const auto vocoder = path(vocoder_path.name());
    const auto detokenizer = path(audio_detokenizer_path.name());
    if (auto it = properties.find(lfm2_audio_mode.name()); it != properties.end()) {
        const auto mode = it->second.as<std::string>();
        OPENVINO_ASSERT(mode == "interleaved" || mode == "sequential", "Invalid LFM2 audio mode");
        m_interleaved = mode == "interleaved";
        properties.erase(it);
    }
    const auto metadata = get_gguf_metadata(vocoder);
    const auto cadence = [&](const std::string& key, size_t fallback) {
        const auto it = metadata.find(key);
        if (it == metadata.end())
            return fallback;
        const auto tensor = std::get<ov::Tensor>(it->second);
        const auto value = tensor.data<const uint32_t>()[0];
        OPENVINO_ASSERT(value > 0, "LFM2 audio cadence must be positive");
        return size_t(value);
    };
    m_text_cadence = cadence("interleaved_n_text", 6);
    m_audio_cadence = cadence("interleaved_n_audio", 12);
    m_models = read_gguf_multimodal(language, projector, properties, extensions);
    OPENVINO_ASSERT(m_models.vlm.config.model_type == VLMModelType::LFM2_AUDIO, "LFM2 speech requires an audio model");
    const auto head = ov::as_type_ptr<ov::op::v0::MatMul>(
        m_models.language->output("logits").get_node()->input_value(0).get_node_shared_ptr());
    OPENVINO_ASSERT(head, "LFM2 language model has no output projection");
    auto hidden = head->input_value(0);
    auto hidden_result = std::make_shared<ov::op::v0::Result>(hidden);
    hidden_result->output(0).set_names({"audio_hidden_states"});
    m_models.language->add_results({hidden_result});
    const auto compile = [&](const std::shared_ptr<ov::Model>& model, const std::string& component) {
        return utils::singleton_core()
            .compile_model(model, device, utils::get_model_properties(properties, component, device))
            .create_infer_request();
    };
    m_language = compile(m_models.language, "language_model");
    m_embedding = compile(m_models.vlm.at("text_embeddings"), "text_embeddings");
    m_embedder = std::make_shared<InputsEmbedder>(m_models.vlm, m_models.tokenizer, device, properties);
    ov::frontend::gguf::FrontEnd frontend;
    register_gguf_extensions(frontend, extensions);
    auto speech = frontend.convert(frontend.load(vocoder));
    m_depth = compile(submodel(speech, "audio.depth_logits"), "audio_depthformer");
    m_feedback = compile(submodel(speech, "audio.feedback"), "audio_feedback");
    m_codec = compile(submodel(speech, "audio.codec_features"), "audio_codec");
    const auto window = ov::as_type_ptr<ov::op::v0::Constant>(
        speech->output("audio.istft_window").get_node()->input_value(0).get_node_shared_ptr());
    OPENVINO_ASSERT(window && ov::shape_size(window->get_shape()) == 1280,
                    "LFM2 speech requires a 1280-sample ISTFT window");
    m_window = window->cast_vector<float>();
    ov::frontend::gguf::FrontEnd detok;
    register_gguf_extensions(detok, extensions, ov::frontend::gguf::GenAIExtension::InputMode::EMBEDS_TO_LOGITS);
    m_detokenizer = compile(detok.convert(detok.load(detokenizer)), "audio_detokenizer");
    using namespace ov::opset15;
    auto spectrum = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{-1, 641, 2});
    auto frames = std::make_shared<IRDFT>(spectrum,
                                          Constant::create(ov::element::i64, {1}, {1}),
                                          Constant::create(ov::element::i64, {1}, {1280}));
    m_ifft =
        compile(std::make_shared<ov::Model>(ov::OutputVector{frames}, ov::ParameterVector{spectrum}), "audio_istft");
}

GenerationConfig GGUFLFM2Speech::get_generation_config() const {
    return m_models.generation_config();
}

std::string GGUFLFM2Speech::default_system_prompt(bool return_audio) const {
    if (!return_audio)
        return "Respond with text.";
    return m_interleaved ? "Respond with interleaved text and audio." : "Perform TTS. Use the UK female voice.";
}

int64_t GGUFLFM2Speech::sample(const float* logits,
                               size_t count,
                               bool greedy,
                               float temperature,
                               size_t top_k,
                               float top_p,
                               std::mt19937& rng) {
    if (greedy || temperature <= 0.f || top_k == 1)
        return std::max_element(logits, logits + count) - logits;
    std::vector<size_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return logits[a] > logits[b];
    });
    if (top_k && top_k < count)
        indices.resize(top_k);
    std::vector<double> probabilities;
    double sum = 0.;
    for (const auto id : indices) {
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

ov::Tensor GGUFLFM2Speech::decode(const std::vector<int32_t>& codes) {
    const auto count = codes.size() / 8;
    ov::Tensor input(ov::element::i32, {1, 1, count, 8}, const_cast<int32_t*>(codes.data()));
    m_codec.set_tensor("audio.codes", input);
    m_codec.infer();
    const auto codec = m_codec.get_output_tensor();
    ov::Tensor upsampled(ov::element::f32, {1, count * 6, 512});
    for (size_t i = 0; i < count * 6; ++i)
        std::copy_n(codec.data<const float>() + (i / 6) * 512, 512, upsampled.data<float>() + i * 512);
    m_detokenizer.reset_state();
    const auto frames = count * 6;
    ov::Tensor mask(ov::element::i64, {1, frames}), positions(ov::element::i64, {1, frames});
    std::fill_n(mask.data<int64_t>(), frames, 1);
    std::iota(positions.data<int64_t>(), positions.data<int64_t>() + frames, 0);
    ov::Tensor beam(ov::element::i32, {1});
    beam.data<int32_t>()[0] = 0;
    m_detokenizer.set_tensor("inputs_embeds", upsampled);
    m_detokenizer.set_tensor("attention_mask", mask);
    m_detokenizer.set_tensor("position_ids", positions);
    m_detokenizer.set_tensor("beam_idx", beam);
    m_detokenizer.infer();
    const auto values = m_detokenizer.get_tensor("logits");
    OPENVINO_ASSERT(values.get_size() == frames * 1282, "LFM2 audio detokenizer must produce every spectral frame");
    return lfm2_spectrum_to_waveform(values, m_window, m_ifft);
}

ov::Tensor lfm2_spectrum_to_waveform(const ov::Tensor& values,
                                     const std::vector<float>& window,
                                     ov::InferRequest& inverse_fft) {
    OPENVINO_ASSERT(values.get_element_type() == ov::element::f32 && values.get_shape().size() == 3 &&
                        values.get_shape()[0] == 1 && values.get_shape()[2] == 1282 && window.size() == 1280,
                    "Invalid LFM2 spectral output or ISTFT window");
    const auto frames = values.get_shape()[1];
    OPENVINO_ASSERT(frames > 0, "LFM2 ISTFT requires spectral frames");
    ov::Tensor spectrum(ov::element::f32, {frames, 641, 2});
    for (size_t t = 0; t < frames; ++t)
        for (size_t k = 0; k < 641; ++k) {
            const auto magnitude = std::exp(values.data<const float>()[t * 1282 + k]);
            const auto angle = values.data<const float>()[t * 1282 + 641 + k];
            spectrum.data<float>()[(t * 641 + k) * 2] = magnitude * std::cos(angle);
            spectrum.data<float>()[(t * 641 + k) * 2 + 1] = magnitude * std::sin(angle);
        }
    inverse_fft.set_input_tensor(spectrum);
    inverse_fft.infer();
    const auto inverse = inverse_fft.get_output_tensor();
    std::vector<double> waveform((frames - 1) * 320 + 1280), envelope(waveform.size());
    for (size_t t = 0; t < frames; ++t)
        for (size_t i = 0; i < 1280; ++i) {
            waveform[t * 320 + i] += inverse.data<const float>()[t * 1280 + i] * window[i];
            envelope[t * 320 + i] += double(window[i]) * window[i];
        }
    ov::Tensor output(ov::element::f32, {1, frames * 320});
    for (size_t i = 0; i < output.get_size(); ++i)
        output.data<float>()[i] = float(waveform[i + 480] / envelope[i + 480]);
    return output;
}

OmniDecodedResults GGUFLFM2Speech::generate(const ChatHistory& history,
                                            const std::vector<ov::Tensor>& audios,
                                            const GenerationConfig& requested_config,
                                            const OmniTalkerSpeechConfig& speech,
                                            const StreamerVariant& streamer,
                                            const OmniSpeechStreamerVariant& speech_streamer) {
    std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
    OPENVINO_ASSERT(lock.owns_lock(), "Concurrent LFM2 speech generation is not supported");
    GenerationConfig config = requested_config;
    config.set_eos_token_id(m_models.tokenizer.get_eos_token_id());
    config.stop_token_ids.insert(m_models.stop_token_ids.begin(), m_models.stop_token_ids.end());
    config.validate();
    validate_omni_talker_speech_config(speech);
    OPENVINO_ASSERT(!config.echo && config.logprobs == 0 && config.min_new_tokens == 0 && config.min_p == 0.f &&
                        config.stop_strings.empty() && config.parsers.empty() && !config.structured_output_config &&
                        !config.adapters && config.pruning_ratio == 0 &&
                        (config.no_repeat_ngram_size == 0 || config.no_repeat_ngram_size == SIZE_MAX),
                    "Unsupported LFM2 speech text generation option");
    OPENVINO_ASSERT(speech.audio_chunk_frames <= SIZE_MAX / 1920, "Audio chunk size is too large");
    OPENVINO_ASSERT(std::holds_alternative<std::string>(speech.speaker) &&
                        std::get<std::string>(speech.speaker).empty() && !speech.talker_temperature &&
                        !speech.talker_top_k && !speech.talker_repetition_penalty,
                    "LFM2 speech uses a fixed voice and cp_temperature/cp_top_k for audio sampling");
    OPENVINO_ASSERT(config.num_beams == 1 && config.num_return_sequences == 1 && !config.is_prompt_lookup() &&
                        !config.is_assisting_generation(),
                    "LFM2 speech generation requires a single autoregressive sequence");
    OPENVINO_ASSERT(
        config.repetition_penalty == 1.f && config.presence_penalty == 0.f && config.frequency_penalty == 0.f,
        "LFM2 speech generation does not yet support token penalties");
    const auto started = std::chrono::steady_clock::now();
    OmniDecodedResults result;
    if (!m_registry)
        m_registry = std::make_shared<VisionRegistry>();
    VLMChatContext chat(history, m_registry, *m_embedder);
    try {
        auto data = chat.process({}, {}, {}, audios);
        auto prompt = m_models.tokenizer.apply_chat_template(data.normalized_history, true);
        m_embedder->finish_chat();
        m_embedder->set_apply_chat_template_status(false, true);
        auto embeds = m_embedder->get_inputs_embeds(prompt,
                                                    {},
                                                    {},
                                                    data.encoded_audios,
                                                    result.perf_metrics,
                                                    true,
                                                    {},
                                                    {},
                                                    data.audio_sequence,
                                                    {});
        m_language.reset_state();
        size_t past = 0;
        bool audio_mode = false, text_done = false;
        size_t remaining = m_text_cadence;
        std::mt19937 text_rng{uint32_t(config.rng_seed)}, audio_rng{uint32_t(speech.rng_seed)};
        std::vector<int64_t> text;
        std::vector<int32_t> codes;
        auto text_streamer = utils::create_streamer(streamer, m_models.tokenizer);
        result.perf_metrics.num_input_tokens = embeds.get_shape()[1];
        const auto max_steps = config.get_max_new_tokens(embeds.get_shape()[1]);
        bool cancelled = false;
        for (size_t step = 0; step < max_steps; ++step) {
            const auto length = embeds.get_shape()[1];
            ov::Tensor mask(ov::element::i64, {1, past + length}), positions(ov::element::i64, {1, length});
            std::fill_n(mask.data<int64_t>(), mask.get_size(), 1);
            std::iota(positions.data<int64_t>(), positions.data<int64_t>() + length, int64_t(past));
            ov::Tensor beam(ov::element::i32, {1});
            beam.data<int32_t>()[0] = 0;
            m_language.set_tensor("inputs_embeds", embeds);
            m_language.set_tensor("attention_mask", mask);
            m_language.set_tensor("position_ids", positions);
            m_language.set_tensor("beam_idx", beam);
            m_language.infer();
            past += length;
            if (!audio_mode) {
                const auto logits = m_language.get_tensor("logits");
                const auto id = sample(logits.data<const float>(),
                                       logits.get_size(),
                                       !config.do_sample,
                                       config.temperature,
                                       config.top_k,
                                       config.top_p,
                                       text_rng);
                if (config.stop_token_ids.count(id) && !(config.ignore_eos && id == config.eos_token_id))
                    break;
                text.push_back(id);
                if (text_streamer) {
                    const auto status = text_streamer->write(id);
                    if (status != StreamingStatus::RUNNING) {
                        cancelled = status == StreamingStatus::CANCEL;
                        break;
                    }
                }
                if (speech.return_audio) {
                    text_done = text_done || id == 130;
                    if (m_interleaved ? (--remaining == 0 || text_done) : id == 128) {
                        audio_mode = true;
                        remaining = m_audio_cadence;
                    }
                }
                ov::Tensor token(ov::element::i64, {1, 1});
                token.data<int64_t>()[0] = id;
                m_embedding.set_input_tensor(token);
                m_embedding.infer();
                embeds = owned(m_embedding.get_output_tensor());
            } else {
                if (codes.size() / 8 >= speech.max_new_tokens)
                    break;
                auto source = m_language.get_tensor("audio_hidden_states");
                ov::Tensor hidden(ov::element::f32, {1, 1, 1, 2048}, source.data());
                ov::Tensor depth(ov::element::i32, {1, 1, 1, 8});
                std::fill_n(depth.data<int32_t>(), 8, 0);
                for (size_t book = 0; book < 8; ++book) {
                    m_depth.set_tensor("audio.hidden_states", hidden);
                    m_depth.set_tensor("audio.depth_codes", depth);
                    m_depth.infer();
                    const auto logits = m_depth.get_output_tensor();
                    depth.data<int32_t>()[book] = int32_t(sample(logits.data<const float>() + book * 2049,
                                                                 2049,
                                                                 !speech.cp_temperature || speech.cp_top_k == 1,
                                                                 speech.cp_temperature.value_or(1.f),
                                                                 speech.cp_top_k.value_or(0),
                                                                 1.f,
                                                                 audio_rng));
                }
                const bool end = depth.data<int32_t>()[0] == 2048;
                if (end)
                    std::fill_n(depth.data<int32_t>(), 8, 2048);
                else {
                    if (std::none_of(depth.data<int32_t>(), depth.data<int32_t>() + 8, [](int32_t code) {
                            return code == 2048;
                        }))
                        codes.insert(codes.end(), depth.data<int32_t>(), depth.data<int32_t>() + 8);
                }
                if (codes.size() / 8 >= speech.max_new_tokens)
                    break;
                m_feedback.set_tensor("audio.codes", depth);
                m_feedback.infer();
                embeds = owned(m_feedback.get_output_tensor());
                embeds.set_shape({1, 1, 2048});
                if (end || (m_interleaved && --remaining == 0 && !text_done)) {
                    audio_mode = false;
                    remaining = m_text_cadence;
                }
            }
        }
        if (text_streamer)
            text_streamer->end();
        result.perf_metrics.num_generated_tokens = text.size();
        result.texts = {m_models.tokenizer.decode(text)};
        result.scores = {0.f};
        if (!codes.empty() && !cancelled) {
            auto waveform = decode(codes);
            result.speech_result.perf_metrics.num_generated_samples = waveform.get_size();
            result.speech_result.waveforms = {waveform};
            const auto chunk_size = speech.audio_chunk_frames * 1920;
            for (size_t offset = 0; offset < waveform.get_size(); offset += chunk_size) {
                const auto size = std::min(chunk_size, waveform.get_size() - offset);
                ov::Tensor chunk(ov::element::f32, {1, size});
                std::copy_n(waveform.data<const float>() + offset, size, chunk.data<float>());
                StreamingStatus status = StreamingStatus::RUNNING;
                if (auto callback = std::get_if<std::function<StreamingStatus(const ov::Tensor&)>>(&speech_streamer))
                    status = (*callback)(chunk);
                else if (auto object = std::get_if<std::shared_ptr<OmniSpeechStreamerBase>>(&speech_streamer)) {
                    OPENVINO_ASSERT(*object, "Speech streamer is null");
                    status = (*object)->write(chunk);
                }
                if (status != StreamingStatus::RUNNING) {
                    cancelled = status == StreamingStatus::CANCEL;
                    break;
                }
            }
        }
        if (auto object = std::get_if<std::shared_ptr<OmniSpeechStreamerBase>>(&speech_streamer)) {
            OPENVINO_ASSERT(*object, "Speech streamer is null");
            (*object)->end();
        }
        result.speech_result.perf_metrics.generation_time_ms =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - started).count();
        if (cancelled) {
            chat.rollback();
            result.speech_result.waveforms.clear();
        }
        return result;
    } catch (...) {
        chat.rollback();
        m_embedder->finish_chat();
        m_language.reset_state();
        throw;
    }
}
}  // namespace ov::genai
#endif
