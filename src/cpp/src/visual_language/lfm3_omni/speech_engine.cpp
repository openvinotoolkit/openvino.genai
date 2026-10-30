// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/lfm3_omni/speech_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <set>

#include "json_utils.hpp"
#include "omni/speech_streamer_utils.hpp"

namespace ov::genai {

namespace {

// One request per submodel: the engine decodes one stream at a time, on the single language model request it borrows.
ov::InferRequest compile(const std::shared_ptr<ov::Model>& model,
                         const std::string& name,
                         const std::string& device,
                         const ov::AnyMap& properties,
                         const ov::AnyMap& defaults = {}) {
    ov::AnyMap model_properties = utils::get_model_properties(properties, name, device);
    // emplace keeps the values the caller set.
    for (const auto& [key, value] : defaults) {
        model_properties.emplace(key, value);
    }
    auto compiled = utils::singleton_core().compile_model(model, device, model_properties);
    ov::genai::utils::print_compiled_model_properties(compiled, ("LFM3-Omni " + name + " model").c_str());
    return compiled.create_infer_request();
}

nlohmann::json read_json(const std::filesystem::path& path) {
    std::ifstream stream(path);
    OPENVINO_ASSERT(stream.is_open(), "Failed to open '", path.string(), "'");
    return nlohmann::json::parse(stream);
}

ov::Tensor copy_tensor(const ov::Tensor& source, const ov::Shape& shape) {
    ov::Tensor result(source.get_element_type(), shape);
    OPENVINO_ASSERT(result.get_byte_size() == source.get_byte_size(), "LFM3-Omni: unexpected tensor size");
    std::memcpy(result.data(), source.data(), source.get_byte_size());
    return result;
}

// Softmax of scores in place; -inf entries get probability 0.
void softmax(std::vector<float>& scores) {
    const float max_score = *std::max_element(scores.begin(), scores.end());
    double sum = 0.0;
    for (float& score : scores) {
        score = std::exp(score - max_score);
        sum += score;
    }
    for (float& score : scores) {
        score = static_cast<float>(score / sum);
    }
}

constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

// Left context of a re-decoded streaming chunk. The codec decoder reaches about 20 frames back: three sliding-window
// attention layers of 30 steps and five convolutions of 7 taps, at 6 steps per frame.
constexpr size_t STREAM_CONTEXT_FRAMES = 24;

// Dynamic activation quantization makes the codec output depend on how many frames are decoded together, so a
// re-decoded streaming chunk would not match the one-shot waveform.
const ov::AnyMap AUDIO_DECODER_DEFAULTS{ov::hint::dynamic_quantization_group_size(0)};

}  // namespace

void LFM3SpeechRequest::validate(const OmniTalkerSpeechConfig& speech_config) {
    const auto* speaker = std::get_if<std::string>(&speech_config.speaker);
    OPENVINO_ASSERT(speaker && speaker->empty(),
                    "LFM3-Omni has no speaker selection; describe the voice in the system prompt");
    OPENVINO_ASSERT(!speech_config.talker_temperature && !speech_config.talker_top_k &&
                        !speech_config.talker_repetition_penalty,
                    "LFM3-Omni has no separate talker; use the cp_* fields for speech sampling");
}

LFM3SpeechRequest::LFM3SpeechRequest(const OmniTalkerSpeechConfig& speech_config, const GenerationConfig& text_config)
    : mode(speech_config.output_mode),
      text_config(text_config),
      audio_temperature(speech_config.cp_temperature),
      audio_top_k(speech_config.cp_top_k),
      audio_repetition_penalty(speech_config.cp_repetition_penalty),
      max_audio_frames(speech_config.max_new_tokens),
      audio_chunk_frames(speech_config.audio_chunk_frames),
      audio_rng_seed(speech_config.rng_seed) {
    validate(speech_config);
    OPENVINO_ASSERT(max_audio_frames > 0 && audio_chunk_frames > 0,
                    "LFM3-Omni speech output needs max_new_tokens and audio_chunk_frames of at least 1");
    OPENVINO_ASSERT(text_config.num_beams == 1 && text_config.num_return_sequences == 1 &&
                        !text_config.is_assisting_generation() && !text_config.is_prompt_lookup(),
                    "LFM3-Omni speech output supports one greedy or sampled continuation, without beam search, "
                    "assisted generation or prompt lookup");
    OPENVINO_ASSERT(text_config.stop_strings.empty() && !text_config.ignore_eos && text_config.min_new_tokens == 0 &&
                        text_config.presence_penalty == 0.0f && text_config.frequency_penalty == 0.0f &&
                        text_config.no_repeat_ngram_size == std::numeric_limits<size_t>::max() &&
                        text_config.logprobs == 0 && !text_config.structured_output_config && !text_config.echo,
                    "LFM3-Omni speech output does not implement stop_strings, ignore_eos, min_new_tokens, presence or "
                    "frequency penalties, no_repeat_ngram_size, logprobs, structured output or echo");
}

LFM3OmniSpeechEngine::ModelConfig LFM3OmniSpeechEngine::read_model_config(
    const std::filesystem::path& config_dir_path) {
    using utils::read_json_param;
    ModelConfig config;
    const auto model_json = read_json(config_dir_path / "config.json");
    read_json_param(model_json, "num_codebooks", config.num_codebooks);
    read_json_param(model_json, "codebook_size", config.codebook_size);
    read_json_param(model_json, "audio_eos_code", config.audio_eos_code);
    read_json_param(model_json, "maskgit_schedule", config.maskgit_schedule);
    read_json_param(model_json, "interleaved_n_text", config.interleaved_n_text);
    read_json_param(model_json, "interleaved_n_audio", config.interleaved_n_audio);
    read_json_param(model_json, "audio_out_token_id", config.audio_out_token_id);
    read_json_param(model_json, "audio_start_token_id", config.audio_start_token_id);
    read_json_param(model_json, "mixed_start_token_id", config.mixed_start_token_id);
    read_json_param(model_json, "text_end_token_id", config.text_end_token_id);
    for (const char* name : {"audio_token_id", "audio_out_token_id", "image_token_id", "video_token_id"}) {
        int64_t token_id = -1;
        read_json_param(model_json, name, token_id);
        OPENVINO_ASSERT(token_id >= 0, "LFM3-Omni config.json has no ", name);
        config.placeholder_token_ids.push_back(token_id);
    }
    OPENVINO_ASSERT(config.audio_start_token_id >= 0 && config.mixed_start_token_id >= 0 &&
                        config.text_end_token_id >= 0,
                    "LFM3-Omni config.json must define audio_start_token_id, mixed_start_token_id and "
                    "text_end_token_id");
    OPENVINO_ASSERT(std::accumulate(config.maskgit_schedule.begin(), config.maskgit_schedule.end(), size_t{0}) ==
                        config.num_codebooks,
                    "LFM3-Omni maskgit_schedule must reveal each of the ", config.num_codebooks, " codebooks once");
    OPENVINO_ASSERT(config.interleaved_n_text > 0 && config.interleaved_n_audio > 0,
                    "LFM3-Omni interleaved block sizes must be positive");
    OPENVINO_ASSERT(config.num_codebooks > 0 && config.audio_eos_code >= 0 &&
                        static_cast<size_t>(config.audio_eos_code) < config.codebook_size,
                    "LFM3-Omni config.json needs num_codebooks > 0 and audio_eos_code < codebook_size, got ",
                    config.num_codebooks, ", ", config.audio_eos_code, ", ", config.codebook_size);

    // Speech sampling defaults; generation_config.json is optional, as for the text decode.
    const auto generation_config_path = config_dir_path / "generation_config.json";
    if (std::filesystem::exists(generation_config_path)) {
        const auto generation_json = read_json(generation_config_path);
        read_json_param(generation_json, "audio_temperature", config.audio_temperature);
        read_json_param(generation_json, "audio_top_k", config.audio_top_k);
        read_json_param(generation_json, "audio_repetition_penalty", config.audio_repetition_penalty);
        read_json_param(generation_json, "audio_repetition_window", config.audio_repetition_window);
        read_json_param(generation_json, "text_repetition_window", config.text_repetition_window);
    }
    OPENVINO_ASSERT(!config.audio_temperature || *config.audio_temperature >= 0.0f,
                    "LFM3-Omni generation_config.json audio_temperature must be >= 0");
    OPENVINO_ASSERT(!config.audio_top_k || *config.audio_top_k >= 1,
                    "LFM3-Omni generation_config.json audio_top_k must be >= 1");
    OPENVINO_ASSERT(config.audio_repetition_penalty > 0.0f,
                    "LFM3-Omni generation_config.json audio_repetition_penalty must be > 0");

    const auto codec_json = read_json(config_dir_path / "audio_tokenizer" / "config.json");
    size_t frame_stack = 6;
    size_t hop_length = 320;
    size_t win_length = 1280;
    read_json_param(codec_json, "stft_frame_stack", frame_stack);
    read_json_param(codec_json, "stft_hop_length", hop_length);
    read_json_param(codec_json, "stft_win_length", win_length);
    // Each codec frame is repeated frame_stack times before the ISTFT, one hop each.
    config.frame_samples = frame_stack * hop_length;
    size_t frame_samples = config.frame_samples;
    read_json_param(codec_json, "frame_samples", frame_samples);
    // The exported decoder trims (win_length - hop_length) / 2 samples on each side of the overlap-add.
    config.decode_lookahead_samples = (win_length - hop_length) / 2;
    OPENVINO_ASSERT(hop_length > 0 && win_length >= hop_length && frame_samples == config.frame_samples &&
                        config.decode_lookahead_samples < config.frame_samples,
                    "LFM3-Omni audio_tokenizer/config.json is inconsistent: stft_frame_stack ", frame_stack,
                    ", stft_hop_length ", hop_length, ", stft_win_length ", win_length, ", frame_samples ",
                    frame_samples);
    return config;
}

LFM3OmniSpeechEngine::SubModels LFM3OmniSpeechEngine::read_submodels(const std::filesystem::path& models_dir) {
    auto read = [&models_dir](const std::string& name) {
        return utils::singleton_core().read_model(models_dir / ("openvino_" + name + "_model.xml"));
    };
    return {read("depth_decoder"), read("audio_out_embeddings"), read("audio_decoder")};
}

LFM3OmniSpeechEngine::SubModels LFM3OmniSpeechEngine::read_submodels(const ModelsMap& models_map) {
    auto read = [&models_map](const std::string& key) {
        const auto& [model, weights] = utils::get_model_weights_pair(models_map, key);
        return utils::singleton_core().read_model(model, weights);
    };
    return {read("depth_decoder"), read("audio_out_embeddings"), read("audio_decoder")};
}

LFM3OmniSpeechEngine::LFM3OmniSpeechEngine(const std::filesystem::path& models_dir,
                                           ov::InferRequest& language,
                                           EmbeddingsModel::Ptr embedding,
                                           const std::string& device,
                                           const ov::AnyMap& properties)
    : LFM3OmniSpeechEngine(read_model_config(models_dir),
                           read_submodels(models_dir),
                           language,
                           std::move(embedding),
                           device,
                           properties) {}

LFM3OmniSpeechEngine::LFM3OmniSpeechEngine(const ModelsMap& models_map,
                                           const std::filesystem::path& config_dir_path,
                                           ov::InferRequest& language,
                                           EmbeddingsModel::Ptr embedding,
                                           const std::string& device,
                                           const ov::AnyMap& properties)
    : LFM3OmniSpeechEngine(read_model_config(config_dir_path),
                           read_submodels(models_map),
                           language,
                           std::move(embedding),
                           device,
                           properties) {}

LFM3OmniSpeechEngine::LFM3OmniSpeechEngine(ModelConfig config,
                                           const SubModels& submodels,
                                           ov::InferRequest& language,
                                           EmbeddingsModel::Ptr embedding,
                                           const std::string& device,
                                           const ov::AnyMap& properties)
    : m_config(std::move(config)),
      m_language(language),
      m_embedding(std::move(embedding)),
      m_depth_decoder(compile(submodels.depth_decoder, "depth_decoder", device, properties)),
      m_audio_out_embeddings(compile(submodels.audio_out_embeddings, "audio_out_embeddings", device, properties)),
      m_audio_decoder(compile(submodels.audio_decoder, "audio_decoder", device, properties, AUDIO_DECODER_DEFAULTS)) {
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

ov::Tensor LFM3OmniSpeechEngine::embed_token(int64_t token_id) {
    ov::Tensor input_ids(ov::element::i64, {1, 1});
    input_ids.data<int64_t>()[0] = token_id;
    CircularBufferQueueElementGuard<EmbeddingsRequest> guard(m_embedding->get_request_queue().get());
    const ov::Tensor embeds = m_embedding->infer(guard.get(), input_ids);
    return copy_tensor(embeds, embeds.get_shape());
}

ov::Tensor LFM3OmniSpeechEngine::embed_frame(const std::vector<int64_t>& codes) {
    ov::Tensor codes_tensor(ov::element::i64, {1, codes.size()});
    std::copy(codes.begin(), codes.end(), codes_tensor.data<int64_t>());
    m_audio_out_embeddings.set_tensor("codes", codes_tensor);
    m_audio_out_embeddings.infer();
    const ov::Tensor embeds = m_audio_out_embeddings.get_tensor("inputs_embeds");
    return copy_tensor(embeds, {1, 1, embeds.get_shape().at(1)});
}

// Not the continuous batching Sampler transforms: those keep the penalty history in a map shared across steps, while
// LFM3-Omni penalizes only the last text_repetition_window text tokens and audio_repetition_window frames of each
// codebook, with the transformers processor order below.
int64_t LFM3OmniSpeechEngine::sample(std::vector<float>& scores,
                                     const Sampling& sampling,
                                     const std::vector<int64_t>& history,
                                     const std::vector<int64_t>& suppress,
                                     std::mt19937& rng) {
    // Same order as the transformers processors: penalty, suppression, then temperature, top-k, top-p and min-p.
    if (sampling.repetition_penalty != 1.0f && !history.empty()) {
        std::vector<int64_t> unique_ids(history);
        std::sort(unique_ids.begin(), unique_ids.end());
        unique_ids.erase(std::unique(unique_ids.begin(), unique_ids.end()), unique_ids.end());
        for (int64_t id : unique_ids) {
            float& score = scores.at(id);
            score = score < 0.0f ? score * sampling.repetition_penalty : score / sampling.repetition_penalty;
        }
    }
    for (int64_t id : suppress) {
        scores.at(id) = NEG_INF;
    }
    const bool greedy = sampling.temperature ? *sampling.temperature == 0.0f : !sampling.top_k.has_value();
    if (greedy) {
        return std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
    }
    if (sampling.temperature) {
        for (float& score : scores) {
            score /= *sampling.temperature;
        }
    }
    if (sampling.top_k && *sampling.top_k < scores.size()) {
        std::vector<float> sorted(scores);
        std::nth_element(sorted.begin(), sorted.begin() + *sampling.top_k - 1, sorted.end(), std::greater<float>());
        const float threshold = sorted[*sampling.top_k - 1];
        for (float& score : scores) {
            if (score < threshold) {
                score = NEG_INF;
            }
        }
    }
    if (sampling.top_p < 1.0f) {
        std::vector<float> probs(scores);
        softmax(probs);
        std::vector<size_t> order(scores.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&scores](size_t a, size_t b) {
            return scores[a] < scores[b];
        });
        // Drop the least likely tokens while their cumulative probability is at most 1 - top_p; keep the top one.
        double cumulative = 0.0;
        for (size_t rank = 0; rank + 1 < order.size(); ++rank) {
            cumulative += probs[order[rank]];
            if (cumulative > 1.0 - sampling.top_p) {
                break;
            }
            scores[order[rank]] = NEG_INF;
        }
    }
    if (sampling.min_p > 0.0f) {
        std::vector<float> probs(scores);
        softmax(probs);
        const float threshold = sampling.min_p * *std::max_element(probs.begin(), probs.end());
        for (size_t id = 0; id < scores.size(); ++id) {
            if (probs[id] < threshold) {
                scores[id] = NEG_INF;
            }
        }
    }
    softmax(scores);
    std::discrete_distribution<size_t> distribution(scores.begin(), scores.end());
    return static_cast<int64_t>(distribution(rng));
}

std::vector<int64_t> LFM3OmniSpeechEngine::sample_frame(const ov::Tensor& hidden_state,
                                                        const Sampling& sampling,
                                                        const std::vector<std::vector<int64_t>>& history) {
    const size_t num_codebooks = m_config.num_codebooks;
    std::vector<int64_t> codes(num_codebooks, 0);
    ov::Tensor codes_tensor(ov::element::i64, {1, num_codebooks});
    ov::Tensor masked(ov::element::boolean, {1, num_codebooks});
    std::fill_n(masked.data<bool>(), num_codebooks, true);
    // The codebooks are revealed in a random order, a few per depth decoder pass.
    std::vector<size_t> order(num_codebooks);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), m_audio_rng);

    size_t revealed = 0;
    for (size_t pass_size : m_config.maskgit_schedule) {
        std::copy(codes.begin(), codes.end(), codes_tensor.data<int64_t>());
        m_depth_decoder.set_tensor("hidden_states", hidden_state);
        m_depth_decoder.set_tensor("codes", codes_tensor);
        m_depth_decoder.set_tensor("masked", masked);
        m_depth_decoder.infer();
        const ov::Tensor logits = m_depth_decoder.get_tensor("logits");
        const size_t codebook_size = m_config.codebook_size;
        OPENVINO_ASSERT(logits.get_shape() == ov::Shape({1, num_codebooks, codebook_size}),
                        "LFM3-Omni depth decoder must return logits [1, ", num_codebooks, ", ", codebook_size, "]");
        for (size_t rank = revealed; rank < revealed + pass_size; ++rank) {
            const size_t codebook = order[rank];
            const float* row = logits.data<const float>() + codebook * codebook_size;
            std::vector<float> scores(row, row + codebook_size);
            codes[codebook] = sample(scores, sampling, history[codebook], {}, m_audio_rng);
        }
        for (size_t rank = revealed; rank < revealed + pass_size; ++rank) {
            masked.data<bool>()[order[rank]] = false;
        }
        revealed += pass_size;
    }
    return codes;
}

ov::Tensor LFM3OmniSpeechEngine::decode_frames(const std::vector<std::vector<int64_t>>& frames,
                                               size_t begin,
                                               size_t end) {
    const size_t num_frames = end - begin;
    ov::Tensor audio_codes(ov::element::i64, {1, m_config.num_codebooks, num_frames});
    int64_t* codes = audio_codes.data<int64_t>();
    for (size_t frame = 0; frame < num_frames; ++frame) {
        for (size_t codebook = 0; codebook < m_config.num_codebooks; ++codebook) {
            codes[codebook * num_frames + frame] = frames[begin + frame][codebook];
        }
    }
    m_audio_decoder.set_tensor("audio_codes", audio_codes);
    m_audio_decoder.infer();
    const ov::Tensor audio = m_audio_decoder.get_tensor("audio_values");
    OPENVINO_ASSERT(audio.get_shape() == ov::Shape({1, num_frames * m_config.frame_samples}),
                    "LFM3-Omni audio decoder must return ", m_config.frame_samples, " samples per frame");
    return copy_tensor(audio, audio.get_shape());
}

LFM3SpeechResult LFM3OmniSpeechEngine::generate(const ov::Tensor& inputs_embeds,
                                                const ov::Tensor& attention_mask,
                                                int64_t last_prompt_token,
                                                const LFM3SpeechRequest& request,
                                                const std::shared_ptr<StreamerBase>& streamer,
                                                const OmniSpeechStreamerVariant& speech_streamer) {
    const GenerationConfig& text_config = request.text_config;
    OPENVINO_ASSERT(is_speech_streamer_active(speech_streamer),
                    "LFM3-Omni speech output needs a speech streamer: it is the only output of the waveform");
    // Created before any step that can throw: end() is due even if generation fails.
    SpeechStreamerGuard speech_guard(speech_streamer, true);
    m_text_rng.seed(static_cast<std::mt19937::result_type>(text_config.rng_seed));
    m_audio_rng.seed(static_cast<std::mt19937::result_type>(request.audio_rng_seed));

    Sampling text_sampling;
    if (text_config.do_sample) {
        text_sampling.temperature = text_config.temperature;
        // As in the continuous batching sampler, 0 and the max value both mean no top-k filter.
        if (text_config.top_k > 0 && text_config.top_k != std::numeric_limits<size_t>::max()) {
            text_sampling.top_k = text_config.top_k;
        }
        text_sampling.top_p = text_config.top_p;
        text_sampling.min_p = text_config.min_p;
    }
    text_sampling.repetition_penalty = text_config.repetition_penalty;
    Sampling audio_sampling;
    audio_sampling.temperature = request.audio_temperature ? request.audio_temperature : m_config.audio_temperature;
    audio_sampling.top_k = request.audio_top_k ? request.audio_top_k : m_config.audio_top_k;
    audio_sampling.repetition_penalty = request.audio_repetition_penalty.value_or(m_config.audio_repetition_penalty);

    std::set<int64_t> stop_token_ids = text_config.stop_token_ids;
    if (text_config.eos_token_id >= 0) {
        stop_token_ids.insert(text_config.eos_token_id);
    }

    enum class Mode { TEXT, SPEECH, INTERLEAVED };
    Mode mode = request.mode == SpeechOutputMode::INTERLEAVED ? Mode::INTERLEAVED : Mode::SPEECH;
    LFM3SpeechResult result;

    // Speech always follows its marker in training: append it unless the prompt already ends with one.
    ov::Tensor prompt_embeds = inputs_embeds;
    const bool has_marker = mode == Mode::INTERLEAVED
                                ? last_prompt_token == m_config.mixed_start_token_id
                                : last_prompt_token == m_config.audio_start_token_id ||
                                      last_prompt_token == m_config.audio_out_token_id;
    if (!has_marker) {
        const int64_t marker = mode == Mode::INTERLEAVED ? m_config.mixed_start_token_id : m_config.audio_start_token_id;
        const ov::Tensor marker_embeds = embed_token(marker);
        const ov::Shape& shape = inputs_embeds.get_shape();
        prompt_embeds = ov::Tensor(inputs_embeds.get_element_type(), {1, shape[1] + 1, shape[2]});
        std::memcpy(prompt_embeds.data(), inputs_embeds.data(), inputs_embeds.get_byte_size());
        std::memcpy(static_cast<uint8_t*>(prompt_embeds.data()) + inputs_embeds.get_byte_size(),
                    marker_embeds.data(),
                    marker_embeds.get_byte_size());
        result.slot_tokens.push_back(marker);
    }
    size_t attention_length = attention_mask.get_shape().at(1) + (has_marker ? 0 : 1);
    // As in the model's own generate, max_new_tokens counts every slot: text tokens and speech frames.
    const size_t max_new_slots = text_config.get_max_new_tokens(attention_length);
    size_t generated_slots = 0;

    RawPerfMetrics& raw_perf = result.raw_perf_metrics;
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

    std::vector<std::vector<int64_t>> frames;
    size_t streamed_samples = 0;
    // Streams the samples that frames [0, end) fix for good. The codec decoder is causal, but the ISTFT overlap
    // changes the last decode_lookahead_samples until the next frame arrives. Each chunk is decoded again with
    // STREAM_CONTEXT_FRAMES frames of left context, which covers the decoder's receptive field.
    auto stream_up_to = [&](size_t end, bool final) {
        const size_t limit = end * m_config.frame_samples - (final ? 0 : m_config.decode_lookahead_samples);
        if (limit <= streamed_samples) {
            return StreamingStatus::RUNNING;
        }
        const size_t first = streamed_samples / m_config.frame_samples;
        const size_t begin = first > STREAM_CONTEXT_FRAMES ? first - STREAM_CONTEXT_FRAMES : 0;
        const ov::Tensor audio = decode_frames(frames, begin, end);
        const float* new_samples = audio.data<const float>() + (streamed_samples - begin * m_config.frame_samples);
        ov::Tensor chunk(ov::element::f32, {1, 1, limit - streamed_samples});
        std::copy_n(new_samples, chunk.get_size(), chunk.data<float>());
        streamed_samples = limit;
        return invoke_speech_streamer(speech_streamer, chunk);
    };
    std::deque<int64_t> text_history;
    std::deque<std::vector<int64_t>> audio_history;
    size_t slots_left = m_config.interleaved_n_text;
    bool speaking = false;
    bool text_done = false;
    bool finished = false;
    while (!finished) {
        const bool interleaving = mode == Mode::INTERLEAVED;
        ov::Tensor next_embeds;
        if (mode == Mode::SPEECH || (interleaving && speaking)) {
            std::vector<std::vector<int64_t>> codebook_history(m_config.num_codebooks);
            for (const auto& frame : audio_history) {
                for (size_t codebook = 0; codebook < m_config.num_codebooks; ++codebook) {
                    codebook_history[codebook].push_back(frame[codebook]);
                }
            }
            std::vector<int64_t> codes = sample_frame(step.hidden_state, audio_sampling, codebook_history);
            const bool is_eos = std::find(codes.begin(), codes.end(), m_config.audio_eos_code) != codes.end();
            if (is_eos) {
                std::fill(codes.begin(), codes.end(), m_config.audio_eos_code);
                mode = Mode::TEXT;
            } else {
                audio_history.push_back(codes);
                if (audio_history.size() > m_config.audio_repetition_window) {
                    audio_history.pop_front();
                }
                frames.push_back(codes);
                if (frames.size() % request.audio_chunk_frames == 0) {
                    const StreamingStatus status = stream_up_to(frames.size(), false);
                    if (status != StreamingStatus::RUNNING) {
                        result.status = status == StreamingStatus::STOP ? GenerationStatus::STOP : GenerationStatus::CANCEL;
                        finished = true;
                    }
                }
            }
            result.slot_tokens.push_back(m_config.audio_out_token_id);
            next_embeds = embed_frame(codes);
            if (!finished && frames.size() >= request.max_audio_frames) {
                result.finish_reason = GenerationFinishReason::LENGTH;
                finished = true;
            }
        } else {
            const std::vector<int64_t> history(text_history.begin(), text_history.end());
            const int64_t token =
                sample(step.logits, text_sampling, history, m_config.placeholder_token_ids, m_text_rng);
            text_history.push_back(token);
            if (text_history.size() > m_config.text_repetition_window) {
                text_history.pop_front();
            }
            result.text_tokens.push_back(token);
            result.slot_tokens.push_back(token);
            if (streamer) {
                const StreamingStatus status = streamer->write(token);
                if (status == StreamingStatus::STOP) {
                    result.status = GenerationStatus::STOP;
                    finished = true;
                } else if (status == StreamingStatus::CANCEL) {
                    result.status = GenerationStatus::CANCEL;
                    finished = true;
                }
            }
            if (token == m_config.text_end_token_id) {
                // Inside an interleaved block it ends the text and starts the speech tail; otherwise it ends the turn.
                if (interleaving) {
                    text_done = true;
                    speaking = true;
                } else {
                    finished = true;
                }
            } else if (stop_token_ids.count(token)) {
                finished = true;
            } else if (!interleaving && token == m_config.audio_start_token_id) {
                mode = Mode::SPEECH;
            } else if (!interleaving && token == m_config.mixed_start_token_id) {
                mode = Mode::INTERLEAVED;
                slots_left = m_config.interleaved_n_text;
                speaking = false;
                text_done = false;
            }
            next_embeds = embed_token(token);
        }
        if (!finished && ++generated_slots >= max_new_slots) {
            result.finish_reason = GenerationFinishReason::LENGTH;
            finished = true;
        }
        raw_perf.m_new_token_times.emplace_back(std::chrono::steady_clock::now());
        raw_perf.m_batch_sizes.emplace_back(1);
        if (interleaving && !text_done && --slots_left == 0) {
            speaking = !speaking;
            slots_left = speaking ? m_config.interleaved_n_audio : m_config.interleaved_n_text;
        }
        if (!finished) {
            // The last sampled slot needs no hidden state, so its decode step is skipped.
            step = timed_step(next_embeds, ++attention_length);
        }
    }
    if (streamer) {
        streamer->end();
    }
    if (result.status == GenerationStatus::RUNNING) {
        result.status = GenerationStatus::FINISHED;
    }
    // STOP keeps what was generated, so the held-back samples are flushed; CANCEL drops them.
    if (result.status != GenerationStatus::CANCEL && !frames.empty()) {
        // With fewer frames than audio_chunk_frames the flush is the only callback, so its status counts too.
        const StreamingStatus status = stream_up_to(frames.size(), true);
        if (status == StreamingStatus::STOP) {
            result.status = GenerationStatus::STOP;
        } else if (status == StreamingStatus::CANCEL) {
            result.status = GenerationStatus::CANCEL;
        }
    }
    speech_guard.end();
    return result;
}

}  // namespace ov::genai
