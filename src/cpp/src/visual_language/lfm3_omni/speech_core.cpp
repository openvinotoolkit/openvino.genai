// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/lfm3_omni/speech_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>

#include "json_utils.hpp"
#include "omni/speech_streamer_utils.hpp"

namespace ov::genai {

namespace {

// One request per submodel: speech is decoded on one thread, one step at a time.
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

// Dynamic activation quantization makes the codec output depend on how many frames are decoded together, so a
// re-decoded streaming chunk would not match the one-shot waveform.
const ov::AnyMap AUDIO_DECODER_DEFAULTS{ov::hint::dynamic_quantization_group_size(0)};

}  // namespace

void LFM3SpeechRequest::validate(const OmniTalkerSpeechConfig& speech_config) {
    const auto* speaker = std::get_if<std::string>(&speech_config.speaker);
    OPENVINO_ASSERT(speaker && speaker->empty(),
                    "LFM3-Omni has no speaker selection; describe the voice in the system prompt");
    OPENVINO_ASSERT(
        !speech_config.talker_temperature && !speech_config.talker_top_k && !speech_config.talker_repetition_penalty,
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
      audio_chunk_left_context_frames(speech_config.audio_chunk_left_context_frames),
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

LFM3SpeechModelConfig LFM3SpeechModelConfig::read(const std::filesystem::path& config_dir_path) {
    using utils::read_json_param;
    LFM3SpeechModelConfig config;
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
    OPENVINO_ASSERT(
        config.audio_start_token_id >= 0 && config.mixed_start_token_id >= 0 && config.text_end_token_id >= 0,
        "LFM3-Omni config.json must define audio_start_token_id, mixed_start_token_id and "
        "text_end_token_id");
    // Each pass is checked against the codebooks still hidden, so the sum cannot wrap around.
    size_t revealed = 0;
    for (size_t pass_size : config.maskgit_schedule) {
        OPENVINO_ASSERT(pass_size > 0 && pass_size <= config.num_codebooks - revealed,
                        "LFM3-Omni maskgit_schedule must reveal each of the ",
                        config.num_codebooks,
                        " codebooks once, in passes of at least 1");
        revealed += pass_size;
    }
    OPENVINO_ASSERT(revealed == config.num_codebooks,
                    "LFM3-Omni maskgit_schedule must reveal each of the ",
                    config.num_codebooks,
                    " codebooks once");
    OPENVINO_ASSERT(config.interleaved_n_text > 0 && config.interleaved_n_audio > 0,
                    "LFM3-Omni interleaved block sizes must be positive");
    OPENVINO_ASSERT(config.num_codebooks > 0 && config.audio_eos_code >= 0 &&
                        static_cast<size_t>(config.audio_eos_code) < config.codebook_size,
                    "LFM3-Omni config.json needs num_codebooks > 0 and audio_eos_code < codebook_size, got ",
                    config.num_codebooks,
                    ", ",
                    config.audio_eos_code,
                    ", ",
                    config.codebook_size);

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
                    "LFM3-Omni audio_tokenizer/config.json is inconsistent: stft_frame_stack ",
                    frame_stack,
                    ", stft_hop_length ",
                    hop_length,
                    ", stft_win_length ",
                    win_length,
                    ", frame_samples ",
                    frame_samples);
    return config;
}

// Not the continuous batching Sampler transforms: those keep the penalty history in a map shared across steps, while
// LFM3-Omni penalizes only the last text_repetition_window text tokens and audio_repetition_window frames of each
// codebook, with the transformers processor order below.
int64_t LFM3SpeechSampling::sample(std::vector<float>& scores,
                                   const std::vector<int64_t>& history,
                                   const std::vector<int64_t>& suppress,
                                   std::mt19937& rng) const {
    if (repetition_penalty != 1.0f && !history.empty()) {
        std::vector<int64_t> unique_ids(history);
        std::sort(unique_ids.begin(), unique_ids.end());
        unique_ids.erase(std::unique(unique_ids.begin(), unique_ids.end()), unique_ids.end());
        for (int64_t id : unique_ids) {
            float& score = scores.at(id);
            score = score < 0.0f ? score * repetition_penalty : score / repetition_penalty;
        }
    }
    for (int64_t id : suppress) {
        scores.at(id) = NEG_INF;
    }
    const bool greedy = temperature ? *temperature == 0.0f : !top_k.has_value();
    if (greedy) {
        return std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
    }
    if (temperature) {
        for (float& score : scores) {
            score /= *temperature;
        }
    }
    if (top_k && *top_k < scores.size()) {
        std::vector<float> sorted(scores);
        std::nth_element(sorted.begin(), sorted.begin() + *top_k - 1, sorted.end(), std::greater<float>());
        const float threshold = sorted[*top_k - 1];
        for (float& score : scores) {
            if (score < threshold) {
                score = NEG_INF;
            }
        }
    }
    if (top_p < 1.0f) {
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
            if (cumulative > 1.0 - top_p) {
                break;
            }
            scores[order[rank]] = NEG_INF;
        }
    }
    if (min_p > 0.0f) {
        std::vector<float> probs(scores);
        softmax(probs);
        const float threshold = min_p * *std::max_element(probs.begin(), probs.end());
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

LFM3SpeechModels::LFM3SpeechModels(const std::filesystem::path& models_dir,
                                   const std::string& device,
                                   const ov::AnyMap& properties)
    : LFM3SpeechModels(
          LFM3SpeechModelConfig::read(models_dir),
          [&models_dir] {
              auto read = [&models_dir](const std::string& name) {
                  return utils::singleton_core().read_model(models_dir / ("openvino_" + name + "_model.xml"));
              };
              return SubModels{read("depth_decoder"), read("audio_out_embeddings"), read("audio_decoder")};
          }(),
          device,
          properties) {}

LFM3SpeechModels::LFM3SpeechModels(const ModelsMap& models_map,
                                   const std::filesystem::path& config_dir_path,
                                   const std::string& device,
                                   const ov::AnyMap& properties)
    : LFM3SpeechModels(
          LFM3SpeechModelConfig::read(config_dir_path),
          [&models_map] {
              auto read = [&models_map](const std::string& key) {
                  const auto& [model, weights] = utils::get_model_weights_pair(models_map, key);
                  return utils::singleton_core().read_model(model, weights);
              };
              return SubModels{read("depth_decoder"), read("audio_out_embeddings"), read("audio_decoder")};
          }(),
          device,
          properties) {}

LFM3SpeechModels::LFM3SpeechModels(LFM3SpeechModelConfig config,
                                   const SubModels& submodels,
                                   const std::string& device,
                                   const ov::AnyMap& properties)
    : m_config(std::move(config)),
      m_depth_decoder(compile(submodels.depth_decoder, "depth_decoder", device, properties)),
      m_audio_out_embeddings(compile(submodels.audio_out_embeddings, "audio_out_embeddings", device, properties)),
      m_audio_decoder(compile(submodels.audio_decoder, "audio_decoder", device, properties, AUDIO_DECODER_DEFAULTS)) {}

std::vector<std::vector<int64_t>> LFM3SpeechModels::sample_frames(const ov::Tensor& hidden_states,
                                                                  const std::vector<LFM3SpeechState*>& states) {
    const size_t batch = states.size();
    const size_t num_codebooks = m_config.num_codebooks;
    const size_t codebook_size = m_config.codebook_size;
    OPENVINO_ASSERT(hidden_states.get_shape().size() == 2 && hidden_states.get_shape()[0] == batch,
                    "LFM3-Omni depth decoder needs one hidden state row per sequence");
    std::vector<std::vector<int64_t>> codes(batch, std::vector<int64_t>(num_codebooks, 0));
    ov::Tensor codes_tensor(ov::element::i64, {batch, num_codebooks});
    ov::Tensor masked(ov::element::boolean, {batch, num_codebooks});
    std::fill_n(masked.data<bool>(), masked.get_size(), true);
    // The codebooks of each frame are revealed in a random order, a few per depth decoder pass.
    std::vector<std::vector<size_t>> orders(batch, std::vector<size_t>(num_codebooks));
    std::vector<std::vector<std::vector<int64_t>>> histories(batch);
    for (size_t row = 0; row < batch; ++row) {
        std::iota(orders[row].begin(), orders[row].end(), 0);
        std::shuffle(orders[row].begin(), orders[row].end(), states[row]->audio_rng());
        histories[row] = states[row]->codebook_history();
    }

    size_t revealed = 0;
    for (size_t pass_size : m_config.maskgit_schedule) {
        for (size_t row = 0; row < batch; ++row) {
            std::copy(codes[row].begin(), codes[row].end(), codes_tensor.data<int64_t>() + row * num_codebooks);
        }
        m_depth_decoder.set_tensor("hidden_states", hidden_states);
        m_depth_decoder.set_tensor("codes", codes_tensor);
        m_depth_decoder.set_tensor("masked", masked);
        m_depth_decoder.infer();
        const ov::Tensor logits = m_depth_decoder.get_tensor("logits");
        OPENVINO_ASSERT(logits.get_shape() == ov::Shape({batch, num_codebooks, codebook_size}),
                        "LFM3-Omni depth decoder must return logits [",
                        batch,
                        ", ",
                        num_codebooks,
                        ", ",
                        codebook_size,
                        "]");
        for (size_t row = 0; row < batch; ++row) {
            for (size_t rank = revealed; rank < revealed + pass_size; ++rank) {
                const size_t codebook = orders[row][rank];
                const float* scores_row = logits.data<const float>() + (row * num_codebooks + codebook) * codebook_size;
                std::vector<float> scores(scores_row, scores_row + codebook_size);
                codes[row][codebook] = states[row]->audio_sampling().sample(scores,
                                                                            histories[row][codebook],
                                                                            {},
                                                                            states[row]->audio_rng());
            }
            for (size_t rank = revealed; rank < revealed + pass_size; ++rank) {
                masked.data<bool>()[row * num_codebooks + orders[row][rank]] = false;
            }
        }
        revealed += pass_size;
    }
    return codes;
}

ov::Tensor LFM3SpeechModels::embed_frames(const std::vector<std::vector<int64_t>>& frames) {
    const size_t num_codebooks = m_config.num_codebooks;
    ov::Tensor codes_tensor(ov::element::i64, {frames.size(), num_codebooks});
    for (size_t row = 0; row < frames.size(); ++row) {
        std::copy(frames[row].begin(), frames[row].end(), codes_tensor.data<int64_t>() + row * num_codebooks);
    }
    m_audio_out_embeddings.set_tensor("codes", codes_tensor);
    m_audio_out_embeddings.infer();
    const ov::Tensor embeds = m_audio_out_embeddings.get_tensor("inputs_embeds");
    return copy_tensor(embeds, embeds.get_shape());
}

ov::Tensor LFM3SpeechModels::decode_frames(const std::vector<std::vector<int64_t>>& frames, size_t begin, size_t end) {
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
                    "LFM3-Omni audio decoder must return ",
                    m_config.frame_samples,
                    " samples per frame");
    return copy_tensor(audio, audio.get_shape());
}

LFM3SpeechState::LFM3SpeechState(const LFM3SpeechModelConfig& config,
                                 const LFM3SpeechRequest& request,
                                 const OmniSpeechStreamerVariant& speech_streamer)
    : m_config(config),
      m_speech_streamer(speech_streamer),
      // Without a speech streamer nobody needs the chunks early, so every frame is decoded once at the end.
      m_stream_chunks(is_speech_streamer_active(speech_streamer)),
      m_max_audio_frames(request.max_audio_frames),
      m_audio_chunk_frames(request.audio_chunk_frames),
      m_left_context_frames(request.audio_chunk_left_context_frames),
      m_text_config(request.text_config),
      m_text_rng(static_cast<std::mt19937::result_type>(request.text_config.rng_seed)),
      m_audio_rng(static_cast<std::mt19937::result_type>(request.audio_rng_seed)),
      m_mode(request.mode == SpeechOutputMode::INTERLEAVED ? Mode::INTERLEAVED : Mode::SPEECH),
      m_slots_left(config.interleaved_n_text) {
    if (m_text_config.do_sample) {
        m_text_sampling.temperature = m_text_config.temperature;
        // As in the continuous batching sampler, 0 and the max value both mean no top-k filter.
        if (m_text_config.top_k > 0 && m_text_config.top_k != std::numeric_limits<size_t>::max()) {
            m_text_sampling.top_k = m_text_config.top_k;
        }
        m_text_sampling.top_p = m_text_config.top_p;
        m_text_sampling.min_p = m_text_config.min_p;
    }
    m_text_sampling.repetition_penalty = m_text_config.repetition_penalty;
    m_audio_sampling.temperature = request.audio_temperature ? request.audio_temperature : config.audio_temperature;
    m_audio_sampling.top_k = request.audio_top_k ? request.audio_top_k : config.audio_top_k;
    m_audio_sampling.repetition_penalty = request.audio_repetition_penalty.value_or(config.audio_repetition_penalty);
    m_stop_token_ids = m_text_config.stop_token_ids;
    if (m_text_config.eos_token_id >= 0) {
        m_stop_token_ids.insert(m_text_config.eos_token_id);
    }
}

int64_t LFM3SpeechState::take_prompt_marker(int64_t last_prompt_token) {
    const bool interleaved = m_mode == Mode::INTERLEAVED;
    const bool has_marker = interleaved ? last_prompt_token == m_config.mixed_start_token_id
                                        : last_prompt_token == m_config.audio_start_token_id ||
                                              last_prompt_token == m_config.audio_out_token_id;
    if (has_marker) {
        return -1;
    }
    const int64_t marker = interleaved ? m_config.mixed_start_token_id : m_config.audio_start_token_id;
    m_result.slot_tokens.push_back(marker);
    return marker;
}

void LFM3SpeechState::start(size_t context_length) {
    // As in the model's own generate, max_new_tokens counts every slot: text tokens and speech frames.
    m_max_new_slots = m_text_config.get_max_new_tokens(context_length);
}

bool LFM3SpeechState::next_is_frame() const {
    return m_mode == Mode::SPEECH || (m_mode == Mode::INTERLEAVED && m_speaking);
}

std::vector<std::vector<int64_t>> LFM3SpeechState::codebook_history() const {
    std::vector<std::vector<int64_t>> history(m_config.num_codebooks);
    for (const auto& frame : m_audio_history) {
        for (size_t codebook = 0; codebook < m_config.num_codebooks; ++codebook) {
            history[codebook].push_back(frame[codebook]);
        }
    }
    return history;
}

int64_t LFM3SpeechState::sample_text(std::vector<float>& logits) {
    const std::vector<int64_t> history(m_text_history.begin(), m_text_history.end());
    return m_text_sampling.sample(logits, history, m_config.placeholder_token_ids, m_text_rng);
}

void LFM3SpeechState::accept_text(int64_t token, StreamingStatus stream_status) {
    const bool interleaving = m_mode == Mode::INTERLEAVED;
    m_text_history.push_back(token);
    if (m_text_history.size() > m_config.text_repetition_window) {
        m_text_history.pop_front();
    }
    m_result.text_tokens.push_back(token);
    m_result.slot_tokens.push_back(token);
    if (stream_status == StreamingStatus::STOP) {
        m_result.status = GenerationStatus::STOP;
        m_finished = true;
    } else if (stream_status == StreamingStatus::CANCEL) {
        m_result.status = GenerationStatus::CANCEL;
        m_finished = true;
    } else if (stream_status == StreamingStatus::TOOL_CALL_STOP) {
        // As in the text pipelines: a tool call parser ends the turn with the TOOL_CALL reason.
        m_result.status = GenerationStatus::STOP;
        m_result.finish_reason = GenerationFinishReason::TOOL_CALL;
        m_finished = true;
    }
    if (token == m_config.text_end_token_id) {
        // Inside an interleaved block it ends the text and starts the speech tail; otherwise it ends the turn.
        if (interleaving) {
            m_text_done = true;
            m_speaking = true;
        } else {
            m_finished = true;
        }
    } else if (m_stop_token_ids.count(token)) {
        m_finished = true;
    } else if (!interleaving && token == m_config.audio_start_token_id) {
        m_mode = Mode::SPEECH;
    } else if (!interleaving && token == m_config.mixed_start_token_id) {
        m_mode = Mode::INTERLEAVED;
        m_slots_left = m_config.interleaved_n_text;
        m_speaking = false;
        m_text_done = false;
    }
    end_slot(interleaving);
}

std::vector<int64_t> LFM3SpeechState::accept_frame(std::vector<int64_t> codes, LFM3SpeechModels& models) {
    const bool interleaving = m_mode == Mode::INTERLEAVED;
    const bool is_eos = std::find(codes.begin(), codes.end(), m_config.audio_eos_code) != codes.end();
    if (is_eos) {
        std::fill(codes.begin(), codes.end(), m_config.audio_eos_code);
        m_mode = Mode::TEXT;
    } else {
        m_audio_history.push_back(codes);
        if (m_audio_history.size() > m_config.audio_repetition_window) {
            m_audio_history.pop_front();
        }
        m_frames.push_back(codes);
        if (m_stream_chunks && m_frames.size() % m_audio_chunk_frames == 0) {
            const StreamingStatus status = stream_up_to(m_frames.size(), false, models);
            if (status != StreamingStatus::RUNNING) {
                m_result.status = status == StreamingStatus::STOP ? GenerationStatus::STOP : GenerationStatus::CANCEL;
                m_speech_streamer_stopped = true;
                m_finished = true;
            }
        }
    }
    m_result.slot_tokens.push_back(m_config.audio_out_token_id);
    if (!m_finished && m_frames.size() >= m_max_audio_frames) {
        m_result.finish_reason = GenerationFinishReason::LENGTH;
        m_finished = true;
    }
    end_slot(interleaving);
    return codes;
}

void LFM3SpeechState::end_with(GenerationStatus status) {
    m_result.status = status;
    m_finished = true;
}

void LFM3SpeechState::end_slot(bool interleaving) {
    if (!m_finished && ++m_generated_slots >= m_max_new_slots) {
        m_result.finish_reason = GenerationFinishReason::LENGTH;
        m_finished = true;
    }
    if (interleaving && !m_text_done && --m_slots_left == 0) {
        m_speaking = !m_speaking;
        m_slots_left = m_speaking ? m_config.interleaved_n_audio : m_config.interleaved_n_text;
    }
}

// Streams the samples that frames [0, end) fix for good. The codec decoder is causal, but the ISTFT overlap changes the
// last decode_lookahead_samples until the next frame arrives. Each chunk is decoded again with
// audio_chunk_left_context_frames frames of left context. The codec decoder reaches about 20 frames back: three
// sliding-window attention layers of 30 steps and five convolutions of 7 taps, at 6 steps per frame.
StreamingStatus LFM3SpeechState::stream_up_to(size_t end, bool final, LFM3SpeechModels& models) {
    const size_t limit = end * m_config.frame_samples - (final ? 0 : m_config.decode_lookahead_samples);
    if (limit <= m_streamed_samples) {
        return StreamingStatus::RUNNING;
    }
    const size_t first = m_streamed_samples / m_config.frame_samples;
    const size_t begin = first > m_left_context_frames ? first - m_left_context_frames : 0;
    const ov::Tensor audio = models.decode_frames(m_frames, begin, end);
    const float* new_samples = audio.data<const float>() + (m_streamed_samples - begin * m_config.frame_samples);
    const size_t count = limit - m_streamed_samples;
    m_samples.insert(m_samples.end(), new_samples, new_samples + count);
    m_streamed_samples = limit;
    if (!m_stream_chunks) {
        return StreamingStatus::RUNNING;
    }
    ov::Tensor chunk(ov::element::f32, {1, 1, count});
    std::copy_n(new_samples, count, chunk.data<float>());
    return invoke_speech_streamer(m_speech_streamer, chunk);
}

LFM3SpeechResult LFM3SpeechState::finish(LFM3SpeechModels& models) {
    if (m_result.status == GenerationStatus::RUNNING) {
        m_result.status = GenerationStatus::FINISHED;
    }
    // A speech streamer that returned STOP or CANCEL gets no further write(). CANCEL from the text streamer drops
    // the held-back samples too; its STOP keeps what was generated, so they are flushed.
    if (!m_speech_streamer_stopped && m_result.status != GenerationStatus::CANCEL && !m_frames.empty()) {
        // With fewer frames than audio_chunk_frames the flush is the only callback, so its status counts too.
        const StreamingStatus status = stream_up_to(m_frames.size(), true, models);
        if (status == StreamingStatus::STOP) {
            m_result.status = GenerationStatus::STOP;
        } else if (status == StreamingStatus::CANCEL) {
            m_result.status = GenerationStatus::CANCEL;
        }
    }
    if (!m_samples.empty()) {
        m_result.waveform = ov::Tensor(ov::element::f32, {1, 1, m_samples.size()});
        std::copy(m_samples.begin(), m_samples.end(), m_result.waveform.data<float>());
    }
    return std::move(m_result);
}

}  // namespace ov::genai
