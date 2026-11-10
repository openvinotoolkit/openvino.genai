// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "automatic_speech_recognition/audio_chunk.hpp"
#include "automatic_speech_recognition/models/sensevoice/npu/static_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "utils.hpp"

namespace {

int64_t resolve_language_id(const std::map<std::string, int64_t>& lid_dict,
                            const std::optional<std::string>& language) {
    const std::string key = language.value_or("auto");
    const auto it = lid_dict.find(key);
    OPENVINO_ASSERT(it != lid_dict.end(), "SenseVoiceSmall does not support language '", key, "'");
    return it->second;
}

// Zero-pads features of `count` audios starting at `begin` to at least `min_frames`; `speech_lengths` keeps real lengths.
ov::Tensor stack_padded_features(const std::vector<ov::Tensor>& features,
                                 size_t begin,
                                 size_t count,
                                 size_t min_frames,
                                 ov::Tensor& speech_lengths) {
    int64_t* lengths = speech_lengths.data<int64_t>();
    size_t max_frames = min_frames;
    for (size_t b = 0; b < count; ++b) {
        const size_t frames = features[begin + b].get_shape()[1];
        lengths[b] = static_cast<int64_t>(frames);
        max_frames = std::max(max_frames, frames);
    }
    if (count == 1 && max_frames == features[begin].get_shape()[1]) {
        return features[begin];
    }

    const size_t feature_size = features[begin].get_shape()[2];
    ov::Tensor batch(ov::element::f32, ov::Shape{count, max_frames, feature_size});
    float* data = batch.data<float>();
    std::fill_n(data, batch.get_size(), 0.0f);
    for (size_t b = 0; b < count; ++b) {
        const ov::Tensor& item = features[begin + b];
        std::copy_n(item.data<const float>(), item.get_size(), data + b * max_frames * feature_size);
    }
    return batch;
}

std::vector<int64_t> ctc_greedy_decode(const ov::Tensor& logits,
                                       const ov::Tensor& encoder_out_lens,
                                       size_t batch_index,
                                       int64_t blank_id) {
    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3,
                    "SenseVoiceSmall expects rank-3 [batch, time, vocab] 'logits', but got shape ",
                    shape,
                    ".");
    const size_t batch_size = shape[0];
    const size_t length = shape[1];
    const size_t vocab_size = shape[2];

    OPENVINO_ASSERT(encoder_out_lens.get_element_type() == ov::element::i32,
                    "SenseVoiceSmall expects 'encoder_out_lens' of type i32, but got ",
                    encoder_out_lens.get_element_type(),
                    ".");
    OPENVINO_ASSERT(encoder_out_lens.get_size() == batch_size && batch_index < batch_size,
                    "SenseVoiceSmall expects 'encoder_out_lens' with one entry per 'logits' batch item, but got ",
                    encoder_out_lens.get_size(),
                    " entries for batch size ",
                    batch_size,
                    ".");
    const int32_t encoder_output_length = encoder_out_lens.data<const int32_t>()[batch_index];
    const size_t valid = std::min(static_cast<size_t>(std::max(encoder_output_length, 0)), length);

    const float* data = logits.data<const float>() + batch_index * length * vocab_size;
    std::vector<int64_t> token_ids;
    int64_t previous = -1;
    for (size_t t = 0; t < valid; ++t) {
        const float* frame = data + t * vocab_size;
        const float* best = std::max_element(frame, frame + vocab_size);
        const int64_t id = static_cast<int64_t>(best - frame);
        if (id != previous && id != blank_id) {
            token_ids.push_back(id);
        }
        previous = id;
    }
    return token_ids;
}

struct SenseVoiceRichPrefix {
    size_t rich_token_count = 0;
    std::string language;
    std::optional<std::string> emotion;
    std::optional<std::string> event;
};

std::optional<std::string> parse_rich_tag(const std::string& decoded) {
    const size_t begin = decoded.find_first_not_of(" \t\n\r");
    if (begin == std::string::npos) {
        return std::nullopt;
    }
    const size_t end = decoded.find_last_not_of(" \t\n\r");
    const std::string trimmed = decoded.substr(begin, end - begin + 1);
    if (trimmed.size() < 4 || trimmed.compare(0, 2, "<|") != 0 ||
        trimmed.compare(trimmed.size() - 2, 2, "|>") != 0) {
        return std::nullopt;
    }
    return trimmed.substr(2, trimmed.size() - 4);
}

SenseVoiceRichPrefix parse_rich_prefix(ov::genai::Tokenizer& tokenizer,
                                       const std::vector<int64_t>& token_ids,
                                       const std::map<std::string, int64_t>& lid_dict) {
    SenseVoiceRichPrefix prefix;
    // SenseVoice prefix order: language, emotion, event, textnorm.
    // Textnorm is consumed but not exposed as result metadata.
    std::array<std::optional<std::string>, 4> tags;
    const size_t max_rich = std::min<size_t>(tags.size(), token_ids.size());
    for (size_t i = 0; i < max_rich; ++i) {
        std::optional<std::string> tag = parse_rich_tag(tokenizer.decode(std::vector<int64_t>{token_ids[i]}));
        if (!tag.has_value()) {
            break;
        }
        tags[i] = std::move(tag);
        ++prefix.rich_token_count;
    }
    if (tags[0].has_value()) {
        const std::string& language = *tags[0];
        if (language != "auto" && language != "nospeech" && lid_dict.find(language) != lid_dict.end()) {
            prefix.language = language;
        }
    }
    prefix.emotion = tags[1];
    prefix.event = tags[2];
    return prefix;
}

// Largest audio length in samples whose features fit into `frames` post-LFR frames.
constexpr size_t max_audio_samples_for_frames(size_t frames) {
    using ov::genai::FunASRFeatureExtractor;
    const size_t max_fbank_frames = frames * FunASRFeatureExtractor::lfr_stride;
    return (max_fbank_frames - 1) * FunASRFeatureExtractor::frame_shift + FunASRFeatureExtractor::frame_length +
           FunASRFeatureExtractor::frame_shift - 1;
}

// Result of one audio chunk. An audio that fits the model limits has a single chunk.
struct SegmentResult {
    std::string text;
    SenseVoiceRichPrefix prefix;
    size_t samples = 0;
};

struct AudioResult {
    std::string text;
    std::string language;
    std::optional<std::string> emotion;
    std::optional<std::string> event;
};

using Votes = std::vector<std::pair<std::string, size_t>>;

// Returns the value with the largest total weight. Values listed in `ignored` are used only if nothing else is voted.
std::optional<std::string> weighted_vote(const Votes& votes, const std::vector<std::string>& ignored) {
    const auto tally = [&](bool skip_ignored) -> std::optional<std::string> {
        Votes totals;
        for (const auto& [value, weight] : votes) {
            if (skip_ignored && std::find(ignored.begin(), ignored.end(), value) != ignored.end()) {
                continue;
            }
            const auto it = std::find_if(totals.begin(), totals.end(), [&](const auto& total) {
                return total.first == value;
            });
            if (it == totals.end()) {
                totals.emplace_back(value, weight);
            } else {
                it->second += weight;
            }
        }
        if (totals.empty()) {
            return std::nullopt;
        }
        return std::max_element(totals.begin(),
                                totals.end(),
                                [](const auto& a, const auto& b) {
                                    return a.second < b.second;
                                })
            ->first;
    };
    if (const auto value = tally(true)) {
        return value;
    }
    return tally(false);
}

// Merges chunk results of one audio. Language, emotion and event are voted weighted by chunk duration.
AudioResult merge_segments(const std::vector<SegmentResult>& segments, const std::optional<std::string>& forced_language) {
    Votes languages;
    Votes emotions;
    Votes events;
    for (const SegmentResult& segment : segments) {
        if (!segment.prefix.language.empty()) {
            languages.emplace_back(segment.prefix.language, segment.samples);
        }
        if (segment.prefix.emotion.has_value()) {
            emotions.emplace_back(*segment.prefix.emotion, segment.samples);
        }
        if (segment.prefix.event.has_value()) {
            events.emplace_back(*segment.prefix.event, segment.samples);
        }
    }

    AudioResult result;
    result.language = (forced_language.has_value() && *forced_language != "auto")
                          ? *forced_language
                          : weighted_vote(languages, {}).value_or("");
    result.emotion = weighted_vote(emotions, {"EMO_UNKNOWN"});
    result.event = weighted_vote(events, {"Speech"});

    // Words are not separated by spaces in these languages.
    const bool joined_without_space = result.language == "zh" || result.language == "ja" || result.language == "yue";
    for (const SegmentResult& segment : segments) {
        if (segment.text.empty()) {
            continue;
        }
        if (!result.text.empty() && !joined_without_space) {
            result.text += ' ';
        }
        result.text += segment.text;
    }
    return result;
}

}  // namespace

namespace ov::genai {

SenseVoiceSmall::SenseVoiceSmall(const std::filesystem::path& models_path,
                                 const std::string& device,
                                 const ov::AnyMap& properties)
    : ASRPipelineImplBase(models_path),
      m_config(models_path / "config.json"),
      m_feature_extractor(models_path / "am.mvn", m_config.dither) {
    ov::AnyMap properties_copy = properties;
    erase_allowed_asr_ctor_properties(properties_copy);

    ov::Core& core = utils::singleton_core();
    const std::filesystem::path model_path = models_path / "openvino_model.xml";
    ov::CompiledModel model = device == "NPU" ? compile_sensevoice_for_npu(model_path, properties_copy)
                                              : core.compile_model(model_path, device, properties_copy);
    const ov::PartialShape input_shape = model.input("input_features").get_partial_shape();
    if (input_shape.is_static()) {
        m_max_batch = input_shape[0].get_length();
        m_padded_frames = input_shape[1].get_length();
        m_max_audio_samples = max_audio_samples_for_frames(m_padded_frames);
    }
    utils::print_compiled_model_properties(model, "sensevoice small model");
    m_request = model.create_infer_request();
}

ASRDecodedResults SenseVoiceSmall::generate(const AudiosView& audios,
                                            const std::optional<ASRGenerationConfig>& generation_config,
                                            const std::shared_ptr<StreamerBase> streamer) {
    const auto start_time = std::chrono::steady_clock::now();

    const ASRGenerationConfig config = resolve_generation_config(generation_config);

    const size_t batch_size = audios.size();
    OPENVINO_ASSERT(!streamer || batch_size == 1,
                    "SenseVoiceSmall supports streamers only for a single audio input, but received ",
                    batch_size,
                    " audio inputs.");

    ASRDecodedResults results;
    results.perf_metrics.load_time = m_load_time_ms;
    results.perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};

    const auto features_start_time = std::chrono::steady_clock::now();
    // Features of audio chunks and the index of the audio each chunk belongs to. Audio longer than the model
    // limit is split into chunks, otherwise the audio is a single chunk.
    std::vector<ov::Tensor> features;
    std::vector<size_t> chunk_audio_index;
    std::vector<size_t> chunk_samples;
    features.reserve(batch_size);
    for (size_t audio_index = 0; audio_index < batch_size; ++audio_index) {
        const std::vector<float>& audio = audios[audio_index];
        if (audio.size() <= m_max_audio_samples) {
            features.push_back(m_feature_extractor.extract(audio));
            chunk_audio_index.push_back(audio_index);
            chunk_samples.push_back(audio.size());
            continue;
        }
        const size_t sampling_rate = FunASRFeatureExtractor::sampling_rate;
        const size_t max_chunk_seconds = m_max_audio_samples / sampling_rate - CHUNK_BOUNDARY_SEARCH_SECONDS;
        for (const AudioChunk& chunk : split_audio_into_chunks({audio}, sampling_rate, max_chunk_seconds)) {
            features.push_back(m_feature_extractor.extract(chunk.wav));
            chunk_audio_index.push_back(audio_index);
            chunk_samples.push_back(chunk.wav.size());
        }
    }
    OPENVINO_ASSERT(!streamer || features.size() == 1,
                    "SenseVoiceSmall supports streamers only for audio that is not split into chunks, "
                    "but the audio is longer than ",
                    m_max_audio_samples / FunASRFeatureExtractor::sampling_rate,
                    " seconds supported by the model.");
    const auto features_stop_time = std::chrono::steady_clock::now();
    results.perf_metrics.asr_raw_metrics.features_extraction_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(features_stop_time - features_start_time)));

    const int64_t language_id = resolve_language_id(m_config.lid_dict, config.language);
    const std::string textnorm_key = config.use_itn ? "withitn" : "woitn";
    const int64_t textnorm_id = m_config.textnorm_dict.at(textnorm_key);

    float detokenization_microseconds = 0.0f;
    size_t generated_token_count = 0;
    std::chrono::steady_clock::time_point detokenization_stop_time;
    std::vector<std::vector<SegmentResult>> audio_segments(batch_size);

    const size_t chunk_count = features.size();
    const size_t group_size = std::min(chunk_count, m_max_batch);
    for (size_t group_begin = 0; group_begin < chunk_count; group_begin += group_size) {
        const size_t count = std::min(group_size, chunk_count - group_begin);

        ov::Tensor speech_lengths(ov::element::i64, ov::Shape{count});
        const ov::Tensor input_features =
            stack_padded_features(features, group_begin, count, m_padded_frames, speech_lengths);
        ov::Tensor language(ov::element::i64, ov::Shape{count});
        std::fill_n(language.data<int64_t>(), count, language_id);
        ov::Tensor textnorm(ov::element::i64, ov::Shape{count});
        std::fill_n(textnorm.data<int64_t>(), count, textnorm_id);

        m_request.set_tensor("input_features", input_features);
        m_request.set_tensor("speech_lengths", speech_lengths);
        m_request.set_tensor("language", language);
        m_request.set_tensor("textnorm", textnorm);

        const auto infer_start_time = std::chrono::steady_clock::now();
        m_request.infer();
        const auto infer_stop_time = std::chrono::steady_clock::now();
        const auto infer_ms = PerfMetrics::get_microsec(infer_stop_time - infer_start_time);
        results.perf_metrics.raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
        results.perf_metrics.asr_raw_metrics.encode_inference_durations.emplace_back(infer_ms);

        const ov::Tensor logits = m_request.get_tensor("logits");
        const ov::Tensor encoder_out_lens = m_request.get_tensor("encoder_out_lens");

        for (size_t b = 0; b < count; ++b) {
            const std::vector<int64_t> token_ids = ctc_greedy_decode(logits, encoder_out_lens, b, m_config.blank_id);

            SegmentResult segment;
            segment.prefix = parse_rich_prefix(m_tokenizer, token_ids, m_config.lid_dict);
            segment.samples = chunk_samples[group_begin + b];
            const std::vector<int64_t> transcription_token_ids(token_ids.begin() + segment.prefix.rich_token_count,
                                                               token_ids.end());

            const auto detokenization_start_time = std::chrono::steady_clock::now();
            segment.text = m_tokenizer.decode(transcription_token_ids);
            detokenization_stop_time = std::chrono::steady_clock::now();
            detokenization_microseconds += PerfMetrics::get_microsec(detokenization_stop_time - detokenization_start_time);
            generated_token_count += token_ids.size();
            audio_segments[chunk_audio_index[group_begin + b]].push_back(std::move(segment));

            if (streamer) {
                streamer->write(transcription_token_ids);
                streamer->end();
            }
        }
    }

    std::vector<ov::AnyMap> metadata;
    metadata.reserve(batch_size);
    for (const std::vector<SegmentResult>& segments : audio_segments) {
        AudioResult audio_result = merge_segments(segments, config.language);
        results.texts.push_back(std::move(audio_result.text));
        results.scores.push_back(0.0f);
        results.languages.push_back(std::move(audio_result.language));

        ov::AnyMap metadata_entry;
        if (audio_result.emotion.has_value()) {
            metadata_entry.emplace("emotion", audio_result.emotion.value());
        }
        if (audio_result.event.has_value()) {
            metadata_entry.emplace("event", audio_result.event.value());
        }
        metadata.push_back(std::move(metadata_entry));
    }
    results.metadata = std::move(metadata);

    results.perf_metrics.raw_metrics.detokenization_durations.emplace_back(MicroSeconds(detokenization_microseconds));
    results.perf_metrics.raw_metrics.m_new_token_times.emplace_back(detokenization_stop_time);
    results.perf_metrics.raw_metrics.m_batch_sizes.emplace_back(generated_token_count);

    const auto stop_time = std::chrono::steady_clock::now();
    results.perf_metrics.raw_metrics.generate_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(stop_time - start_time)));
    results.perf_metrics.evaluate_statistics(start_time);
    return results;
}

void SenseVoiceSmall::set_generation_config(const ASRGenerationConfig& config) {
    validate_generation_config(config);
    m_generation_config = config;
}

ASRGenerationConfig SenseVoiceSmall::resolve_generation_config(
    const std::optional<ASRGenerationConfig>& generation_config) const {
    ASRGenerationConfig config = generation_config.value_or(m_generation_config);
    validate_generation_config(config);
    return config;
}

void SenseVoiceSmall::validate_generation_config(const ASRGenerationConfig& config) const {
    OPENVINO_ASSERT(config.num_return_sequences == 1,
                    "SenseVoiceSmall supports only 'num_return_sequences' == 1. Provided: ",
                    config.num_return_sequences,
                    ".");
    OPENVINO_ASSERT(config.is_greedy_decoding(),
                    "SenseVoiceSmall uses single-pass CTC greedy decoding and does not support beam search, "
                    "sampling, or tree (EAGLE) decoding. Set num_beams=1, do_sample=false, tree_depth=0.");
    OPENVINO_ASSERT(!config.is_assisting_generation(), "SenseVoiceSmall does not support assisted generation.");
}

}  // namespace ov::genai
