// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

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

std::vector<int64_t> ctc_greedy_decode(const ov::Tensor& logits,
                                       const ov::Tensor& encoder_out_lens,
                                       int64_t blank_id) {
    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3,
                    "SenseVoiceSmall expects rank-3 [batch, time, vocab] 'logits', but got shape ",
                    shape,
                    ".");
    const size_t length = shape[1];
    const size_t vocab_size = shape[2];

    OPENVINO_ASSERT(encoder_out_lens.get_element_type() == ov::element::i32,
                    "SenseVoiceSmall expects 'encoder_out_lens' of type i32, but got ",
                    encoder_out_lens.get_element_type(),
                    ".");
    const int32_t valid_signed = encoder_out_lens.data<const int32_t>()[0];
    const size_t valid = std::min(static_cast<size_t>(std::max(valid_signed, 0)), length);

    const float* data = logits.data<const float>();
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
    // Textnorm is consumed but not exposed as a result feature.
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

    ov::Core core = utils::singleton_core();
    ov::CompiledModel model = core.compile_model(models_path / "openvino_model.xml", device, properties_copy);
    utils::print_compiled_model_properties(model, "sensevoice small model");
    m_request = model.create_infer_request();
}

ASRDecodedResults SenseVoiceSmall::generate(const AudioInputs& audio_inputs,
                                            const std::optional<ASRGenerationConfig>& generation_config,
                                            const std::shared_ptr<StreamerBase> streamer) {
    const auto start_time = std::chrono::steady_clock::now();

    const ASRGenerationConfig config = resolve_generation_config(generation_config);

    const std::vector<float>& audio = std::visit(
        ov::genai::utils::overloaded{
            [](const std::vector<float>& input) -> const std::vector<float>& {
                return input;
            },
        },
        audio_inputs);

    ASRDecodedResults results;
    results.perf_metrics.load_time = m_load_time_ms;
    results.perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};

    const auto features_start_time = std::chrono::steady_clock::now();
    const ov::Tensor features = m_feature_extractor.extract(audio);
    const auto features_stop_time = std::chrono::steady_clock::now();
    results.perf_metrics.asr_raw_metrics.features_extraction_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(features_stop_time - features_start_time)));

    ov::Tensor speech_lengths(ov::element::i64, ov::Shape{1});
    speech_lengths.data<int64_t>()[0] = static_cast<int64_t>(features.get_shape()[1]);
    ov::Tensor language(ov::element::i64, ov::Shape{1});
    language.data<int64_t>()[0] = resolve_language_id(m_config.lid_dict, config.language);
    ov::Tensor textnorm(ov::element::i64, ov::Shape{1});
    const std::string textnorm_key = config.use_itn ? "withitn" : "woitn";
    textnorm.data<int64_t>()[0] = m_config.textnorm_dict.at(textnorm_key);

    const auto infer_start_time = std::chrono::steady_clock::now();
    m_request.set_tensor("input_features", features);
    m_request.set_tensor("speech_lengths", speech_lengths);
    m_request.set_tensor("language", language);
    m_request.set_tensor("textnorm", textnorm);
    m_request.infer();
    const ov::Tensor logits = m_request.get_tensor("logits");
    const ov::Tensor encoder_out_lens = m_request.get_tensor("encoder_out_lens");
    const auto infer_stop_time = std::chrono::steady_clock::now();
    const auto infer_ms = PerfMetrics::get_microsec(infer_stop_time - infer_start_time);
    results.perf_metrics.raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    results.perf_metrics.asr_raw_metrics.encode_inference_durations.emplace_back(infer_ms);

    const std::vector<int64_t> token_ids = ctc_greedy_decode(logits, encoder_out_lens, m_config.blank_id);

    const SenseVoiceRichPrefix rich_prefix = parse_rich_prefix(m_tokenizer, token_ids, m_config.lid_dict);
    const std::vector<int64_t> transcription_token_ids(token_ids.begin() + rich_prefix.rich_token_count,
                                                       token_ids.end());

    const auto detokenization_start_time = std::chrono::steady_clock::now();
    const std::string text = m_tokenizer.decode(transcription_token_ids);
    const auto detokenization_stop_time = std::chrono::steady_clock::now();
    results.texts.push_back(text);
    results.perf_metrics.raw_metrics.detokenization_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(detokenization_stop_time - detokenization_start_time)));

    results.scores.push_back(0.0f);
    const std::string result_language = (config.language.has_value() && config.language.value() != "auto")
                                            ? config.language.value()
                                            : rich_prefix.language;
    results.languages.push_back(result_language);

    ov::AnyMap feature_entry;
    if (rich_prefix.emotion.has_value()) {
        feature_entry.emplace("emotion", rich_prefix.emotion.value());
    }
    if (rich_prefix.event.has_value()) {
        feature_entry.emplace("event", rich_prefix.event.value());
    }
    results.features = std::vector<ov::AnyMap>{std::move(feature_entry)};

    if (streamer) {
        streamer->write(transcription_token_ids);
        streamer->end();
    }

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
