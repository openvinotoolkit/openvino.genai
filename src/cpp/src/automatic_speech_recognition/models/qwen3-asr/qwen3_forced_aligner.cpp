// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_forced_aligner.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <optional>
#include <set>

#include "forced_aligner_config.hpp"
#include "openvino/core/except.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "unicode_helpers.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

const std::string AUDIO_START_TOKEN = "<|audio_start|>";
const std::string AUDIO_PAD_TOKEN = "<|audio_pad|>";
const std::string AUDIO_END_TOKEN = "<|audio_end|>";
const std::string TIMESTAMP_TOKEN = "<timestamp>";

bool is_kept_char(uint32_t code) {
    return code == '\'' || unicode_helpers::is_unicode_letter_or_number(code);
}

std::vector<std::string> split_segment_with_chinese(const std::vector<unicode_helpers::Utf8Char>& seg) {
    std::vector<std::string> tokens;
    std::string buffer;
    for (const auto& ch : seg) {
        if (unicode_helpers::is_cjk_char(ch.code)) {
            if (!buffer.empty()) {
                tokens.push_back(buffer);
                buffer.clear();
            }
            tokens.push_back(ch.bytes);
        } else {
            buffer += ch.bytes;
        }
    }
    if (!buffer.empty()) {
        tokens.push_back(buffer);
    }
    return tokens;
}

std::string to_lower(const std::string& value) {
    std::string result = value;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return result;
}

std::string normalize_alignment_language(const std::string& language) {
    static const std::map<std::string, std::string> alias_to_canonical = {
        {"chinese", "chinese"},         {"english", "english"},        {"cantonese", "cantonese"},
        {"arabic", "arabic"},           {"german", "german"},          {"french", "french"},
        {"spanish", "spanish"},         {"portuguese", "portuguese"},  {"indonesian", "indonesian"},
        {"italian", "italian"},         {"korean", "korean"},          {"russian", "russian"},
        {"thai", "thai"},               {"vietnamese", "vietnamese"},  {"japanese", "japanese"},
        {"turkish", "turkish"},         {"hindi", "hindi"},            {"malay", "malay"},
        {"dutch", "dutch"},             {"swedish", "swedish"},        {"danish", "danish"},
        {"finnish", "finnish"},         {"polish", "polish"},          {"czech", "czech"},
        {"filipino", "filipino"},       {"persian", "persian"},        {"greek", "greek"},
        {"romanian", "romanian"},       {"hungarian", "hungarian"},    {"macedonian", "macedonian"},
        // Language-code aliases for the forced-aligner-supported subset.
        {"zh", "chinese"},              {"en", "english"},             {"yue", "cantonese"},
        {"de", "german"},               {"fr", "french"},              {"es", "spanish"},
        {"pt", "portuguese"},           {"it", "italian"},             {"ko", "korean"},
        {"ru", "russian"},              {"ja", "japanese"},
    };
    const size_t first = language.find_first_not_of(" \t\n\r\f\v");
    if (first == std::string::npos) {
        return language;
    }
    const size_t last = language.find_last_not_of(" \t\n\r\f\v");
    const auto it = alias_to_canonical.find(to_lower(language.substr(first, last - first + 1)));
    return it == alias_to_canonical.end() ? language : it->second;
}

std::vector<std::string> segment_alignment_units(const std::string& text,
                                                 const std::string& canonical_language) {
    // Japanese and Korean require nagisa/soynlp in the reference processor; reject them until
    // equivalent in-tree segmentation is available.
    static const std::set<std::string> supported_segmentation_languages = {
        "english", "chinese", "cantonese", "german", "spanish", "french", "italian", "portuguese", "russian"};
    OPENVINO_ASSERT(supported_segmentation_languages.count(canonical_language),
                    "Language '", canonical_language,
                    "' is supported by the forced-aligner model but not yet implemented by OpenVINO GenAI.");
    const std::vector<unicode_helpers::Utf8Char> chars = unicode_helpers::decode_utf8(text);

    std::vector<std::string> units;
    std::vector<unicode_helpers::Utf8Char> segment;

    auto flush_segment = [&]() {
        if (segment.empty()) {
            return;
        }
        std::vector<unicode_helpers::Utf8Char> cleaned;
        for (const auto& ch : segment) {
            if (is_kept_char(ch.code)) {
                cleaned.push_back(ch);
            }
        }
        segment.clear();
        if (!cleaned.empty()) {
            std::vector<std::string> split = split_segment_with_chinese(cleaned);
            units.insert(units.end(), split.begin(), split.end());
        }
    };

    for (const auto& ch : chars) {
        if (unicode_helpers::is_unicode_whitespace(ch.code)) {
            flush_segment();
        } else {
            segment.push_back(ch);
        }
    }
    flush_segment();

    return units;
}

// Repairs forced-aligner timestamps into a non-decreasing sequence. Ported from
// Qwen3ForceAlignProcessor.fix_timestamp:
// https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/qwen3_forced_aligner.py#L162
std::vector<int64_t> fix_timestamp(const std::vector<int64_t>& data) {
    const size_t n = data.size();
    if (n == 0) {
        return {};
    }

    std::vector<size_t> dp(n, 1);
    std::vector<int64_t> parent(n, -1);
    for (size_t i = 1; i < n; ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (data[j] <= data[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                parent[i] = static_cast<int64_t>(j);
            }
        }
    }

    size_t max_idx = 0;
    for (size_t i = 1; i < n; ++i) {
        if (dp[i] > dp[max_idx]) {
            max_idx = i;
        }
    }

    std::vector<bool> is_normal(n, false);
    for (int64_t idx = static_cast<int64_t>(max_idx); idx != -1; idx = parent[idx]) {
        is_normal[static_cast<size_t>(idx)] = true;
    }

    std::vector<double> result(data.begin(), data.end());
    size_t i = 0;
    while (i < n) {
        if (is_normal[i]) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < n && !is_normal[j]) {
            ++j;
        }
        const size_t anomaly_count = j - i;

        bool has_left = false;
        double left_val = 0.0;
        for (int64_t k = static_cast<int64_t>(i) - 1; k >= 0; --k) {
            if (is_normal[static_cast<size_t>(k)]) {
                left_val = result[static_cast<size_t>(k)];
                has_left = true;
                break;
            }
        }
        bool has_right = false;
        double right_val = 0.0;
        for (size_t k = j; k < n; ++k) {
            if (is_normal[k]) {
                right_val = result[k];
                has_right = true;
                break;
            }
        }

        if (anomaly_count <= 2) {
            for (size_t k = i; k < j; ++k) {
                if (!has_left) {
                    result[k] = right_val;
                } else if (!has_right) {
                    result[k] = left_val;
                } else {
                    const size_t dist_left = k - (i - 1);
                    const size_t dist_right = j - k;
                    result[k] = (dist_left <= dist_right) ? left_val : right_val;
                }
            }
        } else {
            if (has_left && has_right) {
                const double step = (right_val - left_val) / static_cast<double>(anomaly_count + 1);
                for (size_t k = i; k < j; ++k) {
                    result[k] = left_val + step * static_cast<double>(k - i + 1);
                }
            } else if (has_left) {
                for (size_t k = i; k < j; ++k) {
                    result[k] = left_val;
                }
            } else if (has_right) {
                for (size_t k = i; k < j; ++k) {
                    result[k] = right_val;
                }
            }
        }
        i = j;
    }

    std::vector<int64_t> fixed(n);
    for (size_t k = 0; k < n; ++k) {
        fixed[k] = static_cast<int64_t>(result[k]);
    }
    return fixed;
}

std::string build_marker_input(const std::vector<std::string>& units, size_t audio_frames) {
    std::string marker;
    marker.reserve(AUDIO_START_TOKEN.size() + audio_frames * AUDIO_PAD_TOKEN.size() + AUDIO_END_TOKEN.size());
    marker += AUDIO_START_TOKEN;
    for (size_t frame = 0; frame < audio_frames; ++frame) {
        marker += AUDIO_PAD_TOKEN;
    }
    marker += AUDIO_END_TOKEN;
    for (const auto& unit : units) {
        marker += unit;
        marker += TIMESTAMP_TOKEN;
        marker += TIMESTAMP_TOKEN;
    }
    return marker;
}

InputsEmbedderQwen3ASR create_inputs_embedder(const std::filesystem::path& models_path,
                                              const Tokenizer& tokenizer,
                                              const std::string& device,
                                              const ov::AnyMap& properties) {
    OPENVINO_ASSERT(std::filesystem::exists(models_path / "openvino_audio_encoder_model.xml"),
                    "Qwen3 forced aligner requires the audio encoder, text embeddings, and language model export, "
                    "but openvino_audio_encoder_model.xml is missing in ",
                    models_path,
                    ". Re-export the forced aligner with an Optimum Intel version that produces this format.");
    return InputsEmbedderQwen3ASR{VLMConfig{models_path / "config.json"}, models_path, tokenizer, device, properties};
}

ov::InferRequest create_language_model_request(const std::filesystem::path& models_path,
                                               int64_t classify_num,
                                               const std::string& device,
                                               const ov::AnyMap& properties) {
    ov::CompiledModel compiled_model =
        utils::singleton_core().compile_model(models_path / "openvino_language_model.xml", device, properties);
    utils::print_compiled_model_properties(compiled_model, "qwen3 forced-aligner language model");

    const ov::PartialShape logits_shape = compiled_model.output("logits").get_partial_shape();
    OPENVINO_ASSERT(logits_shape.rank().is_static() && logits_shape.size() == 3 && logits_shape[2].is_static() &&
                        logits_shape[2].get_length() == classify_num,
                    "Forced-aligner logits last dimension must equal classify_num=",
                    classify_num,
                    ".");
    return compiled_model.create_infer_request();
}

}  // namespace

Qwen3ForcedAligner::Qwen3ForcedAligner(const std::filesystem::path& models_path,
                                       const std::string& device,
                                       const ov::AnyMap& properties)
    : m_config{models_path / "config.json"},
      m_tokenizer{models_path},
      m_embedder{create_inputs_embedder(models_path, m_tokenizer, device, properties)},
      m_language_model{create_language_model_request(models_path, m_config.classify_num, device, properties)} {
    m_embedder.set_prompt_formatting(PromptFormatting::AlreadyTemplated);
    for (const std::string& language : m_config.support_languages) {
        const std::string canonical = normalize_alignment_language(language);
        if (!canonical.empty()) {
            m_supported_languages.insert(canonical);
        }
    }
}

std::string Qwen3ForcedAligner::resolve_language(const std::string& language) const {
    OPENVINO_ASSERT(language.find_first_not_of(" \t\n\r\f\v") != std::string::npos,
                    "Forced alignment requires a non-empty language.");
    const std::string canonical = normalize_alignment_language(language);
    OPENVINO_ASSERT(m_supported_languages.empty() || m_supported_languages.count(canonical),
                    "Language '", language, "' is not supported by this forced-aligner model.");
    return canonical;
}

std::vector<ASRDecodedResultChunk> Qwen3ForcedAligner::align(const std::vector<float>& audio,
                                                             const std::string& transcript,
                                                             const ov::AnyMap& properties) {
    std::optional<std::string> language;
    utils::read_anymap_param(properties, ov::genai::language.name(), language);
    OPENVINO_ASSERT(language.has_value(),
                    "Qwen3 forced aligner requires a language, but none was provided.");
    const std::string canonical_language = resolve_language(*language);

    const std::vector<std::string> units = segment_alignment_units(transcript, canonical_language);
    if (units.empty()) {
        return {};
    }
    OPENVINO_ASSERT(!audio.empty(), "Forced alignment requires non-empty audio.");

    const std::vector<EncodedAudio> audios =
        m_embedder.encode_audios({ov::Tensor{ov::element::f32, {audio.size()}, audio.data()}});
    const size_t audio_frames = audios.at(0).num_audio_tokens;

    // The embedder expands the single audio placeholder of the prompt to audio_frames placeholders.
    VLMPerfMetrics metrics;
    const ov::Tensor inputs_embeds =
        m_embedder.get_inputs_embeds(build_marker_input(units, 1), {}, {}, audios, metrics, true, {}, {}, {0}, {});
    // Each alignment is independent, so drop the token history the embedder accumulates.
    m_embedder.finish_chat();

    const ov::Tensor input_ids =
        m_tokenizer.encode(build_marker_input(units, audio_frames), ov::genai::add_special_tokens(false)).input_ids;
    const size_t input_len = input_ids.get_shape().at(1);
    const int64_t* input_ids_data = input_ids.data<const int64_t>();

    std::vector<size_t> timestamp_positions;
    timestamp_positions.reserve(units.size() * 2);
    for (size_t position = 0; position < input_len; ++position) {
        if (input_ids_data[position] == m_config.timestamp_token_id) {
            timestamp_positions.push_back(position);
        }
    }
    OPENVINO_ASSERT(timestamp_positions.size() == units.size() * 2,
                    "Forced-aligner tokenization produced ", timestamp_positions.size(),
                    " timestamp markers but expected ", units.size() * 2,
                    "; the aligner tokenizer does not encode <timestamp> atomically.");

    OPENVINO_ASSERT(inputs_embeds.get_shape().at(1) == input_len,
                    "Forced-aligner marker tokenization must match the embedded prompt length.");

    ov::Tensor attention_mask(ov::element::i64, {1, input_len});
    std::fill_n(attention_mask.data<int64_t>(), input_len, 1);
    ov::Tensor beam_idx(ov::element::i32, {1});
    beam_idx.data<int32_t>()[0] = 0;

    m_language_model.reset_state();
    m_language_model.set_tensor("attention_mask", attention_mask);
    m_language_model.set_tensor("position_ids", m_embedder.get_position_ids(input_len, 0).first);
    m_language_model.set_tensor("inputs_embeds", inputs_embeds);
    m_language_model.set_tensor("beam_idx", beam_idx);
    m_language_model.infer();

    const ov::Tensor logits = m_language_model.get_tensor("logits");
    const ov::Shape logits_shape = logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3 && logits_shape[0] == 1 && logits_shape[1] == input_len,
                    "Forced-aligner logits must be shaped [1, input_len, classify_num].");
    const size_t class_count = logits_shape.at(2);
    const float* logits_data = logits.data<const float>();

    std::vector<int64_t> timestamp_ms;
    timestamp_ms.reserve(timestamp_positions.size());
    for (const size_t position : timestamp_positions) {
        const float* row = logits_data + position * class_count;
        const size_t class_id = static_cast<size_t>(std::max_element(row, row + class_count) - row);
        timestamp_ms.push_back(std::llround(static_cast<double>(class_id) * m_config.timestamp_segment_ms));
    }

    const std::vector<int64_t> fixed_ms = fix_timestamp(timestamp_ms);

    std::vector<ASRDecodedResultChunk> words;
    words.reserve(units.size());
    for (size_t unit_index = 0; unit_index < units.size(); ++unit_index) {
        const float start_ts = static_cast<float>(fixed_ms[unit_index * 2]) / 1000.0f;
        const float end_ts = static_cast<float>(fixed_ms[unit_index * 2 + 1]) / 1000.0f;

        // Alignment units are normalized independently of ASR tokenization, so there is no reliable
        // per-unit mapping back to decoder token IDs.
        words.push_back({start_ts, end_ts, units[unit_index], {}});
    }

    return words;
}

}  // namespace ov::genai
