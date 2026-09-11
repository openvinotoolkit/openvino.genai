// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_forced_aligner.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <nlohmann/json.hpp>

#include "json_utils.hpp"
#include "openvino/core/except.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

const std::string AUDIO_START_TOKEN = "<|audio_start|>";
const std::string AUDIO_PAD_TOKEN = "<|audio_pad|>";
const std::string AUDIO_END_TOKEN = "<|audio_end|>";
const std::string TIMESTAMP_TOKEN = "<timestamp>";

struct Utf8Char {
    uint32_t code;
    std::string bytes;
};

std::vector<Utf8Char> decode_utf8(const std::string& text) {
    std::vector<Utf8Char> chars;
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        const auto byte = static_cast<unsigned char>(text[i]);
        size_t len = 1;
        uint32_t code = byte;
        if (byte < 0x80) {
            len = 1;
            code = byte;
        } else if ((byte >> 5) == 0x06) {
            len = 2;
            code = byte & 0x1F;
        } else if ((byte >> 4) == 0x0E) {
            len = 3;
            code = byte & 0x0F;
        } else if ((byte >> 3) == 0x1E) {
            len = 4;
            code = byte & 0x07;
        } else {
            len = 1;
            code = byte;
        }
        if (i + len > n) {
            len = 1;
            code = byte;
        }
        for (size_t k = 1; k < len; ++k) {
            code = (code << 6) | (static_cast<unsigned char>(text[i + k]) & 0x3F);
        }
        chars.push_back({code, text.substr(i, len)});
        i += len;
    }
    return chars;
}

bool is_cjk_char(uint32_t code) {
    return (0x4E00 <= code && code <= 0x9FFF) || (0x3400 <= code && code <= 0x4DBF) ||
           (0x20000 <= code && code <= 0x2A6DF) || (0x2A700 <= code && code <= 0x2B73F) ||
           (0x2B740 <= code && code <= 0x2B81F) || (0x2B820 <= code && code <= 0x2CEAF) ||
           (0xF900 <= code && code <= 0xFAFF);
}

// Covers the Unicode letter/number ranges needed by forced-aligner text preprocessing.
// This is not a complete implementation of Unicode L/N categories.
bool is_unicode_letter_or_number(uint32_t code) {
    if (code < 0x80) {
        return std::isalnum(static_cast<int>(code)) != 0;
    }
    if (code == 0xD7 || code == 0xF7) {  // multiplication / division sign (Sm)
        return false;
    }
    if (code >= 0xC0 && code <= 0xFF) {  // Latin-1 letters
        return true;
    }
    if (code == 0xAA || code == 0xB5 || code == 0xBA) {  // ordinal / micro letters
        return true;
    }
    if (code >= 0x0100 && code <= 0x024F) {  // Latin Extended-A / Extended-B
        return true;
    }
    if (code >= 0x1E00 && code <= 0x1EFF) {  // Latin Extended Additional
        return true;
    }
    if (code >= 0x0400 && code <= 0x04FF && !(code >= 0x0483 && code <= 0x0489)) {  // Cyrillic (skip combining)
        return true;
    }
    if (code >= 0x0500 && code <= 0x052F) {  // Cyrillic Supplement
        return true;
    }
    if (code >= 0x3040 && code <= 0x30FF) {  // Hiragana / Katakana
        return true;
    }
    if (code >= 0x1100 && code <= 0x11FF) {  // Hangul Jamo
        return true;
    }
    if (code >= 0x3130 && code <= 0x318F) {  // Hangul Compatibility Jamo
        return true;
    }
    if (code >= 0xAC00 && code <= 0xD7A3) {  // Hangul syllables
        return true;
    }
    if (code >= 0xFF10 && code <= 0xFF19) {  // Fullwidth digits
        return true;
    }
    if ((code >= 0xFF21 && code <= 0xFF3A) || (code >= 0xFF41 && code <= 0xFF5A)) {  // Fullwidth Latin letters
        return true;
    }
    if (code >= 0xFF66 && code <= 0xFF9D) {  // Halfwidth Katakana
        return true;
    }
    // Additional Unicode number code points used by the reference preprocessing.
    if (code == 0xB2 || code == 0xB3 || code == 0xB9) {  // superscript two/three/one (No)
        return true;
    }
    if (code >= 0xBC && code <= 0xBE) {  // vulgar fractions 1/4, 1/2, 3/4 (No)
        return true;
    }
    if (code >= 0x2160 && code <= 0x217F) {  // Roman numerals (Nl)
        return true;
    }
    if (code == 0x3007) {  // ideographic number zero (Nl)
        return true;
    }
    return is_cjk_char(code);
}

bool is_kept_char(uint32_t code) {
    return code == '\'' || is_unicode_letter_or_number(code);
}

// Matches Python str.split()/str.isspace() whitespace semantics used by the reference preprocessing.
bool is_unicode_whitespace(uint32_t code) {
    return (0x09 <= code && code <= 0x0D) ||  // TAB, LF, VT, FF, CR
           (0x1C <= code && code <= 0x1F) ||  // FS, GS, RS, US
           code == 0x20 || code == 0x85 || code == 0xA0 || code == 0x1680 ||
           (0x2000 <= code && code <= 0x200A) || code == 0x2028 || code == 0x2029 || code == 0x202F ||
           code == 0x205F || code == 0x3000;
}

std::vector<std::string> split_segment_with_chinese(const std::vector<Utf8Char>& seg) {
    std::vector<std::string> tokens;
    std::string buffer;
    for (const auto& ch : seg) {
        if (is_cjk_char(ch.code)) {
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

}  // namespace

std::string normalize_alignment_language(const std::string& language) {
    // Normalizes a recognized language name or code to the canonical name used by the forced aligner.
    static const std::map<std::string, std::string> alias_to_canonical = {
        // Qwen3-ASR supported language names.
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
        return {};
    }
    const size_t last = language.find_last_not_of(" \t\n\r\f\v");
    const auto it = alias_to_canonical.find(to_lower(language.substr(first, last - first + 1)));
    return it == alias_to_canonical.end() ? std::string{} : it->second;
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
    const std::vector<Utf8Char> chars = decode_utf8(text);

    std::vector<std::string> units;
    std::vector<Utf8Char> segment;

    auto flush_segment = [&]() {
        if (segment.empty()) {
            return;
        }
        std::vector<Utf8Char> cleaned;
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
        if (is_unicode_whitespace(ch.code)) {
            flush_segment();
        } else {
            segment.push_back(ch);
        }
    }
    flush_segment();

    return units;
}

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

Qwen3ForcedAligner::Qwen3ForcedAligner(const std::filesystem::path& models_path,
                                       const std::string& device,
                                       const ov::AnyMap& properties)
    : m_feature_extractor{models_path / "preprocessor_config.json"},
      m_tokenizer{models_path} {
    const std::filesystem::path config_path = models_path / "config.json";
    OPENVINO_ASSERT(std::filesystem::exists(config_path),
                    "Forced-aligner config.json not found at '", config_path.string(), "'.");
    const char* required_files[] = {"openvino_encoder_model.xml",
                                    "openvino_decoder_model.xml",
                                    "openvino_tokenizer.xml"};
    for (const char* required : required_files) {
        OPENVINO_ASSERT(std::filesystem::exists(models_path / required),
                        "Forced-aligner model is missing required file '", required, "' in '",
                        models_path.string(), "'.");
    }

    std::ifstream config_stream(config_path);
    nlohmann::json config = nlohmann::json::parse(config_stream);

    OPENVINO_ASSERT(config.contains("timestamp_token_id") && config.contains("timestamp_segment_time"),
                    "Model at '", models_path.string(),
                    "' is not a Qwen3 forced aligner: config.json lacks timestamp_token_id / "
                    "timestamp_segment_time.");

    OPENVINO_ASSERT(config.contains("thinker_config") && config.at("thinker_config").contains("classify_num"),
                    "Model at '", models_path.string(),
                    "' is not a Qwen3 forced aligner: config.json lacks thinker_config.classify_num.");
    const nlohmann::json& classify_num_json = config.at("thinker_config").at("classify_num");
    OPENVINO_ASSERT(classify_num_json.is_number_integer() && classify_num_json.get<int64_t>() > 0,
                    "Forced-aligner thinker_config.classify_num must be a positive integer. Got: ",
                    classify_num_json.dump(), ".");
    const size_t classify_num = static_cast<size_t>(classify_num_json.get<int64_t>());

    const nlohmann::json& timestamp_token_id_json = config.at("timestamp_token_id");
    OPENVINO_ASSERT(timestamp_token_id_json.is_number_integer(),
                    "Forced-aligner timestamp_token_id must be an integer. Got: ",
                    timestamp_token_id_json.dump(),
                    ".");
    m_timestamp_token_id = timestamp_token_id_json.get<int64_t>();

    const nlohmann::json& segment_time_json = config.at("timestamp_segment_time");
    OPENVINO_ASSERT(segment_time_json.is_number() && segment_time_json.get<double>() > 0.0,
                    "Forced-aligner timestamp_segment_time must be a positive number. Got: ",
                    segment_time_json.dump(),
                    ".");
    m_timestamp_segment_ms = segment_time_json.get<float>();

    if (config.contains("support_languages") && config.at("support_languages").is_array()) {
        for (const auto& language : config.at("support_languages")) {
            m_supported_languages.insert(to_lower(language.get<std::string>()));
        }
    }

    m_encoder = std::make_unique<Qwen3ForcedAlignerEncoder>(models_path, device, properties);

    ov::Core core = utils::singleton_core();
    ov::CompiledModel compiled_decoder =
        core.compile_model(models_path / "openvino_decoder_model.xml", device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_decoder, "qwen3 forced-aligner decoder model");
    m_decoder = compiled_decoder.create_infer_request();

    const auto has_input = [&](const std::string& name) {
        for (const auto& port : compiled_decoder.inputs()) {
            if (port.get_names().count(name)) {
                return true;
            }
        }
        return false;
    };
    OPENVINO_ASSERT(has_input("encoder_hidden_states") && has_input("input_ids") && has_input("beam_idx"),
                    "Forced-aligner decoder must expose inputs encoder_hidden_states, input_ids and "
                    "beam_idx.");

    const ov::PartialShape logits_shape = compiled_decoder.output("logits").get_partial_shape();
    OPENVINO_ASSERT(logits_shape.rank().is_static() && logits_shape.size() == 3 &&
                        logits_shape[2].is_static() &&
                        static_cast<size_t>(logits_shape[2].get_length()) == classify_num,
                    "Forced-aligner decoder logits last dimension must equal classify_num=", classify_num, ".");
}

std::string Qwen3ForcedAligner::resolve_language(const std::string& language) const {
    const std::string canonical = normalize_alignment_language(language);
    OPENVINO_ASSERT(!canonical.empty(),
                    "Forced alignment received invalid or missing language information: '", language, "'.");
    OPENVINO_ASSERT(m_supported_languages.empty() || m_supported_languages.count(canonical),
                    "Language '", language, "' is not supported by this forced-aligner model.");
    return canonical;
}

std::string Qwen3ForcedAligner::build_marker_input(const std::vector<std::string>& units, size_t audio_frames) const {
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

std::vector<ASRDecodedResultChunk> Qwen3ForcedAligner::align(const std::vector<float>& audio,
                                                             const std::string& transcript,
                                                             const std::string& language) {
    const std::string canonical_language = resolve_language(language);

    const std::vector<std::string> units = segment_alignment_units(transcript, canonical_language);
    if (units.empty()) {
        return {};
    }

    const WhisperFeatures features = m_feature_extractor.extract(audio, false);
    const ov::Tensor encoder_hidden_states = m_encoder->encode(features);
    const size_t audio_frames = encoder_hidden_states.get_shape().at(1);

    const std::string marker = build_marker_input(units, audio_frames);
    const ov::Tensor input_ids = m_tokenizer.encode(std::vector<std::string>{marker}).input_ids;

    const size_t input_len = input_ids.get_shape().at(1);
    const int64_t* input_ids_data = input_ids.data<const int64_t>();

    std::vector<size_t> timestamp_positions;
    timestamp_positions.reserve(units.size() * 2);
    for (size_t position = 0; position < input_len; ++position) {
        if (input_ids_data[position] == m_timestamp_token_id) {
            timestamp_positions.push_back(position);
        }
    }
    OPENVINO_ASSERT(timestamp_positions.size() == units.size() * 2,
                    "Forced-aligner tokenization produced ", timestamp_positions.size(),
                    " timestamp markers but expected ", units.size() * 2,
                    "; the aligner tokenizer does not encode <timestamp> atomically.");

    m_decoder.reset_state();
    m_decoder.set_tensor("encoder_hidden_states", encoder_hidden_states);
    ov::Tensor beam_idx(ov::element::i32, {1});
    beam_idx.data<int32_t>()[0] = 0;
    m_decoder.set_tensor("beam_idx", beam_idx);
    m_decoder.set_tensor("input_ids", input_ids);
    m_decoder.infer();

    const ov::Tensor logits = m_decoder.get_tensor("logits");
    const ov::Shape logits_shape = logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3 && logits_shape[0] == 1 && logits_shape[1] == input_len,
                    "Forced-aligner decoder logits must be shaped [1, input_len, classify_num].");
    const size_t class_count = logits_shape.at(2);
    const float* logits_data = logits.data<const float>();

    std::vector<int64_t> timestamp_ms;
    timestamp_ms.reserve(timestamp_positions.size());
    for (const size_t position : timestamp_positions) {
        const float* row = logits_data + position * class_count;
        const size_t class_id = static_cast<size_t>(std::max_element(row, row + class_count) - row);
        timestamp_ms.push_back(std::llround(static_cast<double>(class_id) * m_timestamp_segment_ms));
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
