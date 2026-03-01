// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "kokoro_tts_model.hpp"

#include <algorithm>
#include <chrono>
#include <codecvt>
#include <cstdlib>
#include <fstream>
#include <locale>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "openvino/genai/perf_metrics.hpp"
#include "utils.hpp"

#if OPENVINO_GENAI_HAS_MISAKI_CPP
#include "misaki/g2p.hpp"
#include "misaki/fallbacks.hpp"
#endif

namespace {

std::string normalize_language_variant(const std::string& language) {
    // Python parity: mirrors alias handling in `kokoro/pipeline.py` (`ALIASES`),
    // where `en-us` -> `a` and `en-gb` -> `b` are treated as equivalent variants.
    std::string normalized = language;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (normalized == "a" || normalized == "en" || normalized == "en-us") {
        return "en-us";
    }
    if (normalized == "b" || normalized == "en-gb") {
        return "en-gb";
    }

    OPENVINO_THROW("Unsupported Kokoro language '", language, "'. Supported values: en-us, en-gb, a, b");
}

std::string to_utf8(char32_t codepoint) {
    std::u32string input(1, codepoint);
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> convert;
    return convert.to_bytes(input);
}

std::u32string from_utf8(const std::string& input) {
    // Lenient UTF-8 decoder used for TTS preprocessing.
    // - Valid UTF-8 sequences are decoded into Unicode code points.
    // - Invalid/truncated sequences fall back to byte-wise passthrough (no throw).
    //
    // Validation rules follow the standard UTF-8 form from RFC 3629:
    // https://datatracker.ietf.org/doc/html/rfc3629
    // with additional checks for overlong encodings and surrogate-range exclusion.
    std::u32string output;
    output.reserve(input.size());

    size_t index = 0;
    while (index < input.size()) {
        const unsigned char b0 = static_cast<unsigned char>(input[index]);

        // 1-byte ASCII: 0xxxxxxx
        if ((b0 & 0x80) == 0) {
            output.push_back(static_cast<char32_t>(b0));
            ++index;
            continue;
        }

        // 2-byte sequence: 110xxxxx 10xxxxxx
        if ((b0 & 0xE0) == 0xC0 && index + 1 < input.size()) {
            const unsigned char b1 = static_cast<unsigned char>(input[index + 1]);
            if ((b1 & 0xC0) == 0x80) {
                const char32_t cp = (static_cast<char32_t>(b0 & 0x1F) << 6) |
                                    static_cast<char32_t>(b1 & 0x3F);
                // Reject overlong 2-byte form (must encode >= U+0080).
                if (cp >= 0x80) {
                    output.push_back(cp);
                    index += 2;
                    continue;
                }
            }
        // 3-byte sequence: 1110xxxx 10xxxxxx 10xxxxxx
        } else if ((b0 & 0xF0) == 0xE0 && index + 2 < input.size()) {
            const unsigned char b1 = static_cast<unsigned char>(input[index + 1]);
            const unsigned char b2 = static_cast<unsigned char>(input[index + 2]);
            if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80) {
                const char32_t cp = (static_cast<char32_t>(b0 & 0x0F) << 12) |
                                    (static_cast<char32_t>(b1 & 0x3F) << 6) |
                                    static_cast<char32_t>(b2 & 0x3F);
                // Reject overlong 3-byte form and UTF-16 surrogate code points.
                if (cp >= 0x800 && !(cp >= 0xD800 && cp <= 0xDFFF)) {
                    output.push_back(cp);
                    index += 3;
                    continue;
                }
            }
        // 4-byte sequence: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        } else if ((b0 & 0xF8) == 0xF0 && index + 3 < input.size()) {
            const unsigned char b1 = static_cast<unsigned char>(input[index + 1]);
            const unsigned char b2 = static_cast<unsigned char>(input[index + 2]);
            const unsigned char b3 = static_cast<unsigned char>(input[index + 3]);
            if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80 && (b3 & 0xC0) == 0x80) {
                const char32_t cp = (static_cast<char32_t>(b0 & 0x07) << 18) |
                                    (static_cast<char32_t>(b1 & 0x3F) << 12) |
                                    (static_cast<char32_t>(b2 & 0x3F) << 6) |
                                    static_cast<char32_t>(b3 & 0x3F);
                // Valid Unicode scalar range for 4-byte UTF-8.
                if (cp >= 0x10000 && cp <= 0x10FFFF) {
                    output.push_back(cp);
                    index += 4;
                    continue;
                }
            }
        }

        // Invalid leading byte or malformed/truncated sequence:
        // preserve raw byte so downstream logic remains non-throwing.
        output.push_back(static_cast<char32_t>(b0));
        ++index;
    }

    return output;
}

size_t utf8_codepoint_length(const std::string& input) {
    // Python parity: `len(ps)` counts Unicode code points, not UTF-8 bytes.
    return from_utf8(input).size();
}

std::string truncate_utf8_codepoints(const std::string& input, size_t max_codepoints) {
    // Python parity: `ps[:N]` slices by code points (characters), not raw bytes.
    // We decode to code points, clamp to N, then re-encode to valid UTF-8.
    const auto codepoints = from_utf8(input);
    const size_t limited = std::min(max_codepoints, codepoints.size());
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> convert;
    return convert.to_bytes(codepoints.data(), codepoints.data() + limited);
}

std::vector<std::string> split_by_newline_groups(const std::string& text) {
    // Python reference: `KPipeline` warns non-English chunking is not fully automatic;
    // users can split long text by '\n'. This helper preserves that usage pattern.
    std::vector<std::string> segments;
    std::string current;
    for (char c : text) {
        if (c == '\n' || c == '\r') {
            if (!current.empty()) {
                segments.push_back(current);
                current.clear();
            }
            continue;
        }
        current.push_back(c);
    }
    if (!current.empty()) {
        segments.push_back(current);
    }
    return segments;
}

std::vector<std::string> split_voice_list(const std::string& voice) {
    // Python parity: mirrors `KPipeline.load_voice(..., delimiter=",")` behavior,
    // where multiple voices can be requested and blended.
    std::vector<std::string> voices;
    std::string item;
    std::stringstream stream(voice);
    while (std::getline(stream, item, ',')) {
        auto begin = item.find_first_not_of(" \t");
        if (begin == std::string::npos) {
            continue;
        }
        auto end = item.find_last_not_of(" \t");
        voices.push_back(item.substr(begin, end - begin + 1));
    }
    return voices;
}

std::vector<std::filesystem::path> resolve_voice_file_candidates(const std::filesystem::path& models_path,
                                                                 const std::string& voice_id) {
    // Python reference: upstream `KPipeline.load_single_voice` resolves voices by id.
    // C++ resolves local .bin candidates deterministically instead of HF downloads.
    std::vector<std::filesystem::path> candidates;
    candidates.push_back(models_path / "voices" / (voice_id + ".bin"));

    if (const char* env_voices = std::getenv("KOKORO_VOICES_DIR")) {
        candidates.push_back(std::filesystem::path(env_voices) / (voice_id + ".bin"));
    }

    candidates.push_back(models_path.parent_path() / "kokoro" / "kokoro.js" / "voices" / (voice_id + ".bin"));
    candidates.push_back(models_path.parent_path().parent_path() / "kokoro" / "kokoro.js" / "voices" /
                         (voice_id + ".bin"));
    return candidates;
}

std::vector<float> load_voice_binary(const std::filesystem::path& models_path, const std::string& voice_id) {
    const auto candidates = resolve_voice_file_candidates(models_path, voice_id);
    for (const auto& candidate : candidates) {
        if (!std::filesystem::exists(candidate)) {
            continue;
        }

        std::ifstream file(candidate, std::ios::binary | std::ios::ate);
        OPENVINO_ASSERT(file.is_open(), "Failed to open Kokoro voice file: ", candidate);

        const std::streamsize size = file.tellg();
        OPENVINO_ASSERT(size > 0, "Kokoro voice file is empty: ", candidate);
        OPENVINO_ASSERT((size % static_cast<std::streamsize>(sizeof(float))) == 0,
                        "Kokoro voice file has invalid byte size: ", candidate);

        file.seekg(0, std::ios::beg);
        std::vector<float> data(static_cast<size_t>(size) / sizeof(float));
        file.read(reinterpret_cast<char*>(data.data()), size);
        OPENVINO_ASSERT(file.good(), "Failed to read Kokoro voice file: ", candidate);
        return data;
    }

    std::stringstream ss;
    ss << "Unable to find Kokoro voice '" << voice_id << "'. Checked:";
    for (const auto& path : candidates) {
        ss << "\n - " << path.string();
    }
    OPENVINO_THROW(ss.str());
}

#if OPENVINO_GENAI_HAS_MISAKI_CPP
void install_espeak_fallback_if_available(std::unique_ptr<misaki::G2P>& g2p,
                                          const std::string& language_variant) {
    if (!g2p) {
        return;
    }

    // Python parity: `KPipeline` wires `en.G2P(..., fallback=EspeakFallback(...))`
    // for English variants (`a`/`b` -> `en-us`/`en-gb`).
    const bool british = (language_variant == "en-gb");

    const char* library_hint = std::getenv("MISAKI_ESPEAK_LIBRARY");
    const char* version_hint = std::getenv("MISAKI_ESPEAK_VERSION");

    misaki::EspeakFallback espeak_fallback(british,
                                           version_hint ? std::string(version_hint) : std::string{},
                                           library_hint ? std::string(library_hint) : std::string{});
    if (!espeak_fallback.backend_available()) {
        std::cout << "Warning: espeak-ng fallback is not available for Kokoro G2P. Install espeak-ng and set "
                     "MISAKI_ESPEAK_LIBRARY to enable fallback support for out-of-vocab words."
                  << std::endl;
        return;
    }

    g2p->set_fallback_hook(espeak_fallback.as_hook());
}

std::vector<std::string> phonemize_single_text(misaki::G2P& g2p,
                                               const std::string& text,
                                               const ov::genai::SpeechGenerationConfig& generation_config) {
    auto rstrip_spaces = [](std::string value) {
        while (!value.empty() && value.back() == ' ') {
            value.pop_back();
        }
        return value;
    };

    auto lstrip_spaces = [](std::string value) {
        std::size_t first = 0;
        while (first < value.size() && value[first] == ' ') {
            ++first;
        }
        return value.substr(first);
    };

    auto tokens_to_ps = [&](const std::vector<misaki::MToken>& tokens) {
        std::string result;
        for (const auto& token : tokens) {
            result += token.phonemes.value_or("");
            if (!token.whitespace.empty()) {
                result.push_back(' ');
            }
        }
        return rstrip_spaces(result);
    };

    auto sanitize_utf8 = [](const std::string& value) {
        return truncate_utf8_codepoints(value, utf8_codepoint_length(value));
    };

    auto in_set = [](const std::string& value, const std::string& set_chars) {
        return value.size() == 1 && set_chars.find(value[0]) != std::string::npos;
    };

    auto waterfall_last = [&](const std::vector<misaki::MToken>& tokens, std::size_t next_count) {
        static const std::vector<std::string> waterfall = {"!.?…", ":;", ",—"};
        static const std::vector<std::string> bumps = {")", "”"};

        for (const auto& marks : waterfall) {
            std::optional<std::size_t> split_index;
            for (std::size_t i = tokens.size(); i-- > 0;) {
                if (in_set(tokens[i].phonemes.value_or(""), marks)) {
                    split_index = i;
                    break;
                }
            }
            if (!split_index.has_value()) {
                continue;
            }

            std::size_t z = *split_index + 1;
            if (z < tokens.size()) {
                const auto next_phonemes = tokens[z].phonemes.value_or("");
                for (const auto& bump : bumps) {
                    if (next_phonemes == bump) {
                        ++z;
                        break;
                    }
                }
            }

            std::vector<misaki::MToken> prefix(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(z));
            const auto prefix_len = utf8_codepoint_length(tokens_to_ps(prefix));
            if (next_count >= prefix_len && (next_count - prefix_len) <= generation_config.max_phoneme_length) {
                return z;
            }
        }

        return tokens.size();
    };

    const auto segments = split_by_newline_groups(text);
    std::vector<std::string> phoneme_chunks;

    for (const auto& segment : segments) {
        if (segment.empty()) {
            continue;
        }
        auto tokenized = g2p.phonemize_with_tokens(segment);
        std::vector<misaki::MToken> tks;
        std::size_t pcount = 0;

        for (const auto& token : tokenized.tokens) {
            // Python reference: equivalent to `KPipeline.tokens_to_ps` building
            // `phonemes + (' ' if whitespace else '')` per token.
            const std::string token_phonemes = token.phonemes.value_or("");
            const std::string suffix = token.whitespace.empty() ? "" : " ";
            std::string next_piece = token_phonemes + suffix;

            if (next_piece.empty()) {
                continue;
            }

            const std::size_t next_pcount = pcount + utf8_codepoint_length(rstrip_spaces(next_piece));

            if (next_pcount > generation_config.max_phoneme_length) {
                const std::size_t z = waterfall_last(tks, next_pcount);
                if (z > 0) {
                    std::vector<misaki::MToken> prefix(tks.begin(), tks.begin() + static_cast<std::ptrdiff_t>(z));
                    auto chunk = tokens_to_ps(prefix);
                    if (!chunk.empty()) {
                        phoneme_chunks.push_back(sanitize_utf8(chunk));
                    }
                    tks.erase(tks.begin(), tks.begin() + static_cast<std::ptrdiff_t>(z));
                    pcount = utf8_codepoint_length(tokens_to_ps(tks));
                    if (tks.empty()) {
                        next_piece = lstrip_spaces(next_piece);
                    }
                }
            }

            if (tks.empty() && utf8_codepoint_length(next_piece) > generation_config.max_phoneme_length) {
                std::string limited_piece = truncate_utf8_codepoints(next_piece, generation_config.max_phoneme_length);
                limited_piece = rstrip_spaces(limited_piece);
                if (!limited_piece.empty()) {
                    phoneme_chunks.push_back(sanitize_utf8(limited_piece));
                }
                continue;
            }

            if (next_piece.empty()) {
                continue;
            }

            auto next_token = token;
            next_token.phonemes = token_phonemes;
            tks.push_back(std::move(next_token));
            pcount += utf8_codepoint_length(next_piece);
        }

        if (!tks.empty()) {
            auto chunk = tokens_to_ps(tks);
            if (!chunk.empty()) {
                phoneme_chunks.push_back(sanitize_utf8(chunk));
            }
        }
    }

    OPENVINO_ASSERT(!phoneme_chunks.empty(), "Kokoro preprocessing produced no phoneme chunks for input text");
    return phoneme_chunks;
}
#endif

}  // namespace

namespace ov {
namespace genai {

class KokoroRuntime {
public:
    explicit KokoroRuntime(const std::filesystem::path& models_path)
        : m_models_path(models_path) {
        init_config();
    }

    const std::unordered_map<std::string, int64_t>& vocab() const {
        return m_vocab;
    }

    const std::string& language_variant() const {
        return m_language_variant;
    }

    uint32_t context_length() const {
        return m_context_length;
    }

    void set_language_variant(const std::string& language) {
        m_language_variant = normalize_language_variant(language);
    }

private:
    void init_config() {
        // Python reference: `KModel` keeps `self.vocab` and `context_length` from config.
        // C++ loads the same data from `config.json` for phoneme->id mapping and limits.
        const std::filesystem::path config_path = m_models_path / "config.json";
        std::ifstream file(config_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

        nlohmann::json data = nlohmann::json::parse(file);

        OPENVINO_ASSERT(data.contains("vocab") && data["vocab"].is_object(),
                        "Kokoro config must contain object 'vocab'");

        for (auto it = data["vocab"].begin(); it != data["vocab"].end(); ++it) {
            m_vocab[it.key()] = it.value().get<int64_t>();
        }

        if (data.contains("plbert") && data["plbert"].is_object() &&
            data["plbert"].contains("max_position_embeddings") &&
            data["plbert"]["max_position_embeddings"].is_number_unsigned()) {
            m_context_length = data["plbert"]["max_position_embeddings"].get<uint32_t>();
        }
    }

private:
    std::filesystem::path m_models_path;
    std::unordered_map<std::string, int64_t> m_vocab;
    std::string m_language_variant = "en-us";
    uint32_t m_context_length = 512;
};

KokoroTTSImpl::KokoroTTSImpl(const std::filesystem::path& models_path,
                             const std::string& device,
                             const ov::AnyMap& properties) {
#if !OPENVINO_GENAI_HAS_MISAKI_CPP
    (void)models_path;
    (void)device;
    (void)properties;
    OPENVINO_THROW("Kokoro backend requires misaki-cpp. Configure with ENABLE_MISAKI_CPP=ON and provide misaki-cpp sources.");
#else
    m_models_path = models_path;
    ov::Core core = ov::genai::utils::singleton_core();

    auto compiled = core.compile_model(models_path / "openvino_model.xml", device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled, "kokoro model");

    m_request = compiled.create_infer_request();

    m_runtime = std::make_shared<KokoroRuntime>(models_path);
    // Python reference: same engine family as `KPipeline(...).g2p` for English variants.
    m_g2p = misaki::make_engine("en", m_runtime->language_variant());
    install_espeak_fallback_if_available(m_g2p, m_runtime->language_variant());

    for (size_t idx = 0; idx < compiled.inputs().size(); ++idx) {
        const auto input_name = compiled.input(static_cast<int>(idx)).get_any_name();
        if (input_name.find("input") != std::string::npos || input_name.find("ids") != std::string::npos) {
            m_input_ids_name = input_name;
        } else if (input_name.find("ref") != std::string::npos || input_name.find("style") != std::string::npos) {
            m_ref_s_name = input_name;
        } else if (input_name.find("speed") != std::string::npos) {
            m_speed_name = input_name;
        }
    }

    if (m_input_ids_name.empty() && compiled.inputs().size() >= 1) {
        m_input_ids_name = compiled.input(0).get_any_name();
    }
    if (m_ref_s_name.empty() && compiled.inputs().size() >= 2) {
        m_ref_s_name = compiled.input(1).get_any_name();
    }
    if (m_speed_name.empty() && compiled.inputs().size() >= 3) {
        m_speed_name = compiled.input(2).get_any_name();
    }
#endif
}

Text2SpeechDecodedResults KokoroTTSImpl::generate(const std::vector<std::string>& texts,
                                                   const ov::Tensor& speaker_embedding,
                                                   const SpeechGenerationConfig& generation_config) {
#if !OPENVINO_GENAI_HAS_MISAKI_CPP
    (void)texts;
    (void)speaker_embedding;
    (void)generation_config;
    OPENVINO_THROW("Kokoro backend requires misaki-cpp. Configure with ENABLE_MISAKI_CPP=ON and provide misaki-cpp sources.");
#else
    // Python -> C++ quick map for side-by-side debugging:
    // - `KPipeline.__init__` (`ALIASES`/G2P selection) -> `normalize_language_variant` + `misaki::make_engine(...)`
    // - `KPipeline.en_tokenize` / `tokens_to_ps` -> newline segmentation + token loop building phoneme chunks
    // - `examples/export.py::ps[:510]` -> `truncate_utf8_codepoints(..., max_phoneme_length)`
    // - `KPipeline.load_voice` (including multi-voice averaging) -> `split_voice_list` + `load_voice_binary` + averaging
    // - `KModel.forward` phoneme->vocab->`[0, *ids, 0]` + length assert -> token-id build + context check below
    // - `KPipeline.infer` / `load_voice(pack[len(ps)-1])` -> `length_index` style-row selection
    // - `KModelForONNX.forward(input_ids, ref_s, speed)` -> OV request tensors (`input_ids`, `ref_s`, `speed`) + infer
    const bool has_external_speaker_embedding = static_cast<bool>(speaker_embedding);
    if (has_external_speaker_embedding) {
        OPENVINO_ASSERT(speaker_embedding.get_element_type() == ov::element::f32,
                        "Kokoro backend expects speaker_embedding element type f32");
    }

    const auto generation_start = std::chrono::steady_clock::now();

    const std::string language_variant = normalize_language_variant(generation_config.language);
    if (language_variant != m_runtime->language_variant()) {
        m_runtime->set_language_variant(language_variant);
        // Python parity: language change rebinds G2P behavior (see `KPipeline.__init__`).
        m_g2p = misaki::make_engine("en", language_variant);
        install_espeak_fallback_if_available(m_g2p, language_variant);
    }

    Text2SpeechDecodedResults result;

    for (const auto& text : texts) {
        std::vector<float> merged_audio;
        const auto text_tokenize_start = std::chrono::steady_clock::now();
        const auto phoneme_chunks = phonemize_single_text(*m_g2p, text, generation_config);

        const auto text_tokenize_end = std::chrono::steady_clock::now();
        result.perf_metrics.raw_metrics.tokenization_durations.emplace_back(
            MicroSeconds(text_tokenize_end - text_tokenize_start));

        ov::Shape speaker_shape;
        const float* external_speaker_ptr = nullptr;
        if (has_external_speaker_embedding) {
            speaker_shape = speaker_embedding.get_shape();
            OPENVINO_ASSERT(speaker_shape.size() == 2,
                            "Kokoro backend expects speaker_embedding shape [num_lengths, style_dim]");
            OPENVINO_ASSERT(speaker_shape[1] > 0, "Kokoro speaker_embedding second dimension must be > 0");
            external_speaker_ptr = speaker_embedding.data<const float>();
        }

        const auto& vocab = m_runtime->vocab();
        const uint32_t context_length = m_runtime->context_length();

        std::vector<std::string> requested_voices;
        std::vector<const std::vector<float>*> loaded_voices;
        if (!has_external_speaker_embedding) {
            OPENVINO_ASSERT(!generation_config.voice.empty(),
                            "Kokoro backend requires either speaker_embedding tensor or non-empty voice config");
            requested_voices = split_voice_list(generation_config.voice);
            OPENVINO_ASSERT(!requested_voices.empty(), "No valid Kokoro voice ids were parsed from voice config");

            for (const auto& voice_id : requested_voices) {
                auto it = m_voice_cache.find(voice_id);
                if (it == m_voice_cache.end()) {
                    it = m_voice_cache.emplace(voice_id, load_voice_binary(m_models_path, voice_id)).first;
                }
                OPENVINO_ASSERT((it->second.size() % 256) == 0,
                                "Kokoro voice binary must have float32 length divisible by 256 for voice: ",
                                voice_id);
                loaded_voices.push_back(&it->second);
            }

            const size_t rows = loaded_voices[0]->size() / 256;
            OPENVINO_ASSERT(rows > 0, "Kokoro voice binary has no rows for voice: ", requested_voices[0]);
            for (size_t idx = 1; idx < loaded_voices.size(); ++idx) {
                OPENVINO_ASSERT((loaded_voices[idx]->size() / 256) == rows,
                                "All Kokoro voices in a mixed request must have matching row count");
            }
            speaker_shape = ov::Shape{rows, 256};
            // Python parity: if multiple voices are provided, equivalent to
            // `torch.mean(torch.stack(packs), dim=0)` in `KPipeline.load_voice`.
        }

        for (const auto& phonemes : phoneme_chunks) {
            std::vector<int64_t> token_ids;
            token_ids.reserve(phonemes.size() + 2);
            token_ids.push_back(0);

            const auto codepoints = from_utf8(phonemes);
            for (char32_t cp : codepoints) {
                const std::string key = to_utf8(cp);
                auto it = vocab.find(key);
                if (it != vocab.end()) {
                    token_ids.push_back(it->second);
                }
            }
            token_ids.push_back(0);

            // Python parity: mirrors `KModel.forward` path:
            // `input_ids = [0, *mapped_vocab_ids, 0]` and context-length assertion.

            OPENVINO_ASSERT(token_ids.size() <= context_length,
                            "Kokoro tokenized length exceeds model context length: ", token_ids.size(),
                            " > ", context_length);

            const size_t num_tokens = token_ids.size() >= 2 ? token_ids.size() - 2 : 0;
            const size_t length_index = std::min<size_t>(num_tokens > 0 ? num_tokens - 1 : 0,
                                                         speaker_shape[0] > 0 ? speaker_shape[0] - 1 : 0);

            // Python parity: equivalent to selecting `pack[len(ps) - 1]` in
            // `KPipeline.infer` / `examples/export.py::load_voice`.

            std::vector<float> style_slice(speaker_shape[1]);
            const size_t offset = length_index * speaker_shape[1];
            if (has_external_speaker_embedding) {
                std::copy(external_speaker_ptr + offset,
                          external_speaker_ptr + offset + speaker_shape[1],
                          style_slice.data());
            } else {
                std::fill(style_slice.begin(), style_slice.end(), 0.0f);
                for (const auto* voice_data : loaded_voices) {
                    std::transform(style_slice.begin(),
                                   style_slice.end(),
                                   voice_data->begin() + offset,
                                   style_slice.begin(),
                                   std::plus<float>());
                }
                const float norm = 1.0f / static_cast<float>(loaded_voices.size());
                for (auto& value : style_slice) {
                    value *= norm;
                }
            }

            ov::Tensor input_ids_tensor(ov::element::i64,
                                        ov::Shape{1, token_ids.size()},
                                        token_ids.data());
            ov::Tensor ref_s_tensor(ov::element::f32,
                                    ov::Shape{1, speaker_shape[1]},
                                    style_slice.data());
            ov::Tensor speed_tensor(ov::element::f32, ov::Shape{1}, &generation_config.speed);

            m_request.set_tensor(m_input_ids_name, input_ids_tensor);
            m_request.set_tensor(m_ref_s_name, ref_s_tensor);
            if (!m_speed_name.empty()) {
                m_request.set_tensor(m_speed_name, speed_tensor);
            }

            m_request.infer();

            ov::Tensor waveform = m_request.get_output_tensor(0);
            const float* waveform_data = waveform.data<const float>();
            merged_audio.insert(merged_audio.end(), waveform_data, waveform_data + waveform.get_size());
        }

        ov::Tensor speech(ov::element::f32, ov::Shape{merged_audio.size()});
        std::copy(merged_audio.begin(), merged_audio.end(), speech.data<float>());
        result.perf_metrics.num_generated_samples += speech.get_size();
        result.speeches.push_back(speech);
    }

    result.perf_metrics.raw_metrics.generate_durations.emplace_back(
        MicroSeconds(std::chrono::steady_clock::now() - generation_start));
    result.perf_metrics.evaluate_statistics();
    m_perf_metrics = result.perf_metrics;
    return result;
#endif
}

std::vector<std::vector<std::string>> KokoroTTSImpl::phonemize(const std::vector<std::string>& texts,
                                                               const SpeechGenerationConfig& generation_config) {
#if !OPENVINO_GENAI_HAS_MISAKI_CPP
    (void)texts;
    (void)generation_config;
    OPENVINO_THROW("Kokoro backend requires misaki-cpp. Configure with ENABLE_MISAKI_CPP=ON and provide misaki-cpp sources.");
#else
    const std::string language_variant = normalize_language_variant(generation_config.language);
    if (language_variant != m_runtime->language_variant()) {
        m_runtime->set_language_variant(language_variant);
        m_g2p = misaki::make_engine("en", language_variant);
        install_espeak_fallback_if_available(m_g2p, language_variant);
    }

    std::vector<std::vector<std::string>> all_chunks;
    all_chunks.reserve(texts.size());
    for (const auto& text : texts) {
        all_chunks.push_back(phonemize_single_text(*m_g2p, text, generation_config));
    }
    return all_chunks;
#endif
}

}  // namespace genai
}  // namespace ov
