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
#endif

namespace {

std::string normalize_language_variant(const std::string& language) {
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
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> convert;
    return convert.from_bytes(input);
}

size_t utf8_codepoint_length(const std::string& input) {
    return from_utf8(input).size();
}

std::string truncate_utf8_codepoints(const std::string& input, size_t max_codepoints) {
    const auto codepoints = from_utf8(input);
    const size_t limited = std::min(max_codepoints, codepoints.size());
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> convert;
    return convert.to_bytes(codepoints.data(), codepoints.data() + limited);
}

std::vector<std::string> split_by_newline_groups(const std::string& text) {
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
    m_g2p = misaki::make_engine("en", m_runtime->language_variant());

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
    const bool has_external_speaker_embedding = static_cast<bool>(speaker_embedding);
    if (has_external_speaker_embedding) {
        OPENVINO_ASSERT(speaker_embedding.get_element_type() == ov::element::f32,
                        "Kokoro backend expects speaker_embedding element type f32");
    }

    const auto generation_start = std::chrono::steady_clock::now();

    const std::string language_variant = normalize_language_variant(generation_config.language);
    if (language_variant != m_runtime->language_variant()) {
        m_runtime->set_language_variant(language_variant);
        m_g2p = misaki::make_engine("en", language_variant);
    }

    Text2SpeechDecodedResults result;

    for (const auto& text : texts) {
        std::vector<float> merged_audio;
        const auto text_tokenize_start = std::chrono::steady_clock::now();

        const auto segments = split_by_newline_groups(text);
        std::vector<std::string> phoneme_chunks;

        for (const auto& segment : segments) {
            if (segment.empty()) {
                continue;
            }

            auto tokenized = m_g2p->phonemize_with_tokens(segment);
            std::string current_chunk;

            for (const auto& token : tokenized.tokens) {
                const std::string token_phonemes = token.phonemes.value_or("");
                const std::string suffix = token.whitespace.empty() ? "" : " ";
                const std::string next_piece = token_phonemes + suffix;

                if (next_piece.empty()) {
                    continue;
                }

                if (utf8_codepoint_length(current_chunk) + utf8_codepoint_length(next_piece) >
                    generation_config.max_phoneme_length) {
                    if (!current_chunk.empty()) {
                        while (!current_chunk.empty() && current_chunk.back() == ' ') {
                            current_chunk.pop_back();
                        }
                        if (!current_chunk.empty()) {
                            phoneme_chunks.push_back(current_chunk);
                        }
                        current_chunk.clear();
                    }

                    std::string limited_piece = next_piece;
                    if (utf8_codepoint_length(limited_piece) > generation_config.max_phoneme_length) {
                        limited_piece = truncate_utf8_codepoints(limited_piece, generation_config.max_phoneme_length);
                    }
                    while (!limited_piece.empty() && limited_piece.back() == ' ') {
                        limited_piece.pop_back();
                    }
                    if (!limited_piece.empty()) {
                        phoneme_chunks.push_back(limited_piece);
                    }
                    continue;
                }

                current_chunk += next_piece;
            }

            while (!current_chunk.empty() && current_chunk.back() == ' ') {
                current_chunk.pop_back();
            }
            if (!current_chunk.empty()) {
                phoneme_chunks.push_back(current_chunk);
            }
        }

        const auto text_tokenize_end = std::chrono::steady_clock::now();
        result.perf_metrics.raw_metrics.tokenization_durations.emplace_back(
            MicroSeconds(text_tokenize_end - text_tokenize_start));

        OPENVINO_ASSERT(!phoneme_chunks.empty(), "Kokoro preprocessing produced no phoneme chunks for input text");

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

            OPENVINO_ASSERT(token_ids.size() <= context_length,
                            "Kokoro tokenized length exceeds model context length: ", token_ids.size(),
                            " > ", context_length);

            const size_t num_tokens = token_ids.size() >= 2 ? token_ids.size() - 2 : 0;
            const size_t length_index = std::min<size_t>(num_tokens > 0 ? num_tokens - 1 : 0,
                                                         speaker_shape[0] > 0 ? speaker_shape[0] - 1 : 0);

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

}  // namespace genai
}  // namespace ov
