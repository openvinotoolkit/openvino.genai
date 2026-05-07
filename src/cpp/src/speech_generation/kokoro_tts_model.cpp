// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "kokoro_tts_model.hpp"

#include <algorithm>
#include <chrono>
#include <codecvt>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <locale>
#include <map>
#include <numeric>
#include <optional>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "openvino/genai/perf_metrics.hpp"
#include "utils.hpp"
#include "logger.hpp"

#if OPENVINO_GENAI_HAS_MISAKI_CPP
#include "misaki/g2p.hpp"
#include "misaki/fallbacks.hpp"

std::optional<std::filesystem::path> resolve_ov_fallback_model_dir(
    const ov::genai::SpeechGenerationConfig& generation_config) {
    if (!generation_config.phonemize_fallback_model_dir.has_value()) {
        return std::nullopt;
    }
    return std::filesystem::path(*generation_config.phonemize_fallback_model_dir);
}

static inline void dump_compiled_model_inputs_outputs(const ov::CompiledModel& model) {
    const auto inputs = model.inputs();
    const auto outputs = model.outputs();

    GENAI_INFO("\tInputs:");
    for (const auto& input : inputs) {
        const std::string name = input.get_any_name();
        const ov::element::Type type = input.get_element_type();
        const ov::PartialShape shape = input.get_partial_shape();
        const ov::Layout layout = ov::layout::get_layout(input);

        std::ostringstream log_message;
        log_message << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string();
        const std::string log_line = log_message.str();

        GENAI_INFO("%s", log_line.c_str());
    }

    GENAI_INFO("\tOutputs:");
    for (const auto& output : outputs) {
        const std::string name = output.get_any_name();
        const ov::element::Type type = output.get_element_type();
        const ov::PartialShape shape = output.get_partial_shape();
        const ov::Layout layout = ov::layout::get_layout(output);

        std::ostringstream log_message;
        log_message << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string();
        const std::string log_line = log_message.str();

        GENAI_INFO("%s", log_line.c_str());
    }
}
class OpenVINOFallbackNetwork {
public:
    OpenVINOFallbackNetwork(const std::filesystem::path& model_dir, ov::Core& core) {
        const auto config_path = model_dir / "config.json";
        const auto encoder_path = model_dir / "openvino_encoder_model.xml";
        const auto decoder_path = model_dir / "openvino_decoder_model.xml";

        OPENVINO_ASSERT(std::filesystem::exists(config_path), "Missing fallback config.json at ", config_path);
        OPENVINO_ASSERT(std::filesystem::exists(encoder_path), "Missing fallback encoder model at ", encoder_path);
        OPENVINO_ASSERT(std::filesystem::exists(decoder_path), "Missing fallback decoder model at ", decoder_path);

        std::ifstream config_file(config_path);
        OPENVINO_ASSERT(config_file.is_open(), "Failed to open fallback config at ", config_path);

        nlohmann::json cfg;
        config_file >> cfg;

        m_bos_token_id = cfg.value("bos_token_id", 1);
        m_decoder_start_token_id = cfg.value("decoder_start_token_id", m_bos_token_id);
        m_eos_token_id = cfg.value("eos_token_id", 2);
        m_unk_token_id = cfg.value("unk_token_id", 3);
        m_max_decode_length = static_cast<size_t>(std::max<int64_t>(8, cfg.value("max_position_embeddings", 64)));

        const std::u32string graphemes = utf8_to_u32(cfg.at("grapheme_chars").get<std::string>());
        for (size_t index = 0; index < graphemes.size(); ++index) {
            m_grapheme_to_token[graphemes[index]] = static_cast<int64_t>(index);
        }

        const std::u32string phonemes = utf8_to_u32(cfg.at("phoneme_chars").get<std::string>());
        m_token_to_phoneme.reserve(phonemes.size());
        for (char32_t cp : phonemes) {
            m_token_to_phoneme.push_back(u32_to_utf8(cp));
        }

        // These fallback BART models are typically *very* small (< 1M params),
        // so we just force them to use CPU for now.
        auto encoder_compiled_model = core.compile_model(encoder_path, "CPU");
        auto decoder_compiled_model = core.compile_model(decoder_path, "CPU");

        if (ov::genai::get_cur_log_level() >= ov::log::Level::INFO) {
            GENAI_INFO("Fallback encoder model info:");
            dump_compiled_model_inputs_outputs(encoder_compiled_model);
            GENAI_INFO("Fallback decoder model info:");
            dump_compiled_model_inputs_outputs(decoder_compiled_model);
        }

        OPENVINO_ASSERT(encoder_compiled_model.inputs().size() >= 2,
                        "Unexpected fallback encoder signature in ",
                        encoder_path);
        OPENVINO_ASSERT(decoder_compiled_model.inputs().size() >= 3,
                        "Unexpected fallback decoder signature in ",
                        decoder_path);

        m_encoder_input_ids_index = find_input_index(encoder_compiled_model, "input_ids").value_or(0);
        m_encoder_attention_mask_index = find_input_index(encoder_compiled_model, "attention_mask").value_or(1);

        m_decoder_input_ids_index = find_input_index(decoder_compiled_model, "input_ids").value_or(0);
        m_decoder_attention_mask_index = find_input_index(decoder_compiled_model, "encoder_attention_mask").value_or(1);
        const auto encoder_state_index = find_input_index(decoder_compiled_model, "encoder_hidden_states");

        OPENVINO_ASSERT(encoder_state_index.has_value(), "Fallback decoder is missing encoder_hidden_states input");
        m_decoder_encoder_state_index = *encoder_state_index;

        m_encoder_request = encoder_compiled_model.create_infer_request();
        m_decoder_request = decoder_compiled_model.create_infer_request();
    }

    std::optional<std::string> infer_token(const misaki::MToken& token) {
        if (token.text.empty()) {
            return std::nullopt;
        }

        const auto graphemes = utf8_to_u32(token.text);
        std::vector<int64_t> input_ids;
        input_ids.reserve(graphemes.size() + 2);
        input_ids.push_back(m_bos_token_id);
        for (char32_t cp : graphemes) {
            const auto it = m_grapheme_to_token.find(cp);
            input_ids.push_back(it == m_grapheme_to_token.end() ? m_unk_token_id : it->second);
        }
        input_ids.push_back(m_eos_token_id);

        std::vector<int64_t> attention_mask(input_ids.size(), 1);

        ov::Tensor encoder_input_ids(ov::element::i64, ov::Shape{1, input_ids.size()}, input_ids.data());
        ov::Tensor encoder_attention_mask(ov::element::i64, ov::Shape{1, attention_mask.size()}, attention_mask.data());

        m_encoder_request.set_input_tensor(m_encoder_input_ids_index, encoder_input_ids);
        m_encoder_request.set_input_tensor(m_encoder_attention_mask_index, encoder_attention_mask);
        m_encoder_request.infer();
        const ov::Tensor encoder_hidden_states = m_encoder_request.get_output_tensor(0);

        m_decoder_request.set_input_tensor(m_decoder_attention_mask_index, encoder_attention_mask);
        m_decoder_request.set_input_tensor(m_decoder_encoder_state_index, encoder_hidden_states);

        std::vector<int64_t> decoder_tokens = {m_decoder_start_token_id};
        for (size_t step = 0; step < m_max_decode_length; ++step) {
            ov::Tensor decoder_input_ids(ov::element::i64, ov::Shape{1, decoder_tokens.size()}, decoder_tokens.data());
            m_decoder_request.set_input_tensor(m_decoder_input_ids_index, decoder_input_ids);
            m_decoder_request.infer();

            const ov::Tensor logits = m_decoder_request.get_output_tensor(0);
            OPENVINO_ASSERT(logits.get_element_type() == ov::element::f32,
                            "Fallback decoder logits are expected to have f32 element type");

            const auto logits_shape = logits.get_shape();
            OPENVINO_ASSERT(logits_shape.size() == 3,
                            "Fallback decoder logits are expected to have rank 3 [batch, seq, vocab]");
            OPENVINO_ASSERT(logits_shape[0] == 1, "Fallback decoder logits batch size must be 1");

            const size_t seq_len = logits_shape[1];
            const size_t vocab_size = logits_shape[2];
            const float* logits_ptr = logits.data<const float>();
            const size_t offset = (seq_len - 1) * vocab_size;

            // simple greedy decoding step: index of highest logit in the last time step is the next token id
            int64_t next_id = 0;
            float max_logit = logits_ptr[offset];
            for (size_t vocab_index = 1; vocab_index < vocab_size; ++vocab_index) {
                const float score = logits_ptr[offset + vocab_index];
                if (score > max_logit) {
                    max_logit = score;
                    next_id = static_cast<int64_t>(vocab_index);
                }
            }

            if (next_id == m_eos_token_id) {
                break;
            }
            decoder_tokens.push_back(next_id);
        }

        std::string phonemes;
        for (int64_t token_id : decoder_tokens) {
            if (token_id <= 3) {
                continue;
            }
            if (token_id >= 0 && static_cast<size_t>(token_id) < m_token_to_phoneme.size()) {
                phonemes += m_token_to_phoneme[static_cast<size_t>(token_id)];
            }
        }
        if (phonemes.empty()) {
            return std::nullopt;
        }
        return phonemes;
    }

private:
    static std::u32string utf8_to_u32(const std::string& input) {
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> convert;
        try {
            return convert.from_bytes(input);
        } catch (...) {
            std::u32string output;
            output.reserve(input.size());
            for (unsigned char c : input) {
                output.push_back(static_cast<char32_t>(c));
            }
            return output;
        }
    }

    static std::string u32_to_utf8(char32_t codepoint) {
        std::u32string input(1, codepoint);
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> convert;
        return convert.to_bytes(input);
    }

    static std::optional<size_t> find_input_index(const ov::CompiledModel& model, const std::string& expected_name) {
        const auto& inputs = model.inputs();
        for (size_t index = 0; index < inputs.size(); ++index) {
            const std::string& name = inputs[index].get_any_name();
            if (name.find(expected_name) != std::string::npos) {
                return index;
            }
        }
        return std::nullopt;
    }

    ov::InferRequest m_encoder_request;
    ov::InferRequest m_decoder_request;
    std::unordered_map<char32_t, int64_t> m_grapheme_to_token;
    std::vector<std::string> m_token_to_phoneme;
    int64_t m_bos_token_id = 1;
    int64_t m_decoder_start_token_id = 1;
    int64_t m_eos_token_id = 2;
    int64_t m_unk_token_id = 3;
    size_t m_max_decode_length = 64;
    size_t m_encoder_input_ids_index = 0;
    size_t m_encoder_attention_mask_index = 1;
    size_t m_decoder_input_ids_index = 0;
    size_t m_decoder_attention_mask_index = 1;
    size_t m_decoder_encoder_state_index = 2;
};

bool has_required_misaki_lexicon_files(const std::filesystem::path& root) {
    return std::filesystem::exists(root / "us_gold.json") &&
           std::filesystem::exists(root / "us_silver.json") &&
           std::filesystem::exists(root / "gb_gold.json") &&
           std::filesystem::exists(root / "gb_silver.json");
}

void configure_misaki_lexicon_data_root_from_model_dir(const std::filesystem::path& models_path) {
    const auto model_lexicon_root = models_path / "data";
    if (!has_required_misaki_lexicon_files(model_lexicon_root)) {
        return;
    }

    misaki::set_english_lexicon_data_root(model_lexicon_root.string());
}
#endif

namespace {

void set_default_property(ov::AnyMap& config, const std::string& key, const ov::Any& value) {
    if (config.count(key) == 0) {
        config.insert({key, value});
    }
}

ov::CompiledModel compile_kokoro_model(ov::Core& core,
                                       const std::filesystem::path& model_path,
                                       const std::string& device,
                                       const ov::AnyMap& properties,
                                       const bool npu_requested,
                                       const size_t static_input_ids_length) {
    ov::AnyMap compile_properties = properties;

    if (device.find("GPU") != std::string::npos) {
        set_default_property(compile_properties, "INFERENCE_PRECISION_HINT", ov::element::f32);
    }

    if (!npu_requested) {
        return core.compile_model(model_path, device, compile_properties);
    }

    // In the case of NPU, set some NPUW properties, and reshape to static.

    set_default_property(compile_properties, "NPU_USE_NPUW", std::string{"YES"});
    set_default_property(compile_properties, "NPUW_DEVICES", std::string{"NPU,CPU"});
    set_default_property(compile_properties, "NPUW_KOKORO", std::string{"YES"});

    auto model = core.read_model(model_path, {}, compile_properties);

    std::map<std::string, ov::PartialShape> static_shapes;
    if (model->inputs().size() >= 1) {
        static_shapes.emplace(model->input(0).get_any_name(), ov::PartialShape{1, static_cast<int64_t>(static_input_ids_length)});
    }
    if (model->inputs().size() >= 2) {
        static_shapes.emplace(model->input(1).get_any_name(), ov::PartialShape{1, 256});
    }
    if (model->inputs().size() >= 3) {
        static_shapes.emplace(model->input(2).get_any_name(), ov::PartialShape{1});
    }

    if (!static_shapes.empty()) {
        model->reshape(static_shapes);
    }

    return core.compile_model(model, device, compile_properties);
}

double sum_tensor_prefix_as_double(const ov::Tensor& tensor, const size_t count) {
    const size_t n = std::min(count, tensor.get_size());
    if (n == 0) {
        return 0.0;
    }

    OPENVINO_ASSERT(tensor.get_element_type() == ov::element::i64,
                    "Expected pred_dur tensor to have element type == i64");

    double sum = 0.0;
    const auto* ptr = tensor.data<const int64_t>();
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(ptr[i]);
    }
    return sum;
}

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

    if (normalized == "e" || normalized == "es") {
        return "es";
    }
    if (normalized == "f" || normalized == "fr-fr") {
        return "fr-fr";
    }
    if (normalized == "h" || normalized == "hi") {
        return "hi";
    }
    if (normalized == "i" || normalized == "it") {
        return "it";
    }
    if (normalized == "p" || normalized == "pt-br") {
        return "pt-br";
    }

    OPENVINO_THROW("Unsupported Kokoro language '",
                   language,
                   "'. Supported values: en-us, en-gb, es, fr-fr, hi, it, pt-br (aliases: a, b, e, f, h, i, p)");
}

bool is_english_variant(const std::string& language_variant) {
    return language_variant == "en-us" || language_variant == "en-gb";
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

std::vector<std::string> split_non_english_chunks(const std::string& graphemes, const size_t chunk_size = 400) {
    std::vector<std::string> chunks;
    if (graphemes.empty()) {
        return chunks;
    }

    std::vector<std::string> sentences;
    std::string current;
    auto is_sentence_punct = [](char c) {
        return c == '.' || c == '!' || c == '?';
    };

    for (size_t i = 0; i < graphemes.size(); ++i) {
        const char c = graphemes[i];
        current.push_back(c);
        if (is_sentence_punct(c)) {
            while (i + 1 < graphemes.size() && is_sentence_punct(graphemes[i + 1])) {
                current.push_back(graphemes[++i]);
            }
            sentences.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        sentences.push_back(current);
    }

    std::string chunk;
    for (const auto& sentence : sentences) {
        if (chunk.size() + sentence.size() <= chunk_size) {
            chunk += sentence;
        } else {
            if (!chunk.empty()) {
                chunks.push_back(chunk);
            }
            chunk = sentence;
        }
    }
    if (!chunk.empty()) {
        chunks.push_back(chunk);
    }

    if (!chunks.empty()) {
        return chunks;
    }

    for (size_t i = 0; i < graphemes.size(); i += chunk_size) {
        chunks.push_back(graphemes.substr(i, chunk_size));
    }
    return chunks;
}

#if OPENVINO_GENAI_HAS_MISAKI_CPP
void install_fallback_if_available(std::unique_ptr<misaki::G2P>& g2p,
                                   const std::string& language_variant,
                                   const ov::genai::SpeechGenerationConfig& generation_config) {
    if (!g2p) {
        return;
    }

    // Python parity: `KPipeline` wires `en.G2P(..., fallback=EspeakFallback(...))`
    // for English variants (`a`/`b` -> `en-us`/`en-gb`).
    const bool british = (language_variant == "en-gb");

    // Reset any previously installed fallback hook before applying the new policy.
    // This guarantees failed backend selection does not retain stale fallback behavior.
    g2p->set_fallback_hook({});

    if (generation_config.phonemize_fallback_model_dir.has_value()) {
        try {
            if (const auto model_dir = resolve_ov_fallback_model_dir(generation_config); model_dir.has_value()) {
                ov::Core core = ov::genai::utils::singleton_core();
                auto ov_fallback = std::make_shared<OpenVINOFallbackNetwork>(*model_dir, core);
                g2p->set_fallback_hook([ov_fallback](const misaki::MToken& token) {
                    return ov_fallback->infer_token(token);
                });
                return;
            }
        } catch (const std::exception& error) {
            OPENVINO_THROW("Failed to initialize OpenVINO fallback network for Kokoro G2P at '",
                           *generation_config.phonemize_fallback_model_dir,
                           "': ",
                           error.what());
        }
    }

    const char* library_hint = std::getenv("MISAKI_ESPEAK_LIBRARY");
    const char* version_hint = std::getenv("MISAKI_ESPEAK_VERSION");

    misaki::EspeakFallback espeak_fallback(british,
                                           version_hint ? std::string(version_hint) : std::string{},
                                           library_hint ? std::string(library_hint) : std::string{});
    if (!espeak_fallback.backend_available()) {
        GENAI_WARN("espeak-ng fallback is not available for Kokoro G2P. Install espeak-ng and optionally set "
                   "MISAKI_ESPEAK_LIBRARY to enable fallback support for out-of-vocab words.");
        return;
    }

    g2p->set_fallback_hook(espeak_fallback.as_hook());
}

std::vector<std::string> phonemize_single_text(misaki::G2P& g2p,
                                               const std::string& text,
                                               const ov::genai::SpeechGenerationConfig& generation_config,
                                               const std::string& language_variant) {
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

    auto matches_any = [](const std::string& value, const std::vector<std::string>& candidates) {
        return std::find(candidates.begin(), candidates.end(), value) != candidates.end();
    };

    auto waterfall_last = [&](const std::vector<misaki::MToken>& tokens, std::size_t next_count) {
        static const std::vector<std::vector<std::string>> waterfall = {{"!", ".", "?", "…"},
                                                                         {":", ";"},
                                                                         {",", "—"}};
        static const std::vector<std::string> bumps = {")", "”"};

        for (const auto& marks : waterfall) {
            std::optional<std::size_t> split_index;
            for (std::size_t i = tokens.size(); i-- > 0;) {
                if (matches_any(tokens[i].phonemes.value_or(""), marks)) {
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

    if (!is_english_variant(language_variant)) {
        for (const auto& segment : segments) {
            if (segment.empty()) {
                continue;
            }

            const auto chunks = split_non_english_chunks(segment);
            for (const auto& chunk : chunks) {
                if (chunk.empty()) {
                    continue;
                }

                std::string ps = g2p.phonemize(chunk);
                if (ps.empty()) {
                    continue;
                }

                if (utf8_codepoint_length(ps) > generation_config.max_phoneme_length) {
                    ps = truncate_utf8_codepoints(ps, generation_config.max_phoneme_length);
                }

                if (!ps.empty()) {
                    phoneme_chunks.push_back(sanitize_utf8(ps));
                }
            }
        }

        return phoneme_chunks;
    }

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
    const bool npu_requested = device == "NPU";

    m_runtime = std::make_shared<KokoroRuntime>(models_path);
    configure_misaki_lexicon_data_root_from_model_dir(models_path);

    if (npu_requested) {
        m_static_input_ids_length = m_runtime->context_length();
    }

    auto compiled = compile_kokoro_model(core,
                                         models_path / "openvino_model.xml",
                                         device,
                                         properties,
                                         npu_requested,
                                         m_static_input_ids_length);
    ov::genai::utils::print_compiled_model_properties(compiled, "kokoro model");
    m_has_pred_dur_output = compiled.outputs().size() > 1;

    m_request = compiled.create_infer_request();

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

#if OPENVINO_GENAI_HAS_MISAKI_CPP
void KokoroTTSImpl::ensure_g2p_initialized(const SpeechGenerationConfig& generation_config) {
    const std::string language_variant = normalize_language_variant(generation_config.language);

    const bool requires_new_engine = !m_g2p || language_variant != m_runtime->language_variant();
    if (requires_new_engine) {
        m_runtime->set_language_variant(language_variant);
        // Python parity: language change rebinds G2P behavior (see `KPipeline.__init__`).
        if (is_english_variant(language_variant)) {
            m_g2p = misaki::make_engine("en", language_variant);
        } else {
            m_g2p = misaki::make_engine("espeak", language_variant);
        }
        // Ensure fallback policy is applied for a newly created engine.
        m_fallback_initialized = false;
    }

    const bool fallback_config_changed = !m_fallback_initialized ||
                                         generation_config.phonemize_fallback_model_dir != m_phonemize_fallback_model_dir;
    if (is_english_variant(language_variant) && (requires_new_engine || fallback_config_changed)) {
        install_fallback_if_available(m_g2p, language_variant, generation_config);
        m_fallback_initialized = true;
        m_phonemize_fallback_model_dir = generation_config.phonemize_fallback_model_dir;
    } else if (!is_english_variant(language_variant)) {
        m_fallback_initialized = false;
        m_phonemize_fallback_model_dir.reset();
    }
}
#endif

Text2SpeechDecodedResults KokoroTTSImpl::generate(const std::vector<std::string>& texts,
                                                   const ov::Tensor& speaker_embedding,
                                                   const SpeechGenerationConfig& generation_config) {
#if !OPENVINO_GENAI_HAS_MISAKI_CPP
    (void)texts;
    (void)speaker_embedding;
    (void)generation_config;
    OPENVINO_THROW("Kokoro backend requires misaki-cpp. Configure with ENABLE_MISAKI_CPP=ON and provide misaki-cpp sources.");
#else
    ensure_g2p_initialized(generation_config);
    const std::string language_variant = normalize_language_variant(generation_config.language);
    if (!is_english_variant(language_variant) && !m_g2p->backend_available()) {
        const auto backend_error = m_g2p->backend_error();
        OPENVINO_THROW("Kokoro non-English text generation requires espeak-ng, but backend is unavailable for language '",
                       language_variant,
                       "'. Install espeak-ng and/or set MISAKI_ESPEAK_LIBRARY. ",
                       backend_error.has_value() ? std::string("Details: ") + *backend_error : std::string{});
    }

    std::vector<std::vector<std::string>> all_phoneme_chunks;
    all_phoneme_chunks.reserve(texts.size());
    for (const auto& text : texts) {
        auto phoneme_chunks = phonemize_single_text(*m_g2p, text, generation_config, language_variant);
        all_phoneme_chunks.push_back(std::move(phoneme_chunks));
    }

    auto result = synthesize_from_phoneme_chunks(all_phoneme_chunks, speaker_embedding, generation_config);
    m_perf_metrics = result.perf_metrics;
    return result;
#endif
}

Text2SpeechDecodedResults KokoroTTSImpl::synthesize_from_phoneme_chunks(
    const std::vector<std::vector<std::string>>& all_phoneme_chunks,
    const ov::Tensor& speaker_embedding,
    const SpeechGenerationConfig& generation_config) {
#if !OPENVINO_GENAI_HAS_MISAKI_CPP
    (void)all_phoneme_chunks;
    (void)speaker_embedding;
    (void)generation_config;
    OPENVINO_THROW("Kokoro backend requires misaki-cpp. Configure with ENABLE_MISAKI_CPP=ON and provide misaki-cpp sources.");
#else
    OPENVINO_ASSERT(static_cast<bool>(speaker_embedding),
                    "Kokoro backend requires speaker_embedding tensor. Prepare the embedding in the application "
                    "and pass the final ov::Tensor to generate().");
    OPENVINO_ASSERT(speaker_embedding.get_element_type() == ov::element::f32,
                    "Kokoro backend expects speaker_embedding element type f32");

    Text2SpeechDecodedResults result;
    result.output_sample_rate = 24000;
    const auto generation_start = std::chrono::steady_clock::now();

    const ov::Shape speaker_shape = speaker_embedding.get_shape();
    OPENVINO_ASSERT(speaker_shape.size() == 3 &&
                    speaker_shape[1] == 1 &&
                    speaker_shape[2] == 256,
                    "Kokoro backend expects speaker_embedding shape [num_lengths, 1, 256]. "
                    "Typical voice packs have num_lengths=510 (one style row per phoneme sequence length 1-510).");
    OPENVINO_ASSERT(speaker_shape[0] > 0, "Kokoro speaker_embedding: num_lengths must be > 0");
    const float* external_speaker_ptr = speaker_embedding.data<const float>();

    const auto& vocab = m_runtime->vocab();
    const uint32_t context_length = m_runtime->context_length();

    for (const auto& phoneme_chunks : all_phoneme_chunks) {
        std::vector<float> merged_audio;

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
            const size_t text_len = token_ids.size();

            OPENVINO_ASSERT(token_ids.size() <= context_length,
                            "Kokoro tokenized length exceeds model context length: ", token_ids.size(),
                            " > ", context_length);

            const size_t phoneme_length = codepoints.size();
            const size_t length_index = std::min<size_t>(phoneme_length > 0 ? phoneme_length - 1 : 0,
                                                         speaker_shape[0] > 0 ? speaker_shape[0] - 1 : 0);

            // Each row in the [num_lengths, 1, 256] pack is 1*256 = 256 floats.
            // Create ref_s_tensor as a zero-copy view into the speaker_embedding memory.
            // speaker_embedding (and thus external_speaker_ptr) outlives the infer() call.
            constexpr size_t style_dim = 256;
            const size_t offset = length_index * style_dim;

            std::vector<int64_t> static_token_ids;
            ov::Tensor input_ids_tensor;
            if (m_static_input_ids_length > 0) {
                OPENVINO_ASSERT(token_ids.size() <= m_static_input_ids_length,
                                "Kokoro tokenized length exceeds static input length: ", token_ids.size(),
                                " > ", m_static_input_ids_length);
                static_token_ids.assign(m_static_input_ids_length, 16);
                std::copy(token_ids.begin(), token_ids.end(), static_token_ids.begin());
                input_ids_tensor = ov::Tensor(ov::element::i64,
                                              ov::Shape{1, m_static_input_ids_length},
                                              static_token_ids.data());
            } else {
                input_ids_tensor = ov::Tensor(ov::element::i64,
                                              ov::Shape{1, token_ids.size()},
                                              token_ids.data());
            }
            ov::Tensor ref_s_tensor(ov::element::f32,
                                    ov::Shape{1, style_dim},
                                    external_speaker_ptr + offset);
            ov::Tensor speed_tensor(ov::element::f32, ov::Shape{1}, &generation_config.speed);

            m_request.set_tensor(m_input_ids_name, input_ids_tensor);
            m_request.set_tensor(m_ref_s_name, ref_s_tensor);
            if (!m_speed_name.empty()) {
                m_request.set_tensor(m_speed_name, speed_tensor);
            }

            m_request.infer();

            ov::Tensor waveform = m_request.get_output_tensor(0);
            const float* waveform_data = waveform.data<const float>();
            size_t samples_to_keep = waveform.get_size();
            if (m_static_input_ids_length > 0 && text_len < m_static_input_ids_length && m_has_pred_dur_output) {
                ov::Tensor pred_dur = m_request.get_output_tensor(1);
                const size_t pred_dur_len = pred_dur.get_size();
                const double total_dur = sum_tensor_prefix_as_double(pred_dur, pred_dur_len);
                const size_t valid_len = std::min(text_len, pred_dur_len);
                const double valid_dur = sum_tensor_prefix_as_double(pred_dur, valid_len);

                if (total_dur > 0.0) {
                    const double ratio = valid_dur / total_dur;
                    samples_to_keep = std::min<size_t>(waveform.get_size(),
                                                       static_cast<size_t>(static_cast<double>(waveform.get_size()) * ratio));
                }
            }

            merged_audio.insert(merged_audio.end(), waveform_data, waveform_data + samples_to_keep);
        }

        ov::Tensor speech(ov::element::f32, ov::Shape{merged_audio.size()});
        std::copy(merged_audio.begin(), merged_audio.end(), speech.data<float>());
        result.perf_metrics.num_generated_samples += speech.get_size();
        result.speeches.push_back(speech);
    }

    result.perf_metrics.raw_metrics.generate_durations.emplace_back(
        MicroSeconds(std::chrono::steady_clock::now() - generation_start));
    result.perf_metrics.evaluate_statistics();
    return result;
#endif
}

ov::Shape KokoroTTSImpl::get_speaker_embedding_shape() const {
    // Matches the native Kokoro voice pack shape: 510 length-indexed rows,
    // one per possible phoneme sequence length (1-510), each a [1, 256] style vector.
    return ov::Shape{510, 1, 256};
}

}  // namespace genai
}  // namespace ov
