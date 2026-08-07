// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "piper_tts_model.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "openvino/genai/perf_metrics.hpp"
#include "utils.hpp"
#include "logger.hpp"

#if OPENVINO_GENAI_HAS_MISAKI_CPP
#include "misaki/fallbacks.hpp"
#endif

namespace ov {
namespace genai {

namespace {

void set_default_property(ov::AnyMap& config, const std::string& key, const ov::Any& value) {
    if (config.count(key) == 0) {
        config.insert({key, value});
    }
}

ov::CompiledModel compile_piper_model(ov::Core& core,
                                      const std::filesystem::path& model_path,
                                      const std::string& device,
                                      const ov::AnyMap& properties) {
    ov::AnyMap compile_properties = properties;
    if (device.find("GPU") != std::string::npos) {
        set_default_property(compile_properties, "INFERENCE_PRECISION_HINT", ov::element::f32);
    }
    return core.compile_model(model_path, device, compile_properties);
}

std::string find_input_name(const ov::CompiledModel& model, const std::string& substring, size_t fallback_index) {
    for (const auto& input : model.inputs()) {
        const std::string name = input.get_any_name();
        if (name.find(substring) != std::string::npos) {
            return name;
        }
    }
    OPENVINO_ASSERT(model.inputs().size() > fallback_index,
                    "Piper model is missing an expected input matching '",
                    substring,
                    "'");
    return model.input(static_cast<int>(fallback_index)).get_any_name();
}

// Splits a UTF-8 string into individual Unicode codepoints, each returned as its own
// UTF-8-encoded string. Piper's phoneme_id_map keys are single codepoints (including
// combining diacritics, which attach to the preceding base character as a *separate*
// sequence element rather than being composed).
std::vector<std::string> split_into_codepoints(const std::string& utf8_text) {
    std::vector<std::string> codepoints;
    size_t index = 0;
    while (index < utf8_text.size()) {
        const unsigned char lead_byte = static_cast<unsigned char>(utf8_text[index]);
        size_t char_length = 1;
        if ((lead_byte & 0x80) == 0x00) {
            char_length = 1;
        } else if ((lead_byte & 0xE0) == 0xC0) {
            char_length = 2;
        } else if ((lead_byte & 0xF0) == 0xE0) {
            char_length = 3;
        } else if ((lead_byte & 0xF8) == 0xF0) {
            char_length = 4;
        }
        char_length = std::min(char_length, utf8_text.size() - index);
        codepoints.push_back(utf8_text.substr(index, char_length));
        index += char_length;
    }
    return codepoints;
}

}  // namespace

PiperTTSImpl::PiperTTSImpl(const std::filesystem::path& models_path,
                           const std::string& device,
                           const ov::AnyMap& properties) {
    const auto load_start = std::chrono::steady_clock::now();
    m_models_path = models_path;

    const std::filesystem::path config_path = models_path / "config.json";
    std::ifstream config_file(config_path);
    OPENVINO_ASSERT(config_file.is_open(), "Failed to open Piper config at ", config_path);

    nlohmann::json data = nlohmann::json::parse(config_file);
    OPENVINO_ASSERT(data.contains("phoneme_id_map") && data["phoneme_id_map"].is_object(),
                    "Piper config.json at ",
                    config_path,
                    " is missing the required 'phoneme_id_map' object");

    for (auto it = data["phoneme_id_map"].begin(); it != data["phoneme_id_map"].end(); ++it) {
        OPENVINO_ASSERT(it.value().is_array() && !it.value().empty(),
                        "Piper phoneme_id_map entry '",
                        it.key(),
                        "' must be a non-empty array");
        m_phoneme_id_map[it.key()] = it.value()[0].get<int64_t>();
    }

    if (data.contains("sample_rate") && data["sample_rate"].is_number_unsigned()) {
        m_sample_rate = data["sample_rate"].get<uint32_t>();
    }
    if (data.contains("language") && data["language"].is_string()) {
        m_espeak_voice = data["language"].get<std::string>();
    }

    ov::Core core = ov::genai::utils::singleton_core();
    auto compiled = compile_piper_model(core, models_path / "openvino_model.xml", device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled, "piper model");

    m_input_name = find_input_name(compiled, "input", 0);
    m_input_lengths_name = find_input_name(compiled, "length", 1);
    m_scales_name = find_input_name(compiled, "scales", 2);

    m_request = compiled.create_infer_request();
    save_load_time(load_start);
}

std::vector<int64_t> PiperTTSImpl::build_phoneme_id_sequence(const std::string& text) const {
    auto id_of = [this](const std::string& symbol) -> int64_t {
        const auto it = m_phoneme_id_map.find(symbol);
        OPENVINO_ASSERT(it != m_phoneme_id_map.end(), "Piper phoneme_id_map has no entry for symbol '", symbol, "'");
        return it->second;
    };

    const int64_t pad_id = id_of("_");
    const int64_t bos_id = id_of("^");
    const int64_t eos_id = id_of("$");

    const std::vector<std::string> codepoints = split_into_codepoints(text);

    std::vector<int64_t> sequence;
    sequence.reserve(codepoints.size() * 2 + 3);
    sequence.push_back(bos_id);
    sequence.push_back(pad_id);
    for (const std::string& codepoint : codepoints) {
        const auto it = m_phoneme_id_map.find(codepoint);
        if (it == m_phoneme_id_map.end()) {
            // Skip phonemes absent from this voice's map (for example stray espeak
            // language-switch artifacts) rather than aborting the whole utterance.
            continue;
        }
        sequence.push_back(it->second);
        sequence.push_back(pad_id);
    }
    sequence.push_back(eos_id);
    return sequence;
}

Text2SpeechDecodedResults PiperTTSImpl::generate(const std::vector<std::string>& texts,
                                                 const ov::Tensor& speaker_embedding,
                                                 const SpeechGenerationConfig& generation_config) {
    OPENVINO_ASSERT(!static_cast<bool>(speaker_embedding) || speaker_embedding.get_size() == 0,
                    "Piper backend is single-speaker and does not accept a speaker_embedding tensor.");

#if OPENVINO_GENAI_HAS_MISAKI_CPP
    const char* library_hint = std::getenv("MISAKI_ESPEAK_LIBRARY");
    const std::string library_path = library_hint ? std::string(library_hint) : std::string{};
    const std::string espeak_voice = generation_config.language.empty() ? m_espeak_voice : generation_config.language;
#endif

    Text2SpeechDecodedResults result;
    result.output_sample_rate = m_sample_rate;
    const auto generation_start = std::chrono::steady_clock::now();

    const std::array<float, 3> scales = {generation_config.noise_scale,
                                         generation_config.length_scale,
                                         generation_config.noise_w};

    for (const std::string& text : texts) {
#if OPENVINO_GENAI_HAS_MISAKI_CPP
        const auto phonemized = misaki::raw_espeak_ipa_phonemize(text, espeak_voice, library_path);
        OPENVINO_ASSERT(phonemized.has_value(),
                        "Piper G2P failed to phonemize input text with espeak-ng voice '",
                        espeak_voice,
                        "'. Install espeak-ng and/or set MISAKI_ESPEAK_LIBRARY.");
        const std::vector<int64_t> phoneme_ids = build_phoneme_id_sequence(*phonemized);
#else
        OPENVINO_THROW(
            "Piper backend requires misaki-cpp for espeak-ng based G2P. Configure with ENABLE_MISAKI_CPP=ON and "
            "provide misaki-cpp sources.");
        const std::vector<int64_t> phoneme_ids;
#endif

        const int64_t input_length = static_cast<int64_t>(phoneme_ids.size());
        ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, phoneme_ids.size()}, const_cast<int64_t*>(phoneme_ids.data()));
        ov::Tensor input_lengths_tensor(ov::element::i64, ov::Shape{1}, const_cast<int64_t*>(&input_length));
        ov::Tensor scales_tensor(ov::element::f32, ov::Shape{3}, const_cast<float*>(scales.data()));

        m_request.set_tensor(m_input_name, input_tensor);
        m_request.set_tensor(m_input_lengths_name, input_lengths_tensor);
        m_request.set_tensor(m_scales_name, scales_tensor);

        m_request.infer();

        const ov::Tensor waveform = m_request.get_output_tensor(0);
        const float* waveform_data = waveform.data<const float>();
        const size_t sample_count = waveform.get_size();

        ov::Tensor speech(ov::element::f32, ov::Shape{sample_count});
        std::copy(waveform_data, waveform_data + sample_count, speech.data<float>());
        result.perf_metrics.num_generated_samples += speech.get_size();
        result.speeches.push_back(speech);
    }

    result.perf_metrics.raw_metrics.generate_durations.emplace_back(
        MicroSeconds(std::chrono::steady_clock::now() - generation_start));
    result.perf_metrics.evaluate_statistics();
    m_perf_metrics = result.perf_metrics;
    return result;
}

}  // namespace genai
}  // namespace ov
