// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <variant>
#include <vector>

#include "openvino/core/except.hpp"
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
    const size_t length = shape[1];
    const size_t vocab_size = shape[2];

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

std::string extract_predicted_language(const std::string& text,
                                       const std::map<std::string, int64_t>& lid_dict) {
    const std::string open = "<|";
    const std::string close = "|>";
    if (text.rfind(open, 0) != 0) {
        return "";
    }
    const size_t end = text.find(close, open.size());
    if (end == std::string::npos) {
        return "";
    }
    const std::string tag = text.substr(open.size(), end - open.size());
    if (tag == "auto" || tag == "nospeech" || lid_dict.find(tag) == lid_dict.end()) {
        return "";
    }
    return tag;
}

}  // namespace

namespace ov::genai {

SenseVoiceSmall::SenseVoiceSmall(const std::filesystem::path& models_path,
                                 const std::string& device,
                                 const ov::AnyMap& properties)
    : ASRPipelineImplBase(models_path),
      m_feature_extractor(models_path / "am.mvn"),
      m_config(models_path / "config.json") {
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

    const auto detokenization_start_time = std::chrono::steady_clock::now();
    const std::string text = m_tokenizer.decode(token_ids);
    const auto detokenization_stop_time = std::chrono::steady_clock::now();
    results.texts.push_back(text);
    results.perf_metrics.raw_metrics.detokenization_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(detokenization_stop_time - detokenization_start_time)));

    results.scores.push_back(0.0f);
    const std::string result_language = (config.language.has_value() && config.language.value() != "auto")
                                            ? config.language.value()
                                            : extract_predicted_language(text, m_config.lid_dict);
    results.languages.push_back(result_language);

    if (streamer) {
        streamer->write(token_ids);
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
