// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speecht5_tts_model.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <openvino/openvino.hpp>
#include <variant>

#include "default_speaker_embedding.hpp"
#include "json_utils.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "speecht5_tts_decoder.hpp"
#include "utils.hpp"

namespace {

ov::InferRequest init_model(const std::filesystem::path& models_path,
                            const std::string& model_file_name,
                            const std::string& model_name,
                            const std::string& device,
                            const ov::AnyMap& properties) {
    ov::Core core = ov::genai::utils::singleton_core();

    auto compiled = core.compile_model(models_path / model_file_name, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled, model_name.c_str());
    ov::InferRequest request = compiled.create_infer_request();

    try {
        ov::RemoteContext context = compiled.get_context();
        for (size_t out_idx = 0; out_idx < compiled.outputs().size(); ++out_idx) {
            ov::Shape output_shape = request.get_output_tensor(out_idx).get_shape();
            ov::element::Type output_type = request.get_output_tensor(out_idx).get_element_type();
            ov::RemoteTensor remote = context.create_tensor(output_type, output_shape);
            request.set_output_tensor(out_idx, remote);
        }
        return request;
    } catch (const ov::Exception&) {
        return request;
    }
}

std::tuple<ov::Tensor, ov::Tensor> encode(ov::InferRequest& request,
                                          const ov::Tensor& input_ids,
                                          ov::genai::RawPerfMetrics& raw_metrics) {
    request.set_tensor("input_ids", input_ids);

    const auto infer_start = std::chrono::steady_clock::now();
    request.infer();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);

    auto last_hidden_state = request.get_tensor("last_hidden_state");
    auto encoder_attention_mask = request.get_tensor("encoder_attention_mask");
    return std::make_tuple(last_hidden_state, encoder_attention_mask);
}

ov::Tensor postnet(ov::InferRequest& request,
                   const ov::Tensor& raw_spectrogram,
                   ov::genai::RawPerfMetrics& raw_metrics) {
    request.set_tensor("raw_spectrogram", raw_spectrogram);

    const auto infer_start = std::chrono::steady_clock::now();
    request.infer();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);

    auto postnet_spectrogram = request.get_tensor("postnet_spectrogram");
    return postnet_spectrogram;
}

ov::Tensor vocoder(ov::InferRequest& request, const ov::Tensor& spectrogram, ov::genai::RawPerfMetrics& raw_metrics) {
    request.set_tensor("spectrogram", spectrogram);

    const auto infer_start = std::chrono::steady_clock::now();
    request.infer();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);

    auto waveform = request.get_tensor("waveform");
    return waveform;
}

const ov::Tensor get_default_speaker_embedding() {
    return ov::Tensor(ov::element::f32,
                      ov::Shape{1, 512},
                      reinterpret_cast<float*>(ov::genai::default_speaker_embedding_bytes));
}

}  // namespace

namespace ov {
namespace genai {

SpeechT5TTSImpl::SpeechT5TTSImpl(const std::filesystem::path& models_path,
                                 const std::string& device,
                                 const ov::AnyMap& properties,
                                 const Tokenizer& tokenizer)
    : m_tokenizer(tokenizer),
      m_reduction_factor(2),
      m_num_mel_bins(80) {
    init_model_config_params(models_path);

    m_encoder = init_model(models_path, "openvino_encoder_model.xml", "speecht5_tts encoder model", device, properties);
    m_postnet = init_model(models_path, "openvino_postnet.xml", "speecht5_tts postnet model", device, properties);
    m_vocoder = init_model(models_path, "openvino_vocoder.xml", "speecht5_tts vocoder model", device, properties);

    m_decoder = std::make_shared<SpeechT5TTSDecoder>(models_path, device, properties);
}

void SpeechT5TTSImpl::init_model_config_params(const std::filesystem::path& root_dir) {
    const std::filesystem::path model_index_path = root_dir / "config.json";
    std::ifstream file(model_index_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using ov::genai::utils::read_json_param;

    if (data.contains("reduction_factor") && data["reduction_factor"].is_number_unsigned()) {
        m_reduction_factor = data["reduction_factor"];
    }

    if (data.contains("num_mel_bins") && data["num_mel_bins"].is_number_unsigned()) {
        m_num_mel_bins = data["num_mel_bins"];
    }
}

Text2SpeechDecodedResults SpeechT5TTSImpl::generate(const std::vector<std::string>& texts,
                                                    const ov::Tensor& speaker_embedding,
                                                    const SpeechGenerationConfig& generation_config) {
    const ov::Tensor& used_speaker_embedding = speaker_embedding ? speaker_embedding : get_default_speaker_embedding();

    Text2SpeechDecodedResults gen_speech_res;

    auto& tokenization_durations = gen_speech_res.perf_metrics.raw_metrics.tokenization_durations;
    const auto generation_start = std::chrono::steady_clock::now();
    for (const auto& text : texts) {
        const auto tokenization_start = std::chrono::steady_clock::now();
        auto tokens = m_tokenizer.encode(text);
        const auto tokenization_end = std::chrono::steady_clock::now();
        tokenization_durations.emplace_back(PerfMetrics::get_microsec(tokenization_end - tokenization_start));
        auto input_ids = tokens.input_ids;
        uint64_t bsz = 1;  // process per batch

        const auto infer_start = std::chrono::steady_clock::now();
        RawPerfMetrics raw_perf_metrics;
        auto [last_hidden_state, encoder_attention_mask] = encode(m_encoder, input_ids, raw_perf_metrics);

        auto last_hidden_state_len = static_cast<float>(last_hidden_state.get_shape()[1]);
        auto reduction_factor = static_cast<float>(m_reduction_factor);

        int64_t maxlen = static_cast<int64_t>(last_hidden_state_len * generation_config.maxlenratio / reduction_factor);
        int64_t minlen = static_cast<int64_t>(last_hidden_state_len * generation_config.minlenratio / reduction_factor);

        // prepare inputs for decoder
        std::vector<float> zeros(bsz * 1 * m_num_mel_bins, 0.0f);
        ov::Tensor inputs_embeds(ov::element::f32, ov::Shape{bsz, 1, m_num_mel_bins}, zeros.data());
        ov::Tensor spectrogram(ov::element::f32, ov::Shape{0, bsz, 2, m_num_mel_bins}, std::vector<float>{}.data());

        int64_t iter = 0;
        // decoder loop
        while (true) {
            iter += 1;
            m_decoder->start_async(inputs_embeds,
                                   used_speaker_embedding,
                                   last_hidden_state,
                                   encoder_attention_mask,
                                   spectrogram);
            auto [out_seq, spectrum, prob, spectrogram_out] = m_decoder->wait();
            inputs_embeds = out_seq;
            spectrogram = spectrogram_out;

            if (iter < minlen) {
                continue;
            }

            // if the generation loop is less than maximum length time, check the ones in the batch that have met
            // the prob threshold.Otherwise, assume all have met thresholds and fill other spectrograms for the batch.
            auto prob_values = prob.data<float>();
            float prob_sum = std::accumulate(prob_values, prob_values + prob.get_size(), 0.0f);
            bool found_spectrum = ((prob_sum >= generation_config.threshold) || (iter >= maxlen));

            if (found_spectrum) {
                // refine spectrogram using postnet
                auto postnet_spectrogram = postnet(m_postnet, spectrogram, raw_perf_metrics);
                spectrogram = postnet_spectrogram;
                break;
            }
        }
        auto waveform = vocoder(m_vocoder, spectrogram, raw_perf_metrics);

        gen_speech_res.perf_metrics.num_generated_samples += waveform.get_size();

        gen_speech_res.speeches.push_back(waveform);
        m_decoder->reset_state();
    }

    const auto generation_end = std::chrono::steady_clock::now();
    gen_speech_res.perf_metrics.raw_metrics.generate_durations.emplace_back(
        PerfMetrics::get_microsec(generation_end - generation_start));

    gen_speech_res.perf_metrics.evaluate_statistics();
    m_perf_metrics = gen_speech_res.perf_metrics;
    return gen_speech_res;
}

SpeechGenerationPerfMetrics SpeechT5TTSImpl::get_performance_metrics() {
    return m_perf_metrics;
}

}  // namespace genai
}  // namespace ov
