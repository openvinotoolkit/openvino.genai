// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "paraformer_impl.hpp"
#include "utils.hpp"

#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

namespace ov {
namespace genai {

// ============================================================================
// ParaformerFeatureExtractor Implementation
// ============================================================================

ParaformerFeatureExtractor::ParaformerFeatureExtractor(int sample_rate, 
                                                         int n_mels,
                                                         int n_fft,
                                                         int hop_length,
                                                         int win_length,
                                                         int lfr_m,
                                                         int lfr_n)
    : m_sample_rate(sample_rate)
    , m_n_mels(n_mels)
    , m_n_fft(n_fft)
    , m_hop_length(hop_length)
    , m_win_length(win_length)
    , m_lfr_m(lfr_m)
    , m_lfr_n(lfr_n) {
    init_mel_filters();
}

void ParaformerFeatureExtractor::init_mel_filters() {
    // Initialize mel filterbank
    // Frequency to mel conversion: mel = 2595 * log10(1 + f/700)
    auto hz_to_mel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
    auto mel_to_hz = [](float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); };
    
    float mel_low = hz_to_mel(0.0f);
    float mel_high = hz_to_mel(static_cast<float>(m_sample_rate) / 2.0f);
    
    // Create mel points
    std::vector<float> mel_points(m_n_mels + 2);
    for (int i = 0; i <= m_n_mels + 1; ++i) {
        mel_points[i] = mel_low + i * (mel_high - mel_low) / (m_n_mels + 1);
    }
    
    // Convert back to Hz and then to FFT bin indices
    std::vector<int> bin_points(m_n_mels + 2);
    for (int i = 0; i <= m_n_mels + 1; ++i) {
        float hz = mel_to_hz(mel_points[i]);
        bin_points[i] = static_cast<int>(std::floor((m_n_fft + 1) * hz / m_sample_rate));
    }
    
    // Create triangular filters
    m_mel_filters.resize(m_n_mels);
    int fft_bins = m_n_fft / 2 + 1;
    for (int i = 0; i < m_n_mels; ++i) {
        m_mel_filters[i].resize(fft_bins, 0.0f);
        for (int j = bin_points[i]; j < bin_points[i + 1]; ++j) {
            if (j >= 0 && j < fft_bins) {
                m_mel_filters[i][j] = static_cast<float>(j - bin_points[i]) / 
                                      (bin_points[i + 1] - bin_points[i]);
            }
        }
        for (int j = bin_points[i + 1]; j < bin_points[i + 2]; ++j) {
            if (j >= 0 && j < fft_bins) {
                m_mel_filters[i][j] = static_cast<float>(bin_points[i + 2] - j) / 
                                      (bin_points[i + 2] - bin_points[i + 1]);
            }
        }
    }
}

std::vector<float> ParaformerFeatureExtractor::hann_window(int length) {
    std::vector<float> window(length);
    for (int i = 0; i < length; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (length - 1)));
    }
    return window;
}

std::vector<float> ParaformerFeatureExtractor::apply_preemphasis(
    const std::vector<float>& audio, float coeff) {
    std::vector<float> result(audio.size());
    result[0] = audio[0];
    for (size_t i = 1; i < audio.size(); ++i) {
        result[i] = audio[i] - coeff * audio[i - 1];
    }
    return result;
}

std::vector<std::vector<float>> ParaformerFeatureExtractor::compute_stft(
    const std::vector<float>& audio) {
    // Simple DFT implementation (for small window sizes)
    // In production, this should use FFT library
    auto window = hann_window(m_win_length);
    int num_frames = (static_cast<int>(audio.size()) - m_win_length) / m_hop_length + 1;
    int fft_bins = m_n_fft / 2 + 1;
    
    std::vector<std::vector<float>> stft(num_frames, std::vector<float>(fft_bins, 0.0f));
    
    for (int frame = 0; frame < num_frames; ++frame) {
        int start = frame * m_hop_length;
        
        // Compute magnitude spectrum for this frame
        for (int k = 0; k < fft_bins; ++k) {
            float real_sum = 0.0f, imag_sum = 0.0f;
            for (int n = 0; n < m_win_length && (start + n) < static_cast<int>(audio.size()); ++n) {
                float windowed_sample = audio[start + n] * window[n];
                float angle = -2.0f * M_PI * k * n / m_n_fft;
                real_sum += windowed_sample * std::cos(angle);
                imag_sum += windowed_sample * std::sin(angle);
            }
            stft[frame][k] = real_sum * real_sum + imag_sum * imag_sum;  // Power spectrum
        }
    }
    
    return stft;
}

ov::Tensor ParaformerFeatureExtractor::extract(const std::vector<float>& audio, int sample_rate) {
    // TODO: Add resampling if sample_rate != m_sample_rate
    
    // Apply pre-emphasis
    auto preemphasized = apply_preemphasis(audio);
    
    // Compute STFT
    auto power_spectrum = compute_stft(preemphasized);
    
    int num_frames = static_cast<int>(power_spectrum.size());
    
    // Apply mel filterbank and take log
    std::vector<float> mel_features(num_frames * m_n_mels);
    for (int frame = 0; frame < num_frames; ++frame) {
        for (int mel = 0; mel < m_n_mels; ++mel) {
            float mel_energy = 0.0f;
            for (size_t bin = 0; bin < m_mel_filters[mel].size(); ++bin) {
                mel_energy += m_mel_filters[mel][bin] * power_spectrum[frame][bin];
            }
            // Log mel energy with floor to avoid log(0)
            mel_features[frame * m_n_mels + mel] = std::log(std::max(mel_energy, 1e-10f));
        }
    }
    
    // Apply LFR (Low Frame Rate) stacking
    // Stack m_lfr_m consecutive frames, skip m_lfr_n frames between stacks
    int num_lfr_frames = (num_frames - m_lfr_m) / m_lfr_n + 1;
    int lfr_feat_dim = m_n_mels * m_lfr_m;
    std::vector<float> lfr_features(num_lfr_frames * lfr_feat_dim);
    
    for (int lfr_frame = 0; lfr_frame < num_lfr_frames; ++lfr_frame) {
        int start_frame = lfr_frame * m_lfr_n;
        for (int m = 0; m < m_lfr_m; ++m) {
            int src_frame = start_frame + m;
            if (src_frame < num_frames) {
                for (int mel = 0; mel < m_n_mels; ++mel) {
                    lfr_features[lfr_frame * lfr_feat_dim + m * m_n_mels + mel] = 
                        mel_features[src_frame * m_n_mels + mel];
                }
            } else {
                // Pad with zeros if we run out of frames
                for (int mel = 0; mel < m_n_mels; ++mel) {
                    lfr_features[lfr_frame * lfr_feat_dim + m * m_n_mels + mel] = 0.0f;
                }
            }
        }
    }
    
    // Create tensor [1, num_lfr_frames, lfr_feat_dim] (e.g., [1, ?, 560])
    ov::Shape shape = {1, static_cast<size_t>(num_lfr_frames), static_cast<size_t>(lfr_feat_dim)};
    return ov::Tensor(ov::element::f32, shape, lfr_features.data());
}

// ============================================================================
// ParaformerDetokenizer Implementation
// ============================================================================

bool ParaformerDetokenizer::load(const std::filesystem::path& tokens_path) {
    std::ifstream file(tokens_path);
    if (!file.is_open()) {
        return false;
    }

    try {
        nlohmann::json tokens_json;
        file >> tokens_json;

        for (auto& [key, value] : tokens_json.items()) {
            try {
                int64_t id = std::stoll(key);
                std::string token = value.get<std::string>();
                m_vocab[id] = token;
                
                // Detect special tokens
                if (token == "<blank>" || token == "<pad>") {
                    m_blank_id = id;
                } else if (token == "<s>" || token == "<sos>") {
                    m_sos_id = id;
                } else if (token == "</s>" || token == "<eos>") {
                    m_eos_id = id;
                }
            } catch (...) {
                continue;
            }
        }
        m_loaded = true;
        return true;
    } catch (const std::exception&) {
        m_loaded = false;
        return false;
    }
}

std::string ParaformerDetokenizer::decode(const std::vector<int64_t>& token_ids) const {
    if (!m_loaded) {
        return "";
    }

    std::string result;
    for (auto id : token_ids) {
        // Skip special tokens
        if (id == m_blank_id || id == m_sos_id || id == m_eos_id) {
            continue;
        }
        
        auto it = m_vocab.find(id);
        if (it != m_vocab.end()) {
            result += it->second;
        }
    }
    return result;
}

// ============================================================================
// ParaformerImpl Implementation
// ============================================================================

ParaformerImpl::ParaformerImpl(const std::filesystem::path& model_dir,
                               const std::string& device,
                               const ov::AnyMap& properties)
    : ASRPipelineImplBase(model_dir, device, properties) {
    auto start = std::chrono::steady_clock::now();

    // Load the OpenVINO model
    auto model_path = model_dir / "openvino_model.xml";
    OPENVINO_ASSERT(std::filesystem::exists(model_path),
                    "ParaformerImpl: model file not found: ", model_path);

    ov::Core core;
    auto model = core.read_model(model_path);
    m_compiled_model = core.compile_model(model, device, properties);

    // Initialize feature extractor
    m_feature_extractor = std::make_unique<ParaformerFeatureExtractor>();

    // Load tokens / initialize detokenizer
    auto tokens_path = model_dir / "tokens.json";
    if (std::filesystem::exists(tokens_path)) {
        load_tokens(tokens_path);
        m_detokenizer.load(tokens_path);
    }

    // Initialize default generation config
    m_generation_config.max_new_tokens = 448;

    auto end = std::chrono::steady_clock::now();
    m_load_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
}

ParaformerImpl::~ParaformerImpl() = default;

void ParaformerImpl::load_tokens(const std::filesystem::path& tokens_path) {
    std::ifstream file(tokens_path);
    if (!file.is_open()) {
        return;
    }

    try {
        nlohmann::json tokens_json;
        file >> tokens_json;

        size_t max_id = 0;
        for (auto& [key, value] : tokens_json.items()) {
            try {
                size_t id = std::stoul(key);
                if (id > max_id) max_id = id;
            } catch (...) {
                continue;
            }
        }

        m_tokens.resize(max_id + 1);
        for (auto& [key, value] : tokens_json.items()) {
            try {
                size_t id = std::stoul(key);
                m_tokens[id] = value.get<std::string>();
            } catch (...) {
                continue;
            }
        }
        m_has_tokens = true;
    } catch (const std::exception&) {
        m_has_tokens = false;
    }
}

std::string ParaformerImpl::decode_ids(const std::vector<int64_t>& ids) const {
    // Prefer detokenizer if loaded
    if (m_detokenizer.is_loaded()) {
        return m_detokenizer.decode(ids);
    }
    
    // Fallback to legacy token map
    if (!m_has_tokens) {
        return "";
    }

    std::string result;
    for (auto id : ids) {
        if (id >= 0 && static_cast<size_t>(id) < m_tokens.size()) {
            result += m_tokens[id];
        }
    }
    return result;
}

std::vector<int64_t> ParaformerImpl::ctc_decode(const ov::Tensor& logits) {
    auto output_shape = logits.get_shape();
    if (output_shape.size() < 2) {
        return {};
    }
    
    auto* data = logits.data<float>();
    size_t seq_len = output_shape[1];
    size_t vocab_size = output_shape.back();
    
    std::vector<int64_t> token_ids;
    int64_t prev_id = -1;
    
    for (size_t t = 0; t < seq_len; ++t) {
        // Find argmax
        float max_val = data[t * vocab_size];
        int64_t max_id = 0;
        for (size_t v = 1; v < vocab_size; ++v) {
            float val = data[t * vocab_size + v];
            if (val > max_val) {
                max_val = val;
                max_id = static_cast<int64_t>(v);
            }
        }
        // CTC collapse: skip blanks (id=0) and repeated tokens
        if (max_id != 0 && max_id != prev_id) {
            token_ids.push_back(max_id);
        }
        prev_id = max_id;
    }
    
    return token_ids;
}

ASRDecodedResults ParaformerImpl::generate(
    const RawSpeechInput& raw_speech_input,
    const ASRGenerationConfig& config,
    const std::shared_ptr<StreamerBase> streamer) {

    ASRDecodedResults results;

    // RawSpeechInput is std::vector<float>, use it directly
    const std::vector<float>& audio_data = raw_speech_input;

    if (audio_data.empty()) {
        results.texts.push_back("");
        return results;
    }

    // Extract features from raw audio
    ov::Tensor speech_tensor = m_feature_extractor->extract(audio_data, 16000);
    auto speech_shape = speech_tensor.get_shape();
    
    // Create speech_lengths tensor
    std::vector<int32_t> lengths = {static_cast<int32_t>(speech_shape[1])};
    ov::Tensor lengths_tensor(ov::element::i32, {1}, lengths.data());

    // Run inference
    auto infer_request = m_compiled_model.create_infer_request();
    auto inputs = m_compiled_model.inputs();
    
    if (inputs.size() >= 2) {
        infer_request.set_input_tensor(0, speech_tensor);
        infer_request.set_input_tensor(1, lengths_tensor);
    } else {
        infer_request.set_input_tensor(0, speech_tensor);
    }

    infer_request.infer();

    // Get output and decode
    auto output_tensor = infer_request.get_output_tensor(0);
    auto token_ids = ctc_decode(output_tensor);
    
    std::string text = decode_ids(token_ids);
    results.texts.push_back(text);

    // Stream tokens if streamer provided
    if (streamer) {
        // For CTC models, we can stream the final result
        // (Token-level streaming would require different approach)
    }

    return results;
}

Tokenizer ParaformerImpl::get_tokenizer() {
    OPENVINO_THROW("ParaformerImpl: get_tokenizer() not supported. "
                   "Paraformer uses internal detokenizer, not Tokenizer class. "
                   "Use the transcription results directly.");
}

ASRGenerationConfig ParaformerImpl::get_generation_config() const {
    return m_generation_config;
}

void ParaformerImpl::set_generation_config(const ASRGenerationConfig& config) {
    m_generation_config = config;
}

}  // namespace genai
}  // namespace ov